//! Tests for task queue names, worker options, and the ECHO-95 Temporal SDK spike.

use std::collections::HashSet;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, bail, ensure};
use futures::StreamExt as _;
use temporal_workflows::testing::start_local_environment;
use temporal_workflows::{
    AGENT_SERVICE_TASK_QUEUE, DEFAULT_TEMPORAL_NAMESPACE, Provider,
    REFRESH_TRIGGER_SEARCH_ATTRIBUTE, SpikeInput, SpikeWorkflow, UPDATE_SERVICE_TASK_QUEUE,
    WorkerQueueSpec, build_spike_worker_options, build_worker_options,
};
use temporalio_client::schedules::{
    CreateScheduleOptions, DeleteScheduleOptions, PauseScheduleOptions, ScheduleAction,
    ScheduleDescription, ScheduleHandle, ScheduleOverlapPolicy, ScheduleSpec,
};
use temporalio_client::tonic::Request;
use temporalio_client::{
    Client, RpcOptions, WorkflowCountOptions, WorkflowDescribeOptions, WorkflowGetResultOptions,
    WorkflowIdConflictPolicy, WorkflowListOptions, WorkflowSignalOptions, WorkflowStartOptions,
};
use temporalio_common::Priority;
use temporalio_common::protos::temporal::api::enums::v1::VersioningBehavior as ProtoVersioningBehavior;
use temporalio_common::protos::temporal::api::workflowservice::v1::SetWorkerDeploymentCurrentVersionRequest;
use temporalio_common::search_attributes::{SearchAttributeKey, SearchAttributes};
use temporalio_common::worker::{VersioningBehavior, WorkerDeploymentVersion};
use temporalio_sdk::Worker;
use temporalio_sdk::runtime::{Runtime, RuntimeOptions};
use uuid::{NoContext, Timestamp, Uuid};

const DEPLOYMENT_NAME: &str = "update-service";
const SERVER_WAIT: Duration = Duration::from_secs(30);
const RETRY_INTERVAL: Duration = Duration::from_millis(200);
const SCHEDULE_INTERVAL: Duration = Duration::from_secs(2);

fn build_deployment_version(build_id: &str) -> WorkerDeploymentVersion {
    WorkerDeploymentVersion::builder()
        .deployment_name(DEPLOYMENT_NAME.to_owned())
        .build_id(build_id.to_owned())
        .build()
}

fn build_unique_workflow_id(prefix: &str) -> anyhow::Result<String> {
    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let timestamp = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    Ok(format!("{prefix}-{}", Uuid::new_v7(timestamp)))
}

fn build_spike_start_options(workflow_id: &str, trigger: &str) -> WorkflowStartOptions {
    let trigger_key = SearchAttributeKey::keyword(REFRESH_TRIGGER_SEARCH_ATTRIBUTE);
    let search_attributes = SearchAttributes::new([trigger_key.value_set(trigger.to_owned())]);
    let mut priority = Priority::default();
    priority.priority_key = Some(1);
    priority.fairness_key = Some(trigger.to_owned());
    priority.fairness_weight = Some(2.0);

    WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id)
        .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
        .priority(priority)
        .search_attributes(search_attributes)
        .build()
}

fn build_spike_input(label: &str) -> SpikeInput {
    SpikeInput {
        label: label.to_owned(),
    }
}

async fn count_workflows_eventually(
    client: &Client,
    query: &str,
    expected_count: usize,
) -> anyhow::Result<usize> {
    let deadline = Instant::now() + SERVER_WAIT;
    loop {
        let workflow_count = client
            .count_workflows(query, WorkflowCountOptions::default())
            .await?
            .count();
        if workflow_count >= expected_count || Instant::now() >= deadline {
            return Ok(workflow_count);
        }
        tokio::time::sleep(RETRY_INTERVAL).await;
    }
}

async fn set_current_deployment_version(client: &Client, build_id: &str) -> anyhow::Result<()> {
    let deadline = Instant::now() + SERVER_WAIT;
    loop {
        let request = SetWorkerDeploymentCurrentVersionRequest {
            namespace: DEFAULT_TEMPORAL_NAMESPACE.to_owned(),
            deployment_name: DEPLOYMENT_NAME.to_owned(),
            build_id: build_id.to_owned(),
            identity: "sdk-spike-test".to_owned(),
            ..SetWorkerDeploymentCurrentVersionRequest::default()
        };
        let set_outcome = client
            .connection()
            .workflow_service()
            .set_worker_deployment_current_version(Request::new(request))
            .await;
        match set_outcome {
            Ok(_) => return Ok(()),
            Err(_) if Instant::now() < deadline => tokio::time::sleep(RETRY_INTERVAL).await,
            Err(status) => bail!("deployment version {build_id} never became current: {status}"),
        }
    }
}

async fn check_refresh_trigger_visibility(
    client: &Client,
    user_workflow_id: &str,
) -> anyhow::Result<()> {
    let user_query = format!("{REFRESH_TRIGGER_SEARCH_ATTRIBUTE} = 'user'");
    let user_count = count_workflows_eventually(client, &user_query, 1).await?;
    ensure!(
        user_count == 1,
        "expected one user refresh, counted {user_count}"
    );
    let sweep_query = format!("{REFRESH_TRIGGER_SEARCH_ATTRIBUTE} = 'sweep'");
    let sweep_count = count_workflows_eventually(client, &sweep_query, 1).await?;
    ensure!(
        sweep_count == 1,
        "expected one sweep refresh, counted {sweep_count}"
    );

    let mut listed_workflows =
        client.list_workflows(user_query.clone(), WorkflowListOptions::default());
    let mut listed_ids = Vec::new();
    while let Some(listed_workflow) = listed_workflows.next().await {
        listed_ids.push(listed_workflow?.id().to_owned());
    }
    ensure!(listed_ids == vec![user_workflow_id.to_owned()]);
    Ok(())
}

async fn wait_for_schedule(
    schedule: &ScheduleHandle<Client>,
    is_reached: impl Fn(&ScheduleDescription) -> bool,
) -> anyhow::Result<ScheduleDescription> {
    let deadline = Instant::now() + SERVER_WAIT;
    loop {
        let description = schedule.describe(RpcOptions::default()).await?;
        if is_reached(&description) {
            return Ok(description);
        }
        if Instant::now() >= deadline {
            bail!(
                "schedule never reached the expected state: {} actions, {} skipped",
                description.action_count(),
                description.overlap_skipped()
            );
        }
        tokio::time::sleep(RETRY_INTERVAL).await;
    }
}

const fn provider_declaration_position(provider: Provider) -> usize {
    match provider {
        Provider::AniDb => 0,
        Provider::AniList => 1,
        Provider::AnimePlanet => 2,
        Provider::AnimeSchedule => 3,
        Provider::AniSearch => 4,
        Provider::Kitsu => 5,
        Provider::Mal => 6,
        Provider::Syoboi => 7,
    }
}

#[test]
fn provider_all_lists_every_provider_once_in_declaration_order() {
    for (position, provider) in Provider::ALL.into_iter().enumerate() {
        assert_eq!(provider_declaration_position(provider), position);
    }
    let distinct_providers: HashSet<Provider> = Provider::ALL.into_iter().collect();
    assert_eq!(distinct_providers.len(), Provider::ALL.len());
}

#[test]
fn provider_task_queue_every_provider_prefixes_own_slug() {
    for provider in Provider::ALL {
        assert_eq!(
            provider.task_queue(),
            format!("provider-{}", provider.slug())
        );
        assert_eq!(provider.to_string(), provider.slug());
    }
}

#[test]
fn provider_slug_every_provider_distinct_lowercase_ascii() {
    let slugs: HashSet<&str> = Provider::ALL.into_iter().map(Provider::slug).collect();
    assert_eq!(slugs.len(), Provider::ALL.len());
    for slug in slugs {
        assert_ne!(slug, "");
        assert!(
            slug.chars()
                .all(|character| character.is_ascii_lowercase() || character == '_'),
            "slug {slug:?} must be lowercase ASCII letters or underscores"
        );
    }
}

#[test]
fn worker_queue_spec_task_queue_every_spec_distinct() {
    let specs: Vec<WorkerQueueSpec> = [
        WorkerQueueSpec::UpdateService,
        WorkerQueueSpec::AgentService,
    ]
    .into_iter()
    .chain(Provider::ALL.into_iter().map(WorkerQueueSpec::Provider))
    .collect();
    let task_queues: HashSet<&str> = specs.iter().map(|spec| spec.task_queue()).collect();
    assert_eq!(task_queues.len(), specs.len());
    assert!(task_queues.contains(UPDATE_SERVICE_TASK_QUEUE));
    assert!(task_queues.contains(AGENT_SERVICE_TASK_QUEUE));
    for spec in specs {
        assert_eq!(spec.to_string(), spec.task_queue());
    }
}

#[test]
fn build_worker_options_no_behavior_leaves_worker_unversioned() {
    let deployment_version = build_deployment_version("0.1.0");
    let options = build_worker_options(
        WorkerQueueSpec::Provider(Provider::AniDb),
        deployment_version.clone(),
        None,
    )
    .build();
    assert_eq!(options.task_queue, "provider-anidb");
    assert_eq!(options.deployment_options.version, deployment_version);
    assert!(!options.deployment_options.use_worker_versioning);
    assert_eq!(options.deployment_options.default_versioning_behavior, None);
}

#[test]
fn build_worker_options_behavior_enables_versioning_with_that_default() {
    for behavior in [VersioningBehavior::Pinned, VersioningBehavior::AutoUpgrade] {
        let options = build_worker_options(
            WorkerQueueSpec::UpdateService,
            build_deployment_version("0.1.0"),
            Some(behavior),
        )
        .build();
        assert_eq!(options.task_queue, UPDATE_SERVICE_TASK_QUEUE);
        assert!(options.deployment_options.use_worker_versioning);
        assert_eq!(
            options.deployment_options.default_versioning_behavior,
            Some(behavior)
        );
    }
}

#[test]
fn build_spike_worker_options_every_queue_and_behavior_builds() -> anyhow::Result<()> {
    let specs = [
        WorkerQueueSpec::UpdateService,
        WorkerQueueSpec::AgentService,
    ]
    .into_iter()
    .chain(Provider::ALL.into_iter().map(WorkerQueueSpec::Provider));
    for spec in specs {
        for behavior in [
            None,
            Some(VersioningBehavior::Pinned),
            Some(VersioningBehavior::AutoUpgrade),
        ] {
            let options =
                build_spike_worker_options(spec, build_deployment_version("0.1.0"), behavior)?;
            assert_eq!(options.task_queue, spec.task_queue());
        }
    }
    Ok(())
}

#[tokio::test]
async fn worker_new_spike_options_every_versioning_choice_accepted() -> anyhow::Result<()> {
    let environment = start_local_environment(None).await?;
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    for (build_id, behavior) in [
        ("unversioned", None),
        ("pinned", Some(VersioningBehavior::Pinned)),
        ("auto-upgrade", Some(VersioningBehavior::AutoUpgrade)),
    ] {
        let options = build_spike_worker_options(
            WorkerQueueSpec::UpdateService,
            build_deployment_version(build_id),
            behavior,
        )?;
        Worker::new(&runtime, environment.client().clone(), options)
            .with_context(|| format!("worker rejected versioning behavior {behavior:?}"))?;
    }
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn spike_workflow_unversioned_worker_proves_conflict_policy_signals_and_visibility()
-> anyhow::Result<()> {
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let worker_options = build_spike_worker_options(
        WorkerQueueSpec::UpdateService,
        build_deployment_version("0.1.0"),
        None,
    )?;
    let mut worker = Worker::new(&runtime, client.clone(), worker_options)?;
    let shutdown_worker = worker.shutdown_handle();

    let workflow_id = build_unique_workflow_id("spike")?;
    let signal_start_id = build_unique_workflow_id("spike-sws")?;
    let first_handle = client
        .start_workflow(
            SpikeWorkflow::run,
            build_spike_input("spike"),
            build_spike_start_options(&workflow_id, "user"),
        )
        .await?;
    let second_handle = client
        .start_workflow(
            SpikeWorkflow::run,
            build_spike_input("ignored"),
            build_spike_start_options(&workflow_id, "user"),
        )
        .await?;
    ensure!(first_handle.info().workflow_id == second_handle.info().workflow_id);
    ensure!(first_handle.info().run_id == second_handle.info().run_id);

    first_handle
        .signal(
            SpikeWorkflow::finish,
            "done".to_owned(),
            WorkflowSignalOptions::default(),
        )
        .await?;
    first_handle
        .signal(
            SpikeWorkflow::finish,
            "duplicate".to_owned(),
            WorkflowSignalOptions::default(),
        )
        .await?;
    let client_checks = async {
        let spike_output = first_handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        ensure!(
            spike_output == "spike:done",
            "unexpected output {spike_output:?}"
        );

        let description = first_handle
            .describe(WorkflowDescribeOptions::default())
            .await?;
        let execution_info = description
            .raw_description
            .workflow_execution_info
            .context("description lacks execution info")?;
        let priority = execution_info.priority.context("priority missing")?;
        ensure!(priority.priority_key == 1);
        ensure!(priority.fairness_key == "user");

        let signal_start_handle = client
            .signal_with_start_workflow(
                SpikeWorkflow::run,
                build_spike_input("sws"),
                SpikeWorkflow::finish,
                "ready".to_owned(),
                build_spike_start_options(&signal_start_id, "sweep"),
            )
            .await?;
        let signal_start_output = signal_start_handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        ensure!(signal_start_output == "sws:ready");

        check_refresh_trigger_visibility(&client, &workflow_id).await?;
        Ok::<(), anyhow::Error>(())
    };

    let (worker_outcome, client_outcome) = tokio::join!(worker.run(), async {
        let client_outcome = client_checks.await;
        shutdown_worker();
        client_outcome
    });
    client_outcome?;
    worker_outcome?;
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn spike_workflow_versioned_worker_runs_on_current_version_with_behavior()
-> anyhow::Result<()> {
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let worker_options = build_spike_worker_options(
        WorkerQueueSpec::UpdateService,
        build_deployment_version("spike-v1"),
        Some(VersioningBehavior::AutoUpgrade),
    )?;
    let mut worker = Worker::new(&runtime, client.clone(), worker_options)?;
    let shutdown_worker = worker.shutdown_handle();

    let workflow_id = build_unique_workflow_id("spike-versioned")?;
    let client_checks = async {
        set_current_deployment_version(&client, "spike-v1").await?;
        let handle = client
            .signal_with_start_workflow(
                SpikeWorkflow::run,
                build_spike_input("versioned"),
                SpikeWorkflow::finish,
                "ok".to_owned(),
                build_spike_start_options(&workflow_id, "episode"),
            )
            .await?;
        let versioned_output = handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        ensure!(versioned_output == "versioned:ok");

        let description = handle.describe(WorkflowDescribeOptions::default()).await?;
        let versioning_info = description
            .raw_description
            .workflow_execution_info
            .and_then(|execution_info| execution_info.versioning_info)
            .context("versioned workflow lacks versioning info")?;
        ensure!(versioning_info.behavior == i32::from(ProtoVersioningBehavior::AutoUpgrade));
        let deployment_version = versioning_info
            .deployment_version
            .context("versioned workflow lacks deployment version")?;
        ensure!(deployment_version.deployment_name == DEPLOYMENT_NAME);
        ensure!(deployment_version.build_id == "spike-v1");
        Ok::<(), anyhow::Error>(())
    };

    let (worker_outcome, client_outcome) = tokio::join!(worker.run(), async {
        let client_outcome = client_checks.await;
        shutdown_worker();
        client_outcome
    });
    client_outcome?;
    worker_outcome?;
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn spike_schedule_skip_overlap_skips_runs_while_previous_run_open() -> anyhow::Result<()> {
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let worker_options = build_spike_worker_options(
        WorkerQueueSpec::UpdateService,
        build_deployment_version("0.1.0"),
        None,
    )?;
    let mut worker = Worker::new(&runtime, client.clone(), worker_options)?;
    let shutdown_worker = worker.shutdown_handle();

    let client_checks = async {
        let schedule = client
            .create_schedule(
                "spike-schedule",
                CreateScheduleOptions::builder()
                    .action(ScheduleAction::start_workflow(
                        SpikeWorkflow::run,
                        build_spike_input("scheduled"),
                        UPDATE_SERVICE_TASK_QUEUE,
                        "spike-scheduled",
                    ))
                    .spec(ScheduleSpec::from_interval(SCHEDULE_INTERVAL))
                    .overlap_policy(ScheduleOverlapPolicy::Skip)
                    .build(),
            )
            .await?;
        let after_skip =
            wait_for_schedule(&schedule, |description| description.overlap_skipped() >= 1).await?;
        ensure!(
            after_skip.action_count() == 1,
            "a scheduled run started while the previous one was open: {:?}",
            after_skip.raw().info
        );
        let scheduled_workflow_id = after_skip
            .recent_actions()
            .first()
            .map(|recent_action| recent_action.workflow_id.clone())
            .context("schedule started no workflow")?;

        schedule
            .pause(Some("spike test finished"), PauseScheduleOptions::default())
            .await?;
        let scheduled_handle = client.get_workflow_handle::<SpikeWorkflow>(scheduled_workflow_id);
        scheduled_handle
            .signal(
                SpikeWorkflow::finish,
                "done".to_owned(),
                WorkflowSignalOptions::default(),
            )
            .await?;
        let scheduled_output = scheduled_handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        ensure!(scheduled_output == "scheduled:done");

        schedule.delete(DeleteScheduleOptions::default()).await?;
        Ok::<(), anyhow::Error>(())
    };

    let (worker_outcome, client_outcome) = tokio::join!(worker.run(), async {
        let client_outcome = client_checks.await;
        shutdown_worker();
        client_outcome
    });
    client_outcome?;
    worker_outcome?;
    environment.shutdown().await?;
    Ok(())
}
