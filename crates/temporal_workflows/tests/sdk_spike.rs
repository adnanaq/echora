//! Temporal SDK spike and workflow integration tests for ECHO-95.

use std::time::{SystemTime, UNIX_EPOCH};

use futures::StreamExt as _;
use temporal_workflows::{
    AGENT_SERVICE_TASK_QUEUE, PROVIDER_QUEUE_NAMES, REFRESH_TRIGGER_SEARCH_ATTRIBUTE, SpikeInput,
    SpikeWorkflow, UPDATE_SERVICE_TASK_QUEUE, UpdateServicePorts, WorkerQueueSpec,
    build_worker_options, provider_task_queue,
};
use temporalio_client::{
    Client, ClientOptions, Connection, ConnectionOptions, WorkflowGetResultOptions,
    WorkflowIdConflictPolicy, WorkflowListOptions, WorkflowSignalOptions, WorkflowStartOptions,
};
use temporalio_common::Priority;
use temporalio_common::search_attributes::{SearchAttributeKey, SearchAttributes};
use temporalio_common::worker::WorkerDeploymentVersion;
use temporalio_sdk::Worker;
use temporalio_sdk::runtime::{Runtime, RuntimeOptions};
use temporalio_sdk::testing::WorkflowEnvironment;
use url::Url;
use uuid::{NoContext, Timestamp, Uuid};

#[test]
fn task_queue_names_and_worker_options_match_specification() {
    assert_eq!(UPDATE_SERVICE_TASK_QUEUE, "update-service");
    assert_eq!(AGENT_SERVICE_TASK_QUEUE, "agent-service");
    assert_eq!(REFRESH_TRIGGER_SEARCH_ATTRIBUTE, "RefreshTrigger");
    assert_eq!(provider_task_queue("anidb"), "provider-anidb");
    assert_eq!(PROVIDER_QUEUE_NAMES.len(), 8);

    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("0.1.0".to_owned())
        .build();
    let options = build_worker_options(
        &WorkerQueueSpec::UpdateService,
        deployment.clone(),
        true,
        UpdateServicePorts::noop(),
    )
    .expect("build update-service worker options");
    assert_eq!(options.task_queue, UPDATE_SERVICE_TASK_QUEUE);
    assert_eq!(options.deployment_options.version, deployment);
    assert!(options.deployment_options.use_worker_versioning);

    let provider_opts = build_worker_options(
        &WorkerQueueSpec::Provider("anidb".to_owned()),
        deployment.clone(),
        false,
        UpdateServicePorts::noop(),
    )
    .expect("build provider worker options");
    assert_eq!(provider_opts.task_queue, "provider-anidb");
    assert!(!provider_opts.deployment_options.use_worker_versioning);

    let agent_opts = build_worker_options(
        &WorkerQueueSpec::AgentService,
        deployment,
        false,
        UpdateServicePorts::noop(),
    )
    .expect("build agent worker options");
    assert_eq!(agent_opts.task_queue, AGENT_SERVICE_TASK_QUEUE);
}

fn build_spike_start_options(workflow_id: &str) -> WorkflowStartOptions {
    let trigger_key = SearchAttributeKey::keyword(REFRESH_TRIGGER_SEARCH_ATTRIBUTE);
    let search_attrs = SearchAttributes::new([trigger_key.value_set("user".to_owned())]);
    let mut priority = Priority::default();
    priority.priority_key = Some(1);
    priority.fairness_key = Some("user".to_owned());
    priority.fairness_weight = Some(1.0);

    WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id)
        .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
        .priority(priority)
        .search_attributes(search_attrs)
        .build()
}

#[tokio::test]
async fn sdk_spike_proves_conflict_policy_signals_priority_and_search_attributes()
-> anyhow::Result<()> {
    let temporal_addr =
        std::env::var("TEMPORAL_ADDRESS").unwrap_or_else(|_| "http://localhost:7233".to_owned());
    let connection =
        Connection::connect(ConnectionOptions::new(Url::parse(&temporal_addr)?).build()).await?;
    let client = Client::new(connection, ClientOptions::new("echora").build())?;
    let env = WorkflowEnvironment::from_client(client);
    let client = env.client().clone();

    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())
        .map_err(|err| anyhow::anyhow!("failed to initialize runtime: {err}"))?;
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("spike-1".to_owned())
        .build();
    let worker_options = build_worker_options(
        &WorkerQueueSpec::UpdateService,
        deployment,
        false,
        UpdateServicePorts::noop(),
    )
    .map_err(|err| anyhow::anyhow!("failed to build worker options: {err}"))?;
    let mut worker = Worker::new(&runtime, client.clone(), worker_options)
        .map_err(|err| anyhow::anyhow!("failed to create worker: {err}"))?;

    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    let workflow_id = format!("spike-{}", Uuid::new_v7(ts));
    let signal_start_id = format!("spike-sws-{}", Uuid::new_v7(ts));
    let start_opts = build_spike_start_options(&workflow_id);
    let sws_opts = build_spike_start_options(&signal_start_id);

    let client_future = async {
        // 1. USE_EXISTING conflict policy + signal + activity execution.
        let handle_1 = client
            .start_workflow(
                SpikeWorkflow::run,
                SpikeInput {
                    label: "spike".to_owned(),
                },
                start_opts.clone(),
            )
            .await?;
        let handle_2 = client
            .start_workflow(
                SpikeWorkflow::run,
                SpikeInput {
                    label: "spike".to_owned(),
                },
                start_opts,
            )
            .await?;
        assert_eq!(handle_1.info().workflow_id, handle_2.info().workflow_id);

        handle_1
            .signal(
                SpikeWorkflow::finish,
                "done".to_owned(),
                WorkflowSignalOptions::default(),
            )
            .await?;
        let output = handle_1
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        assert_eq!(output, "spike:done");

        // 2. signal_with_start_workflow starts and signals atomically.
        let sws_handle = client
            .signal_with_start_workflow(
                SpikeWorkflow::run,
                SpikeInput {
                    label: "sws".to_owned(),
                },
                SpikeWorkflow::finish,
                "ready".to_owned(),
                sws_opts,
            )
            .await?;
        let sws_output = sws_handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        assert_eq!(sws_output, "sws:ready");

        // 3. Visibility query by RefreshTrigger search attribute.
        let mut listed = client.list_workflows(
            format!("{REFRESH_TRIGGER_SEARCH_ATTRIBUTE} = 'user'"),
            WorkflowListOptions::default(),
        );
        let mut count = 0usize;
        while let Some(entry) = listed.next().await {
            let _ = entry?;
            count += 1;
        }
        assert!(count >= 2);
        Ok::<(), anyhow::Error>(())
    };

    tokio::select! {
        res = worker.run() => {
            res.map_err(|err| anyhow::anyhow!("worker failed: {err}"))?;
        }
        res = client_future => {
            res?;
        }
    }

    Ok(())
}
