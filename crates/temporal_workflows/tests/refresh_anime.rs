//! Unit and local-server integration tests for `RefreshAnimeWorkflow`, activities, and ports.

#![expect(
    clippy::expect_used,
    reason = "integration tests unwrap setup locks and assertions"
)]

use std::any::Any;
use std::collections::BTreeMap;
use std::str::FromStr as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::ensure;
use futures::future::BoxFuture;
use temporal_workflows::testing::start_local_environment;
use temporal_workflows::{
    CatalogueDataPort, EntityPageSummary, EntityType, FetchRequestedSignal, ListEntityPagesOutput,
    MAX_ABORTED_RETRIES_PER_ATTEMPT, MAX_PAGE_RETRY_BACKOFF_HOURS, MergeAnimeInput,
    MergeAnimeOutput, MergeAttemptOutcome, PortError, Provider, ProviderActivities,
    RecordRefreshInput, RefreshAnimeInput, RefreshAnimeOutcome, RefreshAnimeWorkflow,
    RefreshOutcome, RefreshPageInput, RefreshPageKey, RefreshPageOutput, RefreshTrigger,
    StartEntityWorkflowsInput, UPDATE_SERVICE_TASK_QUEUE, UpdateServiceActivities,
    UpdateServicePorts, WorkerQueueSpec, WorkflowStarterPort, build_priority,
    build_refresh_anime_workflow_id, build_update_service_worker_options,
    compute_page_backoff_seconds, is_page_due, matches_fetch_request, merge_priority,
    select_pages_to_refresh,
};
use temporalio_client::{
    WorkflowGetResultOptions, WorkflowIdConflictPolicy, WorkflowSignalOptions, WorkflowStartOptions,
};
use temporalio_common::RetryPolicy;
use temporalio_common::worker::WorkerDeploymentVersion;
use temporalio_sdk::Worker;
use temporalio_sdk::activities::ActivityError;
use temporalio_sdk::runtime::{Runtime, RuntimeOptions};
use temporalio_sdk::testing::{
    ActivityEnvironment, ActivityEnvironmentError, TestActivityInfoOptions,
};
use tokio::sync::watch;
use uuid::{NoContext, Timestamp, Uuid};

const TEST_WAIT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct FakeState {
    pages: ListEntityPagesOutput,
    aborted_remaining: usize,
    merge_fail_non_retryable: bool,
    page_delays: BTreeMap<Provider, Duration>,
    page_errors: BTreeMap<Provider, PageErrorMode>,
    page_changes: BTreeMap<Provider, bool>,
    fetch_calls: Vec<(RefreshPageKey, i64)>,
    recorded_page_failures: Vec<RefreshPageKey>,
    merge_calls: usize,
    started_batches: Vec<StartEntityWorkflowsInput>,
    recorded_outcomes: Vec<RecordRefreshInput>,
}

#[derive(Clone, Copy)]
enum PageErrorMode {
    Transient,
    Blocked,
    NonRetryable,
}

#[derive(Clone, Default)]
struct FakePorts {
    state: Arc<Mutex<FakeState>>,
}

impl CatalogueDataPort for FakePorts {
    fn list_entity_pages(
        &self,
        _anime_id: String,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>> {
        let output = self.state.lock().expect("lock").pages.clone();
        Box::pin(async move { Ok(output) })
    }

    fn fetch_and_store_page(
        &self,
        page_key: RefreshPageKey,
        max_age_seconds: i64,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>> {
        let state = self.state.clone();
        Box::pin(async move {
            let (delay, error_mode, has_changed) = {
                let mut guard = state.lock().expect("lock");
                guard.fetch_calls.push((page_key.clone(), max_age_seconds));
                let delay = guard
                    .page_delays
                    .get(&page_key.provider)
                    .copied()
                    .unwrap_or(Duration::ZERO);
                let error_mode = guard.page_errors.get(&page_key.provider).copied();
                let has_changed = guard
                    .page_changes
                    .get(&page_key.provider)
                    .copied()
                    .unwrap_or(true);
                (delay, error_mode, has_changed)
            };
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            match error_mode {
                Some(PageErrorMode::Transient) => {
                    Err(PortError::retryable("provider temporarily unavailable"))
                }
                Some(PageErrorMode::Blocked) => {
                    Err(PortError::provider_blocked("provider blocked IP"))
                }
                Some(PageErrorMode::NonRetryable) => {
                    Err(PortError::non_retryable("invalid page address"))
                }
                None => Ok(RefreshPageOutput {
                    page_key,
                    has_changed,
                    has_failed: false,
                }),
            }
        })
    }

    fn record_page_failure(
        &self,
        page_key: RefreshPageKey,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        self.state
            .lock()
            .expect("lock")
            .recorded_page_failures
            .push(page_key);
        Box::pin(async { Ok(()) })
    }

    fn merge_anime_attempt(
        &self,
        _anime_id: String,
        _merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>> {
        let mut guard = self.state.lock().expect("lock");
        guard.merge_calls += 1;
        if guard.merge_fail_non_retryable {
            return Box::pin(async { Err(PortError::non_retryable("invalid anime_id in merge")) });
        }
        if guard.aborted_remaining > 0 {
            guard.aborted_remaining -= 1;
            return Box::pin(async {
                Ok(MergeAttemptOutcome::VersionConflict(
                    "concurrent save".to_owned(),
                ))
            });
        }
        Box::pin(async {
            Ok(MergeAttemptOutcome::Saved(MergeAnimeOutput {
                has_changed: true,
                changed_episode_ids: vec!["ep-1".to_owned()],
                has_changed_synopses: true,
            }))
        })
    }

    fn record_refresh(&self, input: RecordRefreshInput) -> BoxFuture<'_, Result<(), PortError>> {
        self.state
            .lock()
            .expect("lock")
            .recorded_outcomes
            .push(input);
        Box::pin(async { Ok(()) })
    }
}

impl WorkflowStarterPort for FakePorts {
    fn start_entity_workflows(
        &self,
        input: StartEntityWorkflowsInput,
    ) -> BoxFuture<'_, Result<usize, PortError>> {
        let count = input.episode_ids.len() + usize::from(input.start_synopsis);
        self.state.lock().expect("lock").started_batches.push(input);
        Box::pin(async move { Ok(count) })
    }
}

fn build_unique_anime_id() -> anyhow::Result<String> {
    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    Ok(Uuid::new_v7(ts).to_string())
}

fn build_page_summary(
    provider: Provider,
    address: &str,
    is_partial_source: bool,
    fetched_at_epoch_seconds: Option<i64>,
) -> EntityPageSummary {
    EntityPageSummary {
        page_key: RefreshPageKey {
            provider,
            entity_type: EntityType::Anime,
            address: address.to_owned(),
        },
        is_partial_source,
        fetched_at_epoch_seconds,
        is_dead: false,
        fetch_failed_at_epoch_seconds: None,
        fetch_failures: 0,
    }
}

fn build_deployment(build_id: &str) -> WorkerDeploymentVersion {
    WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id(build_id.to_owned())
        .build()
}

#[test]
fn scheduling_helpers_compute_backoff_priority_and_parse_slugs() {
    assert_eq!(compute_page_backoff_seconds(0), 0);
    assert_eq!(compute_page_backoff_seconds(-2), 0);
    assert_eq!(compute_page_backoff_seconds(1), 2 * 3_600);
    assert_eq!(compute_page_backoff_seconds(2), 4 * 3_600);
    assert_eq!(
        compute_page_backoff_seconds(20),
        MAX_PAGE_RETRY_BACKOFF_HOURS * 3_600
    );

    assert_eq!(merge_priority(0, 2), 2);
    assert_eq!(merge_priority(2, 0), 2);
    assert_eq!(merge_priority(3, 1), 1);
    assert_eq!(merge_priority(1, 3), 1);

    let remerge_priority = build_priority(0, RefreshTrigger::Remerge);
    assert_eq!(remerge_priority.priority_key, Some(4));
    assert_eq!(remerge_priority.fairness_key.as_deref(), Some("remerge"));

    for provider in Provider::ALL {
        assert_eq!(Provider::from_str(provider.slug()).ok(), Some(provider));
    }
    assert!(Provider::from_str("unknown").is_err());

    for trigger in [
        RefreshTrigger::User,
        RefreshTrigger::Episode,
        RefreshTrigger::Review,
        RefreshTrigger::Sweep,
        RefreshTrigger::Seed,
        RefreshTrigger::Remerge,
    ] {
        assert_eq!(
            RefreshTrigger::from_str(trigger.as_str()).ok(),
            Some(trigger)
        );
    }
    assert!(RefreshTrigger::from_str("unknown").is_err());
}

#[test]
fn scheduling_helpers_select_due_and_requested_pages() {
    let now = 1_700_000_000_i64;
    let partial = build_page_summary(Provider::Kitsu, "1", true, None);
    assert!(!is_page_due(&partial, now, 7));
    assert!(!matches_fetch_request(
        &partial,
        &FetchRequestedSignal::default(),
        now
    ));

    let dead = EntityPageSummary {
        is_partial_source: false,
        is_dead: true,
        ..partial.clone()
    };
    assert!(!is_page_due(&dead, now, 7));

    let in_backoff = EntityPageSummary {
        is_dead: false,
        fetch_failed_at_epoch_seconds: Some(now - 60),
        fetch_failures: 1,
        ..dead.clone()
    };
    assert!(!is_page_due(&in_backoff, now, 7));

    let past_backoff = EntityPageSummary {
        fetch_failed_at_epoch_seconds: Some(now - 10_000),
        fetch_failures: 1,
        ..in_backoff
    };
    assert!(is_page_due(&past_backoff, now, 7));

    let fresh = EntityPageSummary {
        fetched_at_epoch_seconds: Some(now - 100),
        fetch_failed_at_epoch_seconds: None,
        fetch_failures: 0,
        ..past_backoff
    };
    assert!(!is_page_due(&fresh, now, 7));
    assert!(matches_fetch_request(
        &fresh,
        &FetchRequestedSignal {
            providers: vec![Provider::Kitsu],
            max_age_seconds: 0,
            priority: 1,
        },
        now
    ));
    assert!(!matches_fetch_request(
        &fresh,
        &FetchRequestedSignal {
            providers: vec![Provider::Mal],
            max_age_seconds: 0,
            priority: 1,
        },
        now
    ));
    assert!(!matches_fetch_request(
        &fresh,
        &FetchRequestedSignal {
            providers: Vec::new(),
            max_age_seconds: 500,
            priority: 1,
        },
        now
    ));

    let selected = select_pages_to_refresh(
        &[fresh],
        now,
        7,
        &[
            FetchRequestedSignal {
                providers: vec![Provider::Kitsu],
                max_age_seconds: 50,
                priority: 3,
            },
            FetchRequestedSignal {
                providers: vec![Provider::Kitsu],
                max_age_seconds: 0,
                priority: 1,
            },
        ],
        false,
    );
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].max_age_seconds, 0);
}

#[tokio::test]
async fn refresh_page_transient_error_retries_until_final_attempt_then_records_failure() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_errors
        .insert(Provider::Mal, PageErrorMode::Transient);
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let page_input = RefreshPageInput {
        page_key: RefreshPageKey {
            provider: Provider::Mal,
            entity_type: EntityType::Anime,
            address: "21".to_owned(),
        },
        max_age_seconds: 0,
    };

    let attempt_one_info = TestActivityInfoOptions::builder()
        .attempt(1)
        .retry_policy(RetryPolicy::builder().maximum_attempts(5).build())
        .build();
    let env_attempt_one = ActivityEnvironment::builder()
        .info(attempt_one_info)
        .register_activities(ProviderActivities::new(ports.clone()))
        .build();
    let attempt_one_err = env_attempt_one
        .run(ProviderActivities::refresh_page, page_input.clone())
        .await
        .expect_err("attempt 1 must return retryable error");
    assert!(matches!(
        attempt_one_err,
        ActivityEnvironmentError::Activity(ActivityError::Application(ref failure))
            if !failure.is_non_retryable()
    ));
    assert_eq!(
        fake.state
            .lock()
            .expect("lock")
            .recorded_page_failures
            .len(),
        0
    );

    let final_attempt_info = TestActivityInfoOptions::builder()
        .attempt(5)
        .retry_policy(RetryPolicy::builder().maximum_attempts(5).build())
        .build();
    let env_final = ActivityEnvironment::builder()
        .info(final_attempt_info)
        .register_activities(ProviderActivities::new(ports))
        .build();
    let final_output = env_final
        .run(ProviderActivities::refresh_page, page_input.clone())
        .await
        .expect("final attempt records failure and succeeds");
    assert!(final_output.has_failed);
    assert!(!final_output.has_changed);
    assert_eq!(
        fake.state.lock().expect("lock").recorded_page_failures,
        vec![page_input.page_key]
    );
}

#[tokio::test]
async fn refresh_page_provider_blocked_fails_fast_on_first_attempt() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_errors
        .insert(Provider::AniSearch, PageErrorMode::Blocked);
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let page_input = RefreshPageInput {
        page_key: RefreshPageKey {
            provider: Provider::AniSearch,
            entity_type: EntityType::Anime,
            address: "100".to_owned(),
        },
        max_age_seconds: 0,
    };
    let attempt_one_info = TestActivityInfoOptions::builder()
        .attempt(1)
        .retry_policy(RetryPolicy::builder().maximum_attempts(5).build())
        .build();
    let env = ActivityEnvironment::builder()
        .info(attempt_one_info)
        .register_activities(ProviderActivities::new(ports))
        .build();

    let output = env
        .run(ProviderActivities::refresh_page, page_input.clone())
        .await
        .expect("blocked provider must fail fast and record page failure");
    assert!(output.has_failed);
    assert!(!output.has_changed);
    assert_eq!(
        fake.state.lock().expect("lock").recorded_page_failures,
        vec![page_input.page_key]
    );
}

#[tokio::test]
async fn refresh_page_permanent_error_returns_non_retryable_failure() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_errors
        .insert(Provider::AniList, PageErrorMode::NonRetryable);
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let env = ActivityEnvironment::builder()
        .register_activities(ProviderActivities::new(ports))
        .build();
    let error = env
        .run(
            ProviderActivities::refresh_page,
            RefreshPageInput {
                page_key: RefreshPageKey {
                    provider: Provider::AniList,
                    entity_type: EntityType::Anime,
                    address: "bad".to_owned(),
                },
                max_age_seconds: 0,
            },
        )
        .await
        .expect_err("permanent error must surface as non-retryable");
    assert!(matches!(
        error,
        ActivityEnvironmentError::Activity(ActivityError::Application(ref failure))
            if failure.is_non_retryable()
    ));
}

#[tokio::test]
async fn refresh_page_emits_heartbeats_and_cancels_cleanly() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_delays
        .insert(Provider::Mal, Duration::from_secs(10));
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let heartbeats = Arc::new(Mutex::new(Vec::<String>::new()));
    let heartbeats_cb = heartbeats.clone();
    let env = Arc::new(
        ActivityEnvironment::builder()
            .on_heartbeat(move |detail: Box<dyn Any>| {
                if let Some(text) = detail.downcast_ref::<String>() {
                    heartbeats_cb.lock().expect("lock").push(text.clone());
                }
            })
            .register_activities(
                ProviderActivities::new(ports).with_heartbeat_interval(Duration::from_millis(15)),
            )
            .build(),
    );

    let env_for_cancel = env.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(75)).await;
        env_for_cancel.cancel();
    });

    let result = env
        .run(
            ProviderActivities::refresh_page,
            RefreshPageInput {
                page_key: RefreshPageKey {
                    provider: Provider::Mal,
                    entity_type: EntityType::Anime,
                    address: "42".to_owned(),
                },
                max_age_seconds: 0,
            },
        )
        .await;
    assert!(matches!(
        result,
        Err(ActivityEnvironmentError::Activity(
            ActivityError::Cancelled { .. }
        ))
    ));
    assert!(!heartbeats.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn merge_anime_exceeding_aborted_limit_returns_retryable_failure() {
    let fake = FakePorts::default();
    fake.state.lock().expect("lock").aborted_remaining = MAX_ABORTED_RETRIES_PER_ATTEMPT + 5;
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let env = ActivityEnvironment::builder()
        .register_activities(UpdateServiceActivities::new(ports))
        .build();
    let error = env
        .run(
            UpdateServiceActivities::merge_anime,
            MergeAnimeInput {
                anime_id: "anime-1".to_owned(),
                merge_version: 2,
            },
        )
        .await
        .expect_err("exceeding ABORTED limit must return retryable failure");
    assert!(matches!(
        error,
        ActivityEnvironmentError::Activity(ActivityError::Application(ref failure))
            if !failure.is_non_retryable()
    ));
    assert_eq!(
        fake.state.lock().expect("lock").merge_calls,
        MAX_ABORTED_RETRIES_PER_ATTEMPT
    );
}

#[tokio::test]
async fn noop_ports_reject_all_operations() {
    let ports = UpdateServicePorts::noop();
    let key = RefreshPageKey {
        provider: Provider::Mal,
        entity_type: EntityType::Anime,
        address: "1".to_owned(),
    };
    assert!(
        ports
            .catalogue()
            .list_entity_pages("x".to_owned())
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue()
            .fetch_and_store_page(key.clone(), 0)
            .await
            .is_err()
    );
    assert!(ports.catalogue().record_page_failure(key).await.is_err());
    assert!(
        ports
            .catalogue()
            .merge_anime_attempt("x".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue()
            .record_refresh(RecordRefreshInput {
                anime_id: "x".to_owned(),
                outcome: RefreshOutcome::Succeeded,
                refresh_after_days: 7,
            })
            .await
            .is_err()
    );
    assert!(
        ports
            .starter()
            .start_entity_workflows(StartEntityWorkflowsInput {
                anime_id: "x".to_owned(),
                episode_ids: Vec::new(),
                start_synopsis: false,
                trigger: RefreshTrigger::Sweep,
                priority: 3,
            })
            .await
            .is_err()
    );
}

fn build_test_worker(
    runtime: &Runtime,
    client: temporalio_client::Client,
    queue: WorkerQueueSpec,
    deployment: WorkerDeploymentVersion,
    ports: UpdateServicePorts,
) -> anyhow::Result<Worker> {
    let options = build_update_service_worker_options(
        queue,
        deployment,
        None,
        Some(Duration::from_secs(5)),
        ports,
    )?;
    Ok(Worker::new(runtime, client, options)?)
}

#[tokio::test]
async fn refresh_anime_workflow_refreshes_due_pages_retries_aborted_and_starts_downstream()
-> anyhow::Result<()> {
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();

    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.aborted_remaining = 2;
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::Kitsu, "2", true, None),
            ],
            entity_merge_version: Some(1),
        };
    }

    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let deployment = build_deployment("test-111");
    let update_worker = build_test_worker(
        &runtime,
        client.clone(),
        WorkerQueueSpec::UpdateService,
        deployment.clone(),
        ports.clone(),
    )?;
    let mal_worker = build_test_worker(
        &runtime,
        client.clone(),
        WorkerQueueSpec::Provider(Provider::Mal),
        deployment,
        ports,
    )?;

    let (stop_tx, stop_rx) = watch::channel(false);
    let mut stop_update = stop_rx.clone();
    let mut stop_mal = stop_rx;

    let anime_id = build_unique_anime_id()?;
    let workflow_id = build_refresh_anime_workflow_id(&anime_id);
    let start_options = WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id)
        .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
        .priority(build_priority(1, RefreshTrigger::User))
        .build();

    let test_run = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.clone(),
                    trigger: RefreshTrigger::User,
                    priority: 1,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    merge_coalesce_wait_seconds: Some(0),
                    pages_relinked: false,
                },
                start_options,
            )
            .await?;
        let outcome: RefreshAnimeOutcome = tokio::time::timeout(
            TEST_WAIT,
            handle.get_result(WorkflowGetResultOptions::default()),
        )
        .await??;
        stop_tx.send_replace(true);
        anyhow::Ok(outcome)
    };

    let (test_outcome, update_res, mal_res) = tokio::join!(
        test_run,
        update_worker.run_until(async move {
            let _ = stop_update.wait_for(|stopping| *stopping).await;
        }),
        mal_worker.run_until(async move {
            let _ = stop_mal.wait_for(|stopping| *stopping).await;
        }),
    );
    update_res?;
    mal_res?;
    let outcome = test_outcome?;

    ensure!(outcome.anime_id == anime_id);
    ensure!(outcome.pages_refreshed == 1);
    ensure!(outcome.pages_failed == 0);
    ensure!(outcome.merges_run == 1);
    ensure!(outcome.episode_workflows_started == 1);
    ensure!(outcome.synopsis_workflow_started);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);

    {
        let state = fake.state.lock().expect("lock");
        ensure!(state.merge_calls == 3);
        ensure!(state.recorded_outcomes.len() == 1);
        ensure!(state.recorded_outcomes[0].outcome == RefreshOutcome::Succeeded);
        ensure!(state.started_batches.len() == 2);
        ensure!(state.started_batches[0].episode_ids == vec!["ep-1"]);
        ensure!(!state.started_batches[0].start_synopsis);
        ensure!(state.started_batches[1].start_synopsis);
    }

    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_older_merge_version_merges_without_fetching() -> anyhow::Result<()>
{
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();

    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.pages = ListEntityPagesOutput {
            // Due page exists, but Remerge trigger must merge from stored pages without fetching.
            pages: vec![build_page_summary(Provider::Mal, "1", false, None)],
            entity_merge_version: Some(1),
        };
    }

    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_update_service_worker_options(
            WorkerQueueSpec::UpdateService,
            build_deployment("test-remerge"),
            None,
            None,
            ports,
        )?,
    )?;

    let (stop_tx, mut stop_rx) = watch::channel(false);
    let anime_id = build_unique_anime_id()?;
    let workflow_id = build_refresh_anime_workflow_id(&anime_id);

    let test_run = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.clone(),
                    trigger: RefreshTrigger::Remerge,
                    priority: 0,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    merge_coalesce_wait_seconds: Some(0),
                    pages_relinked: false,
                },
                WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id).build(),
            )
            .await?;
        let outcome: RefreshAnimeOutcome = tokio::time::timeout(
            TEST_WAIT,
            handle.get_result(WorkflowGetResultOptions::default()),
        )
        .await??;
        stop_tx.send_replace(true);
        anyhow::Ok(outcome)
    };

    let (test_outcome, worker_res) = tokio::join!(
        test_run,
        update_worker.run_until(async move {
            let _ = stop_rx.wait_for(|stopping| *stopping).await;
        }),
    );
    worker_res?;
    let outcome = test_outcome?;

    ensure!(outcome.pages_refreshed == 0);
    ensure!(outcome.merges_run == 1);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);
    ensure!(fake.state.lock().expect("lock").fetch_calls.is_empty());

    environment.shutdown().await?;
    Ok(())
}

async fn start_and_signal_early_merge(
    client: &temporalio_client::Client,
    anime_id: &str,
) -> anyhow::Result<RefreshAnimeOutcome> {
    let workflow_id = build_refresh_anime_workflow_id(anime_id);
    let handle = client
        .start_workflow(
            RefreshAnimeWorkflow::run,
            RefreshAnimeInput {
                anime_id: anime_id.to_owned(),
                trigger: RefreshTrigger::Sweep,
                priority: 3,
                refresh_after_days: 7,
                merge_version: 2,
                merge_first_wait_seconds: Some(1),
                merge_coalesce_wait_seconds: Some(1),
                pages_relinked: false,
            },
            WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id)
                .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
                .build(),
        )
        .await?;

    tokio::time::sleep(Duration::from_millis(1_300)).await;
    handle
        .signal(
            RefreshAnimeWorkflow::fetch_requested,
            FetchRequestedSignal {
                providers: vec![Provider::Mal],
                max_age_seconds: 0,
                priority: 1,
            },
            WorkflowSignalOptions::default(),
        )
        .await?;
    handle
        .signal(
            RefreshAnimeWorkflow::merge_requested,
            (),
            WorkflowSignalOptions::default(),
        )
        .await?;

    let outcome: RefreshAnimeOutcome = tokio::time::timeout(
        TEST_WAIT,
        handle.get_result(WorkflowGetResultOptions::default()),
    )
    .await??;
    Ok(outcome)
}

#[tokio::test]
async fn refresh_anime_workflow_handles_early_merge_signals_and_blocked_page() -> anyhow::Result<()>
{
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();

    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_delays
            .insert(Provider::AniList, Duration::from_millis(2_000));
        guard
            .page_errors
            .insert(Provider::AniList, PageErrorMode::Blocked);
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::AniList, "2", false, None),
            ],
            entity_merge_version: Some(2),
        };
    }

    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let deployment = build_deployment("test-early");
    let update_worker = build_test_worker(
        &runtime,
        client.clone(),
        WorkerQueueSpec::UpdateService,
        deployment.clone(),
        ports.clone(),
    )?;
    let mal_worker = build_test_worker(
        &runtime,
        client.clone(),
        WorkerQueueSpec::Provider(Provider::Mal),
        deployment.clone(),
        ports.clone(),
    )?;
    let anilist_worker = build_test_worker(
        &runtime,
        client.clone(),
        WorkerQueueSpec::Provider(Provider::AniList),
        deployment,
        ports,
    )?;

    let (stop_tx, stop_rx) = watch::channel(false);
    let mut stop_update = stop_rx.clone();
    let mut stop_mal = stop_rx.clone();
    let mut stop_anilist = stop_rx;
    let anime_id = build_unique_anime_id()?;

    let test_run = async {
        let outcome = start_and_signal_early_merge(&client, &anime_id).await?;
        stop_tx.send_replace(true);
        anyhow::Ok(outcome)
    };

    let (test_outcome, update_res, mal_res, anilist_res) = tokio::join!(
        test_run,
        update_worker.run_until(async move {
            let _ = stop_update.wait_for(|stopping| *stopping).await;
        }),
        mal_worker.run_until(async move {
            let _ = stop_mal.wait_for(|stopping| *stopping).await;
        }),
        anilist_worker.run_until(async move {
            let _ = stop_anilist.wait_for(|stopping| *stopping).await;
        }),
    );
    update_res?;
    mal_res?;
    anilist_res?;
    let outcome = test_outcome?;

    ensure!(outcome.final_outcome == RefreshOutcome::PagesFailed);
    ensure!(outcome.pages_failed == 1);
    ensure!(outcome.merges_run >= 2);
    ensure!(
        fake.state
            .lock()
            .expect("lock")
            .recorded_page_failures
            .len()
            == 1
    );

    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_records_failed_when_merge_fails_non_retryably() -> anyhow::Result<()>
{
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();

    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.merge_fail_non_retryable = true;
        guard.pages = ListEntityPagesOutput {
            pages: Vec::new(),
            entity_merge_version: Some(0),
        };
    }

    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_update_service_worker_options(
            WorkerQueueSpec::UpdateService,
            build_deployment("test-fail"),
            None,
            None,
            ports,
        )?,
    )?;

    let (stop_tx, mut stop_rx) = watch::channel(false);
    let anime_id = build_unique_anime_id()?;
    let workflow_id = build_refresh_anime_workflow_id(&anime_id);

    let test_run = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.clone(),
                    trigger: RefreshTrigger::User,
                    priority: 1,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    merge_coalesce_wait_seconds: Some(0),
                    pages_relinked: true,
                },
                WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id).build(),
            )
            .await?;
        let outcome = tokio::time::timeout(
            TEST_WAIT,
            handle.get_result(WorkflowGetResultOptions::default()),
        )
        .await?;
        stop_tx.send_replace(true);
        anyhow::Ok(outcome)
    };

    let (test_outcome, worker_res) = tokio::join!(
        test_run,
        update_worker.run_until(async move {
            let _ = stop_rx.wait_for(|stopping| *stopping).await;
        }),
    );
    worker_res?;
    ensure!(test_outcome?.is_err());

    {
        let state = fake.state.lock().expect("lock");
        ensure!(state.recorded_outcomes.len() == 1);
        ensure!(state.recorded_outcomes[0].outcome == RefreshOutcome::Failed);
    }

    environment.shutdown().await?;
    Ok(())
}
