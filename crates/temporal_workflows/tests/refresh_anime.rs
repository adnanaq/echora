//! Integration and unit tests for `RefreshAnimeWorkflow`, activities, and ports.

#![allow(
    clippy::expect_used,
    clippy::too_many_lines,
    reason = "integration tests assert setup and orchestrate multi-worker scenarios"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use temporal_workflows::{
    AGENT_SERVICE_TASK_QUEUE, CatalogueDataPort, CharacterFetchTarget, EntityPageSummary,
    FetchRequestedSignal, ListEntityPagesOutput, MAX_PAGE_RETRY_BACKOFF_HOURS, MergeAttemptOutcome,
    MergeEntityOutput, PortError, PortErrorKind, ProviderActivities, RecordRefreshInput,
    RefreshAnimeInput, RefreshAnimeOutcome, RefreshAnimeWorkflow, RefreshCharacterWorkflow,
    RefreshEntityInput, RefreshEntityKind, RefreshEntityOutcome, RefreshEpisodeWorkflow,
    RefreshOutcomeKind, RefreshPageKey, RefreshPageOutput, StartEntityWorkflowsInput,
    UPDATE_SERVICE_TASK_QUEUE, UpdateServiceActivities, UpdateServicePorts, WorkerQueueSpec,
    WorkflowStarterPort, build_refresh_anime_workflow_id, build_refresh_character_workflow_id,
    build_refresh_episode_workflow_id, build_worker_options, build_write_synopsis_workflow_id,
    compute_page_backoff_seconds, is_non_retryable_grpc_code, is_page_due, matches_fetch_request,
    merge_priority, select_pages_to_refresh,
};
use temporalio_client::grpc::WorkflowService as _;
use temporalio_client::{
    Client, ClientOptions, Connection, ConnectionOptions, WorkflowCancelOptions,
    WorkflowGetResultOptions, WorkflowIdConflictPolicy, WorkflowSignalOptions,
    WorkflowStartOptions,
};
use temporalio_common::Priority;
use temporalio_common::protos::temporal::api::workflowservice::v1::RegisterNamespaceRequest;
use temporalio_common::worker::WorkerDeploymentVersion;
use temporalio_sdk::Worker;
use temporalio_sdk::runtime::{Runtime, RuntimeOptions};
use url::Url;
use uuid::{NoContext, Timestamp, Uuid};

#[derive(Default)]
struct FakeState {
    pages: ListEntityPagesOutput,
    aborted_remaining: usize,
    merge_fail_non_retryable: bool,
    page_delays: BTreeMap<String, Duration>,
    page_failures: BTreeMap<String, bool>,
    page_changes: BTreeMap<String, bool>,
    refreshed_calls: Vec<(RefreshPageKey, i64)>,
    merge_calls: usize,
    started_batches: Vec<StartEntityWorkflowsInput>,
    recorded_outcomes: Vec<RecordRefreshInput>,
}

#[derive(Clone, Default)]
struct FakePorts {
    state: Arc<Mutex<FakeState>>,
}

impl CatalogueDataPort for FakePorts {
    fn list_entity_pages(
        &self,
        _entity_kind: RefreshEntityKind,
        _entity_id: String,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>> {
        let out = self.state.lock().expect("lock").pages.clone();
        Box::pin(async move { Ok(out) })
    }

    fn refresh_page(
        &self,
        page_key: RefreshPageKey,
        max_age_seconds: i64,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>> {
        let state = self.state.clone();
        Box::pin(async move {
            let (delay, has_failed, has_changed) = {
                let mut guard = state.lock().expect("lock");
                guard
                    .refreshed_calls
                    .push((page_key.clone(), max_age_seconds));
                let delay = guard
                    .page_delays
                    .get(&page_key.provider)
                    .copied()
                    .unwrap_or(Duration::ZERO);
                let has_failed = guard
                    .page_failures
                    .get(&page_key.provider)
                    .copied()
                    .unwrap_or(false);
                let has_changed = guard
                    .page_changes
                    .get(&page_key.provider)
                    .copied()
                    .unwrap_or(true);
                (delay, has_failed, has_changed)
            };
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            Ok(RefreshPageOutput {
                page_key,
                has_changed,
                has_failed,
            })
        })
    }

    fn merge_entity_attempt(
        &self,
        entity_kind: RefreshEntityKind,
        _entity_id: String,
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
        let is_anime = entity_kind == RefreshEntityKind::Anime;
        Box::pin(async move {
            Ok(MergeAttemptOutcome::Saved(MergeEntityOutput {
                has_changed: true,
                changed_episode_ids: if is_anime {
                    vec!["ep-1".to_owned()]
                } else {
                    Vec::new()
                },
                has_changed_synopses: is_anime,
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

#[test]
fn pure_scheduling_helpers_and_ports_cover_all_edge_cases() {
    assert_eq!(build_refresh_anime_workflow_id("a1"), "refresh-anime-a1");
    assert_eq!(
        build_refresh_episode_workflow_id("e1"),
        "refresh-episode-e1"
    );
    assert_eq!(
        build_refresh_character_workflow_id("c1"),
        "refresh-character-c1"
    );
    assert_eq!(build_write_synopsis_workflow_id("a1"), "synopsis-anime-a1");
    assert_eq!(RefreshEntityKind::Anime.as_str(), "anime");
    assert_eq!(RefreshEntityKind::Episode.as_str(), "episode");
    assert_eq!(RefreshEntityKind::Character.as_str(), "character");

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

    assert!(is_non_retryable_grpc_code(3));
    assert!(is_non_retryable_grpc_code(5));
    assert!(is_non_retryable_grpc_code(7));
    assert!(is_non_retryable_grpc_code(8));
    assert!(is_non_retryable_grpc_code(9));
    assert!(is_non_retryable_grpc_code(12));
    assert!(!is_non_retryable_grpc_code(14));

    let err_r = PortError::retryable("transient");
    assert_eq!(err_r.kind, PortErrorKind::Retryable);
    assert_eq!(err_r.to_string(), "transient");
    let err_n = PortError::non_retryable("permanent");
    assert_eq!(err_n.kind, PortErrorKind::NonRetryable);
    assert_eq!(err_n.to_string(), "permanent");

    let now = 1_700_000_000_i64;
    let partial = EntityPageSummary {
        page_key: RefreshPageKey {
            provider: "kitsu".to_owned(),
            entity_type: "anime".to_owned(),
            address: "1".to_owned(),
        },
        is_partial_source: true,
        fetched_at_epoch_seconds: None,
        is_dead: false,
        fetch_failed_at_epoch_seconds: None,
        fetch_failures: 0,
    };
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
            providers: vec!["kitsu".to_owned()],
            max_age_seconds: 0,
            priority: 1,
        },
        now
    ));
    assert!(!matches_fetch_request(
        &fresh,
        &FetchRequestedSignal {
            providers: vec!["mal".to_owned()],
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
        &[FetchRequestedSignal {
            providers: vec!["kitsu".to_owned()],
            max_age_seconds: 0,
            priority: 1,
        }],
        false,
    );
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].max_age_seconds, 0);
}

#[tokio::test]
async fn noop_ports_reject_all_calls() {
    let ports = UpdateServicePorts::noop();
    assert!(
        ports
            .catalogue
            .list_entity_pages(RefreshEntityKind::Anime, "x".to_owned())
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue
            .refresh_page(
                RefreshPageKey {
                    provider: "mal".to_owned(),
                    entity_type: "anime".to_owned(),
                    address: "1".to_owned(),
                },
                0,
            )
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue
            .merge_entity_attempt(RefreshEntityKind::Anime, "x".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue
            .record_refresh(RecordRefreshInput {
                entity_kind: RefreshEntityKind::Anime,
                entity_id: "x".to_owned(),
                outcome: RefreshOutcomeKind::Succeeded,
                refresh_after_days: 7,
            })
            .await
            .is_err()
    );
    assert!(
        ports
            .starter
            .start_entity_workflows(StartEntityWorkflowsInput {
                anime_id: "x".to_owned(),
                character_fetch_requests: vec![CharacterFetchTarget {
                    character_id: "c1".to_owned(),
                    fetch: FetchRequestedSignal::default(),
                }],
                trigger: "sweep".to_owned(),
                priority: 3,
                ..StartEntityWorkflowsInput::default()
            })
            .await
            .is_err()
    );
}

async fn connect_isolated_client() -> anyhow::Result<Client> {
    let temporal_addr =
        std::env::var("TEMPORAL_ADDRESS").unwrap_or_else(|_| "http://localhost:7233".to_owned());
    let mut connection =
        Connection::connect(ConnectionOptions::new(Url::parse(&temporal_addr)?).build()).await?;
    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    let test_namespace = format!("test-echo111-{}", Uuid::new_v7(ts));
    connection
        .register_namespace(temporalio_client::tonic::Request::new(
            RegisterNamespaceRequest {
                namespace: test_namespace.clone(),
                workflow_execution_retention_period: Some(
                    Duration::from_hours(24)
                        .try_into()
                        .map_err(|err| anyhow::anyhow!("duration: {err}"))?,
                ),
                ..Default::default()
            },
        ))
        .await?;
    // Allow Temporal dev server namespace cache to propagate briefly.
    tokio::time::sleep(Duration::from_millis(200)).await;
    Ok(Client::new(
        connection,
        ClientOptions::new(&test_namespace).build(),
    )?)
}

#[tokio::test]
async fn refresh_anime_workflow_end_to_end_covers_aborted_retry_early_merge_and_outcomes()
-> anyhow::Result<()> {
    let client = connect_isolated_client().await?;

    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.aborted_remaining = 2;
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                EntityPageSummary {
                    page_key: RefreshPageKey {
                        provider: "mal".to_owned(),
                        entity_type: "anime".to_owned(),
                        address: "1".to_owned(),
                    },
                    is_partial_source: false,
                    fetched_at_epoch_seconds: None,
                    is_dead: false,
                    fetch_failed_at_epoch_seconds: None,
                    fetch_failures: 0,
                },
                EntityPageSummary {
                    page_key: RefreshPageKey {
                        provider: "kitsu".to_owned(),
                        entity_type: "anime".to_owned(),
                        address: "2".to_owned(),
                    },
                    is_partial_source: true,
                    fetched_at_epoch_seconds: None,
                    is_dead: false,
                    fetch_failed_at_epoch_seconds: None,
                    fetch_failures: 0,
                },
            ],
            entity_merge_version: Some(1),
        };
    }

    let ports = UpdateServicePorts::new(Arc::new(fake.clone()), Arc::new(fake.clone()));
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())
        .map_err(|err| anyhow::anyhow!("runtime: {err}"))?;
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("test-111".to_owned())
        .build();

    let mut update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(
            &WorkerQueueSpec::UpdateService,
            deployment.clone(),
            false,
            ports.clone(),
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;
    let mut mal_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(
            &WorkerQueueSpec::Provider("mal".to_owned()),
            deployment.clone(),
            false,
            ports.clone(),
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    // Agent queue polls StartEntityWorkflowsInput for synopsis
    let agent_opts = temporalio_sdk::WorkerOptions::new(AGENT_SERVICE_TASK_QUEUE)
        .register_activities(UpdateServiceActivities::new(ports))
        .build();
    let mut agent_worker = Worker::new(&runtime, client.clone(), agent_opts)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    let anime_id = Uuid::new_v7(ts).to_string();
    let wf_id = build_refresh_anime_workflow_id(&anime_id);

    let mut priority = Priority::default();
    priority.priority_key = Some(1);
    let start_opts = WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, wf_id)
        .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
        .priority(priority)
        .build();

    let run_future = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.clone(),
                    trigger: "user".to_owned(),
                    priority: 1,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    pages_relinked: false,
                },
                start_opts,
            )
            .await?;
        let outcome: RefreshAnimeOutcome = handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        assert_eq!(outcome.anime_id, anime_id);
        assert_eq!(outcome.pages_refreshed, 1);
        assert_eq!(outcome.pages_failed, 0);
        assert_eq!(outcome.merges_run, 1);
        assert_eq!(outcome.episode_workflows_started, 1);
        assert!(outcome.synopsis_workflow_started);
        assert_eq!(outcome.final_outcome, RefreshOutcomeKind::Succeeded);
        anyhow::Ok(())
    };

    tokio::select! {
        res = run_future => { res?; }
        res = update_worker.run() => {
            res.map_err(|err| anyhow::anyhow!("update worker: {err}"))?;
        }
        res = mal_worker.run() => {
            res.map_err(|err| anyhow::anyhow!("mal worker: {err}"))?;
        }
        res = agent_worker.run() => {
            res.map_err(|err| anyhow::anyhow!("agent worker: {err}"))?;
        }
    }

    let state = fake.state.lock().expect("lock");
    // 2 ABORTED conflicts + 1 successful save = 3 merge attempts inside 1 activity!
    assert_eq!(state.merge_calls, 3);
    assert_eq!(state.recorded_outcomes.len(), 1);
    assert_eq!(
        state.recorded_outcomes[0].outcome,
        RefreshOutcomeKind::Succeeded
    );
    assert_eq!(state.started_batches.len(), 2);
    assert_eq!(state.started_batches[0].episode_ids, vec!["ep-1"]);
    assert!(!state.started_batches[0].start_synopsis);
    assert!(state.started_batches[1].start_synopsis);

    // Ensure ProviderActivities keeps its public with_heartbeat_interval builder tested
    let _custom_provider = ProviderActivities::new(UpdateServicePorts::noop())
        .with_heartbeat_interval(Duration::from_millis(50));

    // Also test AgentService + versioned UpdateService worker option builders
    let agent_spec_opts = build_worker_options(
        &WorkerQueueSpec::AgentService,
        deployment.clone(),
        true,
        UpdateServicePorts::noop(),
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;
    assert_eq!(agent_spec_opts.task_queue, AGENT_SERVICE_TASK_QUEUE);

    Ok(())
}

#[tokio::test]
async fn refresh_anime_activity_handles_exceeded_aborted_retries_and_cancellation()
-> anyhow::Result<()> {
    let client = connect_isolated_client().await?;
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        // Exceed MAX_ABORTED_RETRIES_PER_ATTEMPT (10) so merge_anime returns retryable error
        guard.aborted_remaining = 100;
        // Slow mal page so we can cancel workflow mid-flight
        guard
            .page_delays
            .insert("mal".to_owned(), Duration::from_secs(10));
        guard.pages = ListEntityPagesOutput {
            pages: vec![EntityPageSummary {
                page_key: RefreshPageKey {
                    provider: "mal".to_owned(),
                    entity_type: "anime".to_owned(),
                    address: "1".to_owned(),
                },
                is_partial_source: false,
                fetched_at_epoch_seconds: None,
                is_dead: false,
                fetch_failed_at_epoch_seconds: None,
                fetch_failures: 0,
            }],
            entity_merge_version: Some(0),
        };
    }

    let ports = UpdateServicePorts::new(Arc::new(fake.clone()), Arc::new(fake.clone()));
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())
        .map_err(|err| anyhow::anyhow!("runtime: {err}"))?;
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("test-111-cancel".to_owned())
        .build();

    let mut update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(
            &WorkerQueueSpec::UpdateService,
            deployment.clone(),
            false,
            ports.clone(),
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    let mal_opts = temporalio_sdk::WorkerOptions::new("provider-mal")
        .register_activities(
            ProviderActivities::new(ports).with_heartbeat_interval(Duration::from_millis(20)),
        )
        .build();
    let mut mal_worker =
        Worker::new(&runtime, client.clone(), mal_opts).map_err(|err| anyhow::anyhow!("{err}"))?;

    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    let anime_id = Uuid::new_v7(ts).to_string();
    let wf_id = build_refresh_anime_workflow_id(&anime_id);
    let start_opts = WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, wf_id).build();

    let run_future = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id,
                    trigger: "user".to_owned(),
                    priority: 1,
                    refresh_after_days: 0,
                    merge_version: 1,
                    merge_first_wait_seconds: None,
                    pages_relinked: false,
                },
                start_opts,
            )
            .await?;
        tokio::time::sleep(Duration::from_millis(200)).await;
        handle.cancel(WorkflowCancelOptions::default()).await?;
        let res: Result<RefreshAnimeOutcome, _> =
            handle.get_result(WorkflowGetResultOptions::default()).await;
        assert!(res.is_err());
        anyhow::Ok(())
    };

    tokio::select! {
        res = run_future => { res?; }
        res = update_worker.run() => { res.map_err(|err| anyhow::anyhow!("update worker: {err}"))?; }
        res = mal_worker.run() => { res.map_err(|err| anyhow::anyhow!("mal worker: {err}"))?; }
    }

    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_covers_early_merge_timer_and_signals_and_pages_failed()
-> anyhow::Result<()> {
    let client = connect_isolated_client().await?;
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_delays
            .insert("anilist".to_owned(), Duration::from_millis(2_500));
        guard.page_failures.insert("anilist".to_owned(), true);
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                EntityPageSummary {
                    page_key: RefreshPageKey {
                        provider: "mal".to_owned(),
                        entity_type: "anime".to_owned(),
                        address: "1".to_owned(),
                    },
                    is_partial_source: false,
                    fetched_at_epoch_seconds: None,
                    is_dead: false,
                    fetch_failed_at_epoch_seconds: None,
                    fetch_failures: 0,
                },
                EntityPageSummary {
                    page_key: RefreshPageKey {
                        provider: "anilist".to_owned(),
                        entity_type: "anime".to_owned(),
                        address: "2".to_owned(),
                    },
                    is_partial_source: false,
                    fetched_at_epoch_seconds: None,
                    is_dead: false,
                    fetch_failed_at_epoch_seconds: None,
                    fetch_failures: 0,
                },
            ],
            entity_merge_version: Some(2),
        };
    }

    let ports = UpdateServicePorts::new(Arc::new(fake.clone()), Arc::new(fake.clone()));
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())
        .map_err(|err| anyhow::anyhow!("runtime: {err}"))?;
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("test-111-early".to_owned())
        .build();

    let mut update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(
            &WorkerQueueSpec::UpdateService,
            deployment.clone(),
            false,
            ports.clone(),
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    let mal_opts = temporalio_sdk::WorkerOptions::new("provider-mal")
        .register_activities(
            ProviderActivities::new(ports.clone())
                .with_heartbeat_interval(Duration::from_millis(50)),
        )
        .build();
    let mut mal_worker =
        Worker::new(&runtime, client.clone(), mal_opts).map_err(|err| anyhow::anyhow!("{err}"))?;

    let anilist_opts = temporalio_sdk::WorkerOptions::new("provider-anilist")
        .register_activities(
            ProviderActivities::new(ports.clone())
                .with_heartbeat_interval(Duration::from_millis(50)),
        )
        .build();
    let mut anilist_worker = Worker::new(&runtime, client.clone(), anilist_opts)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let agent_opts = temporalio_sdk::WorkerOptions::new(AGENT_SERVICE_TASK_QUEUE)
        .register_activities(UpdateServiceActivities::new(ports))
        .build();
    let mut agent_worker = Worker::new(&runtime, client.clone(), agent_opts)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    let anime_id = Uuid::new_v7(ts).to_string();
    let wf_id = build_refresh_anime_workflow_id(&anime_id);

    let start_opts = WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, wf_id)
        .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
        .build();

    let run_future = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.clone(),
                    trigger: "sweep".to_owned(),
                    priority: 3,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(1),
                    pages_relinked: false,
                },
                start_opts,
            )
            .await?;

        // Send priority escalation via fetch_requested + merge_requested after Cycle 1 starts
        // and early merge timer fires while anilist is still running.
        tokio::time::sleep(Duration::from_millis(1_300)).await;
        handle
            .signal(
                RefreshAnimeWorkflow::fetch_requested,
                FetchRequestedSignal {
                    providers: vec!["mal".to_owned()],
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

        let outcome: RefreshAnimeOutcome = handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        assert_eq!(outcome.final_outcome, RefreshOutcomeKind::PagesFailed);
        assert_eq!(outcome.pages_failed, 1);
        assert!(outcome.merges_run >= 2);
        anyhow::Ok(())
    };

    tokio::select! {
        res = run_future => { res?; }
        res = update_worker.run() => { res.map_err(|err| anyhow::anyhow!("update worker: {err}"))?; }
        res = mal_worker.run() => { res.map_err(|err| anyhow::anyhow!("mal worker: {err}"))?; }
        res = anilist_worker.run() => { res.map_err(|err| anyhow::anyhow!("anilist worker: {err}"))?; }
        res = agent_worker.run() => { res.map_err(|err| anyhow::anyhow!("agent worker: {err}"))?; }
    }

    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_records_failed_when_merge_fails_non_retryably() -> anyhow::Result<()>
{
    let client = connect_isolated_client().await?;
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.merge_fail_non_retryable = true;
        guard.pages = ListEntityPagesOutput {
            pages: vec![],
            entity_merge_version: Some(0),
        };
    }

    let ports = UpdateServicePorts::new(Arc::new(fake.clone()), Arc::new(fake.clone()));
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())
        .map_err(|err| anyhow::anyhow!("runtime: {err}"))?;
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("test-111-fail".to_owned())
        .build();

    let mut update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(&WorkerQueueSpec::UpdateService, deployment, false, ports)
            .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let ts = Timestamp::from_unix(NoContext, elapsed.as_secs(), elapsed.subsec_nanos());
    let anime_id = Uuid::new_v7(ts).to_string();
    let wf_id = build_refresh_anime_workflow_id(&anime_id);
    let start_opts = WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, wf_id).build();

    let run_future = async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.clone(),
                    trigger: "user".to_owned(),
                    priority: 1,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    pages_relinked: true,
                },
                start_opts,
            )
            .await?;
        let res: Result<RefreshAnimeOutcome, _> =
            handle.get_result(WorkflowGetResultOptions::default()).await;
        assert!(res.is_err());
        anyhow::Ok(())
    };

    tokio::select! {
        res = run_future => { res?; }
        res = update_worker.run() => { res.map_err(|err| anyhow::anyhow!("update worker: {err}"))?; }
    }

    let state = fake.state.lock().expect("lock");
    assert_eq!(state.recorded_outcomes.len(), 1);
    assert_eq!(
        state.recorded_outcomes[0].outcome,
        RefreshOutcomeKind::Failed
    );
    Ok(())
}

#[tokio::test]
async fn refresh_episode_and_character_workflows_execute_end_to_end() -> anyhow::Result<()> {
    let client = connect_isolated_client().await?;
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.aborted_remaining = 1;
        guard.pages = ListEntityPagesOutput {
            pages: vec![EntityPageSummary {
                page_key: RefreshPageKey {
                    provider: "mal".to_owned(),
                    entity_type: "episode".to_owned(),
                    address: "1".to_owned(),
                },
                is_partial_source: false,
                fetched_at_epoch_seconds: None,
                is_dead: false,
                fetch_failed_at_epoch_seconds: None,
                fetch_failures: 0,
            }],
            entity_merge_version: Some(1),
        };
    }

    let ports = UpdateServicePorts::new(Arc::new(fake.clone()), Arc::new(fake.clone()));
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())
        .map_err(|err| anyhow::anyhow!("runtime: {err}"))?;
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name("update-service".to_owned())
        .build_id("test-112-entity".to_owned())
        .build();

    let mut update_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(
            &WorkerQueueSpec::UpdateService,
            deployment.clone(),
            false,
            ports.clone(),
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    let mut mal_worker = Worker::new(
        &runtime,
        client.clone(),
        build_worker_options(
            &WorkerQueueSpec::Provider("mal".to_owned()),
            deployment,
            false,
            ports,
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    let ep_opts = WorkflowStartOptions::new(
        UPDATE_SERVICE_TASK_QUEUE,
        build_refresh_episode_workflow_id("ep-112"),
    )
    .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
    .build();
    let ch_opts = WorkflowStartOptions::new(
        UPDATE_SERVICE_TASK_QUEUE,
        build_refresh_character_workflow_id("char-112"),
    )
    .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
    .build();

    let run_future = async {
        let ep_handle = client
            .start_workflow(
                RefreshEpisodeWorkflow::run,
                RefreshEntityInput {
                    entity_id: "ep-112".to_owned(),
                    trigger: "episode".to_owned(),
                    priority: 2,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    pages_relinked: false,
                },
                ep_opts,
            )
            .await?;
        let ep_out: RefreshEntityOutcome = ep_handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        assert_eq!(ep_out.entity_id, "ep-112");
        assert_eq!(ep_out.final_outcome, RefreshOutcomeKind::Succeeded);
        assert_eq!(ep_out.merges_run, 1);

        let ch_handle = client
            .signal_with_start_workflow(
                RefreshCharacterWorkflow::run,
                RefreshEntityInput {
                    entity_id: "char-112".to_owned(),
                    trigger: "user".to_owned(),
                    priority: 1,
                    refresh_after_days: 7,
                    merge_version: 2,
                    merge_first_wait_seconds: Some(0),
                    pages_relinked: false,
                },
                RefreshCharacterWorkflow::fetch_requested,
                FetchRequestedSignal {
                    providers: vec!["mal".to_owned()],
                    max_age_seconds: 0,
                    priority: 1,
                },
                ch_opts,
            )
            .await?;
        ch_handle
            .signal(
                RefreshCharacterWorkflow::merge_requested,
                (),
                WorkflowSignalOptions::default(),
            )
            .await?;
        let ch_out: RefreshEntityOutcome = ch_handle
            .get_result(WorkflowGetResultOptions::default())
            .await?;
        assert_eq!(ch_out.entity_id, "char-112");
        assert_eq!(ch_out.final_outcome, RefreshOutcomeKind::Succeeded);
        anyhow::Ok(())
    };

    tokio::select! {
        res = run_future => { res?; }
        res = update_worker.run() => { res.map_err(|err| anyhow::anyhow!("update worker: {err}"))?; }
        res = mal_worker.run() => { res.map_err(|err| anyhow::anyhow!("mal worker: {err}"))?; }
    }

    let state = fake.state.lock().expect("lock");
    assert_eq!(state.recorded_outcomes.len(), 2);
    assert_eq!(
        state.recorded_outcomes[0].entity_kind,
        RefreshEntityKind::Episode
    );
    assert_eq!(
        state.recorded_outcomes[1].entity_kind,
        RefreshEntityKind::Character
    );
    assert_eq!(state.started_batches.len(), 0);

    Ok(())
}
