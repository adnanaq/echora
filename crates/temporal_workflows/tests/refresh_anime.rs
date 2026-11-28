//! Unit and local-server integration tests for `RefreshAnimeWorkflow`, activities, and ports.

#![expect(
    clippy::expect_used,
    clippy::struct_excessive_bools,
    reason = "integration tests unwrap setup locks and mock state flags"
)]

use std::any::Any;
use std::collections::BTreeMap;
use std::error::Error as _;
use std::future::Future;
use std::num::NonZeroU32;
use std::str::FromStr as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use anyhow::ensure;
use futures::future::{BoxFuture, join_all};
use temporal_workflows::testing::start_local_environment;
use temporal_workflows::{
    CatalogueDataPort, EntityPageSummary, EntityType, FetchRequestedSignal, ListEntityPagesOutput,
    MAX_ABORTED_RETRIES_PER_ATTEMPT, MAX_PAGE_RETRY_BACKOFF_HOURS, MergeAnimeOutput,
    MergeAttemptOutcome, PortError, Provider, ProviderActivities, RefreshAnimeInput,
    RefreshAnimeOutcome, RefreshAnimeWorkflow, RefreshOutcome, RefreshPageInput, RefreshPageKey,
    RefreshPageOutput, RefreshSettings, RefreshTrigger, StartEntityWorkflowsInput,
    UPDATE_SERVICE_TASK_QUEUE, UpdateServiceActivities, UpdateServicePorts,
    UpdateServiceWorkerConfig, WorkerQueueSpec, WorkflowStarterPort, build_priority,
    build_refresh_anime_workflow_id, build_update_service_worker_options, build_worker_options,
    compute_page_backoff_seconds, is_page_due, matches_fetch_request, merge_priority,
    select_pages_to_refresh,
};
use temporalio_client::{
    Client, WorkflowCancelOptions, WorkflowGetResultOptions, WorkflowIdConflictPolicy,
    WorkflowSignalOptions, WorkflowStartOptions,
};
use temporalio_common::worker::WorkerDeploymentVersion;
use temporalio_sdk::Worker;
use temporalio_sdk::activities::ActivityError;
use temporalio_sdk::runtime::{Runtime, RuntimeOptions};
use temporalio_sdk::testing::{ActivityEnvironment, ActivityEnvironmentError};
use tokio::sync::{Notify, watch};
use uuid::Uuid;

const TEST_WAIT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct FakeState {
    pages: ListEntityPagesOutput,
    list_pages_fail: bool,
    aborted_remaining: usize,
    merge_fail_non_retryable: bool,
    synopsis_start_fail: bool,
    record_refresh_fail: bool,
    record_page_failure_fail: bool,
    has_changed_synopses: bool,
    changed_episode_ids: Vec<String>,
    page_delays: BTreeMap<Provider, Duration>,
    page_errors: BTreeMap<Provider, PageErrorMode>,
    page_changes: BTreeMap<Provider, bool>,
    page_reported_failed: BTreeMap<Provider, bool>,
    page_started: BTreeMap<Provider, Arc<Notify>>,
    record_refresh_delay: Duration,
    record_refresh_entered: Option<Arc<Notify>>,
    synopsis_delay: Duration,
    synopsis_entered: Option<Arc<Notify>>,
    fetch_calls: Vec<(RefreshPageKey, Duration)>,
    recorded_page_failures: Vec<RefreshPageKey>,
    merge_calls: usize,
    merge_versions_seen: Vec<i32>,
    started_batches: Vec<StartEntityWorkflowsInput>,
    recorded_outcomes: Vec<(Uuid, RefreshOutcome, i64)>,
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
        _anime_id: Uuid,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>> {
        let guard = self.state.lock().expect("lock");
        if guard.list_pages_fail {
            return Box::pin(async {
                Err(PortError::non_retryable("list_entity_pages unavailable"))
            });
        }
        let output = guard.pages.clone();
        Box::pin(async move { Ok(output) })
    }

    fn fetch_and_store_page(
        &self,
        page_key: RefreshPageKey,
        max_age: Duration,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>> {
        let state = self.state.clone();
        Box::pin(async move {
            let (delay, error_mode, has_changed, has_failed, started_notify) = {
                let mut guard = state.lock().expect("lock");
                guard.fetch_calls.push((page_key.clone(), max_age));
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
                let has_failed = guard
                    .page_reported_failed
                    .get(&page_key.provider)
                    .copied()
                    .unwrap_or(false);
                let started_notify = guard.page_started.get(&page_key.provider).cloned();
                (delay, error_mode, has_changed, has_failed, started_notify)
            };
            if let Some(notify) = started_notify {
                notify.notify_one();
            }
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
                    has_failed,
                }),
            }
        })
    }

    fn record_page_failure(
        &self,
        page_key: RefreshPageKey,
        _failed_at: SystemTime,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        let should_fail = {
            let mut guard = self.state.lock().expect("lock");
            guard.recorded_page_failures.push(page_key);
            guard.record_page_failure_fail
        };
        Box::pin(async move {
            if should_fail {
                return Err(PortError::non_retryable("store_page failed"));
            }
            Ok(())
        })
    }

    fn merge_anime_attempt(
        &self,
        _anime_id: Uuid,
        merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>> {
        let mut guard = self.state.lock().expect("lock");
        guard.merge_calls += 1;
        guard.merge_versions_seen.push(merge_version);
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
        let changed_episode_ids = guard.changed_episode_ids.clone();
        let has_changed_synopses = guard.has_changed_synopses;
        Box::pin(async move {
            Ok(MergeAttemptOutcome::Saved(MergeAnimeOutput {
                has_changed: true,
                changed_episode_ids,
                has_changed_synopses,
            }))
        })
    }

    fn record_refresh(
        &self,
        anime_id: Uuid,
        outcome: RefreshOutcome,
        refresh_after_days: i64,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        let state = self.state.clone();
        Box::pin(async move {
            let (delay, entered, should_fail) = {
                let mut guard = state.lock().expect("lock");
                guard
                    .recorded_outcomes
                    .push((anime_id, outcome, refresh_after_days));
                (
                    guard.record_refresh_delay,
                    guard.record_refresh_entered.take(),
                    guard.record_refresh_fail,
                )
            };
            if let Some(notify) = entered {
                notify.notify_one();
            }
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            if should_fail {
                return Err(PortError::non_retryable("missing entity in record_refresh"));
            }
            Ok(())
        })
    }
}

impl WorkflowStarterPort for FakePorts {
    fn start_entity_workflows(
        &self,
        input: StartEntityWorkflowsInput,
    ) -> BoxFuture<'_, Result<usize, PortError>> {
        let state = self.state.clone();
        Box::pin(async move {
            let count = input.episode_ids.len() + usize::from(input.start_synopsis);
            let (delay, entered, should_fail) = {
                let mut guard = state.lock().expect("lock");
                guard.started_batches.push(input.clone());
                if input.start_synopsis {
                    (
                        guard.synopsis_delay,
                        guard.synopsis_entered.take(),
                        guard.synopsis_start_fail,
                    )
                } else {
                    (Duration::ZERO, None, false)
                }
            };
            if let Some(notify) = entered {
                notify.notify_one();
            }
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            if should_fail {
                return Err(PortError::non_retryable("synopsis workflow start failed"));
            }
            Ok(count)
        })
    }
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

fn fast_refresh_settings(merge_version: i32) -> RefreshSettings {
    RefreshSettings::new(7, merge_version, 0, 0)
}

async fn run_with_workers<T, F, Fut>(
    build_id: &str,
    fake: FakePorts,
    refresh_settings: RefreshSettings,
    providers: &[Provider],
    provider_heartbeat: Option<Duration>,
    body: F,
) -> anyhow::Result<T>
where
    F: FnOnce(Client) -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    let environment = start_local_environment(None).await?;
    let client = environment.client().clone();
    let ports = UpdateServicePorts::new(fake.clone(), fake);
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let deployment = build_deployment(build_id);

    let config = UpdateServiceWorkerConfig {
        deployment_version: deployment.clone(),
        versioning_behavior: None,
        graceful_shutdown_period: Some(Duration::from_secs(2)),
        ports: ports.clone(),
        refresh_settings,
    };

    let mut workers = Vec::with_capacity(1 + providers.len());
    workers.push(Worker::new(
        &runtime,
        client.clone(),
        build_update_service_worker_options(WorkerQueueSpec::UpdateService, &config)?,
    )?);
    for &provider in providers {
        let options = if let Some(interval) = provider_heartbeat {
            build_worker_options(
                WorkerQueueSpec::Provider(provider),
                deployment.clone(),
                None,
            )
            .graceful_shutdown_period(Duration::from_secs(2))
            .register_activities(
                ProviderActivities::new(ports.clone()).with_heartbeat_interval(interval),
            )
            .build()
        } else {
            build_update_service_worker_options(WorkerQueueSpec::Provider(provider), &config)?
        };
        workers.push(Worker::new(&runtime, client.clone(), options)?);
    }

    let (stop_tx, stop_rx) = watch::channel(false);
    let worker_runs = workers.into_iter().map(|worker| {
        let mut rx = stop_rx.clone();
        async move {
            worker
                .run_until(async move {
                    let _ = rx.wait_for(|stopping| *stopping).await;
                })
                .await
        }
    });

    let test_run = async {
        let output = body(client).await;
        stop_tx.send_replace(true);
        output
    };

    let (test_outcome, worker_results) = tokio::join!(test_run, join_all(worker_runs));
    for worker_result in worker_results {
        worker_result?;
    }
    let output = test_outcome?;
    environment.shutdown().await?;
    Ok(output)
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

    assert_eq!(merge_priority(None, None), None);
    assert_eq!(merge_priority(None, NonZeroU32::new(2)), NonZeroU32::new(2));
    assert_eq!(merge_priority(NonZeroU32::new(2), None), NonZeroU32::new(2));
    assert_eq!(
        merge_priority(NonZeroU32::new(3), NonZeroU32::new(1)),
        NonZeroU32::new(1)
    );
    assert_eq!(
        merge_priority(NonZeroU32::new(1), NonZeroU32::new(3)),
        NonZeroU32::new(1)
    );

    let remerge_priority = build_priority(None, RefreshTrigger::Remerge);
    assert_eq!(remerge_priority.priority_key, Some(4));
    assert_eq!(remerge_priority.fairness_key.as_deref(), Some("remerge"));

    for provider in Provider::ALL {
        assert_eq!(Provider::from_str(provider.slug()).ok(), Some(provider));
        assert_eq!(
            Provider::try_from(provider.slug().to_owned()).ok(),
            Some(provider)
        );
        assert_eq!(<&'static str>::from(provider), provider.slug());
    }
    assert!(Provider::from_str("unknown").is_err());

    for entity_type in EntityType::ALL {
        assert_eq!(
            EntityType::from_str(entity_type.as_str()).ok(),
            Some(entity_type)
        );
        assert_eq!(
            EntityType::try_from(entity_type.as_str().to_owned()).ok(),
            Some(entity_type)
        );
        assert_eq!(<&'static str>::from(entity_type), entity_type.as_str());
    }
    assert!(EntityType::from_str("unknown").is_err());

    for trigger in RefreshTrigger::ALL {
        assert_eq!(
            RefreshTrigger::from_str(trigger.as_str()).ok(),
            Some(trigger)
        );
        assert_eq!(
            RefreshTrigger::try_from(trigger.as_str().to_owned()).ok(),
            Some(trigger)
        );
        assert_eq!(<&'static str>::from(trigger), trigger.as_str());
        assert!((1..=4).contains(&trigger.default_priority_key()));
    }
    assert!(RefreshTrigger::from_str("unknown").is_err());

    let err_with_source =
        PortError::retryable("boom").with_source(std::io::Error::other("io failure"));
    assert!(err_with_source.source().is_some());
    assert!(format!("{err_with_source:?}").contains("has_source: true"));
    assert_eq!(format!("{}", EntityType::Anime), "anime");
    assert_eq!(format!("{}", RefreshTrigger::User), "user");
    assert!(
        format!(
            "{}",
            Provider::from_str("bad").expect_err("invalid provider")
        )
        .contains("unknown provider")
    );
    let noop_ports = UpdateServicePorts::noop();
    assert!(format!("{noop_ports:?}").contains("UpdateServicePorts"));
    let agent_config = UpdateServiceWorkerConfig {
        deployment_version: build_deployment("agent"),
        versioning_behavior: None,
        graceful_shutdown_period: None,
        ports: noop_ports,
        refresh_settings: RefreshSettings::default(),
    };
    let agent_opts =
        build_update_service_worker_options(WorkerQueueSpec::AgentService, &agent_config)
            .expect("agent worker options");
    assert_eq!(agent_opts.task_queue, "agent-service");
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
            max_age: Duration::ZERO,
            priority: NonZeroU32::new(1),
        },
        now
    ));
    assert!(!matches_fetch_request(
        &fresh,
        &FetchRequestedSignal {
            providers: vec![Provider::Mal],
            max_age: Duration::ZERO,
            priority: NonZeroU32::new(1),
        },
        now
    ));
    assert!(!matches_fetch_request(
        &fresh,
        &FetchRequestedSignal {
            providers: Vec::new(),
            max_age: Duration::from_secs(500),
            priority: NonZeroU32::new(1),
        },
        now
    ));

    let selected = select_pages_to_refresh(
        &[fresh, build_page_summary(Provider::Mal, "9", false, None)],
        now,
        7,
        &[
            FetchRequestedSignal {
                providers: vec![Provider::Kitsu],
                max_age: Duration::from_secs(50),
                priority: NonZeroU32::new(3),
            },
            FetchRequestedSignal {
                providers: vec![Provider::Kitsu],
                max_age: Duration::ZERO,
                priority: NonZeroU32::new(1),
            },
        ],
        true,
    );
    assert_eq!(selected.len(), 2);
    assert!(matches_fetch_request(
        &build_page_summary(Provider::Mal, "9", false, None),
        &FetchRequestedSignal {
            providers: vec![Provider::Mal],
            max_age: Duration::from_secs(60),
            priority: None,
        },
        now
    ));
}

#[tokio::test]
async fn refresh_page_transient_error_returns_retryable_failure() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_errors
        .insert(Provider::Mal, PageErrorMode::Transient);
    let ports = UpdateServicePorts::new(fake.clone(), fake);
    let env = ActivityEnvironment::builder()
        .register_activities(ProviderActivities::new(ports))
        .build();
    let error = env
        .run(
            ProviderActivities::refresh_page,
            RefreshPageInput {
                page_key: RefreshPageKey {
                    provider: Provider::Mal,
                    entity_type: EntityType::Anime,
                    address: "21".to_owned(),
                },
                max_age: Duration::ZERO,
            },
        )
        .await
        .expect_err("transient error must return retryable failure");
    assert!(matches!(
        error,
        ActivityEnvironmentError::Activity(ActivityError::Application(ref failure))
            if !failure.is_non_retryable()
    ));
}

#[tokio::test]
async fn refresh_page_provider_blocked_returns_non_retryable_failure() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_errors
        .insert(Provider::AniSearch, PageErrorMode::Blocked);
    let ports = UpdateServicePorts::new(fake.clone(), fake);
    let env = ActivityEnvironment::builder()
        .register_activities(ProviderActivities::new(ports))
        .build();
    let error = env
        .run(
            ProviderActivities::refresh_page,
            RefreshPageInput {
                page_key: RefreshPageKey {
                    provider: Provider::AniSearch,
                    entity_type: EntityType::Anime,
                    address: "100".to_owned(),
                },
                max_age: Duration::ZERO,
            },
        )
        .await
        .expect_err("blocked provider must fail non-retryably on attempt 1");
    assert!(matches!(
        error,
        ActivityEnvironmentError::Activity(ActivityError::Application(ref failure))
            if failure.is_non_retryable()
    ));
}

#[tokio::test]
async fn refresh_page_permanent_error_returns_non_retryable_failure() {
    let fake = FakePorts::default();
    fake.state
        .lock()
        .expect("lock")
        .page_errors
        .insert(Provider::AniList, PageErrorMode::NonRetryable);
    let ports = UpdateServicePorts::new(fake.clone(), fake);
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
                max_age: Duration::ZERO,
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
    let ports = UpdateServicePorts::new(fake.clone(), fake);

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
                max_age: Duration::ZERO,
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
        .register_activities(UpdateServiceActivities::with_settings(
            ports,
            fast_refresh_settings(2),
        ))
        .build();
    let error = env
        .run(UpdateServiceActivities::merge_anime, Uuid::now_v7())
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
async fn merge_anime_cancels_cleanly_when_activity_context_is_cancelled() {
    let fake = FakePorts::default();
    fake.state.lock().expect("lock").aborted_remaining = 4;
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let env = ActivityEnvironment::builder()
        .register_activities(
            UpdateServiceActivities::new(ports).with_heartbeat_interval(Duration::from_millis(15)),
        )
        .build();
    env.cancel();

    let merge_outcome = env
        .run(UpdateServiceActivities::merge_anime, Uuid::now_v7())
        .await;
    assert!(matches!(
        merge_outcome,
        Err(ActivityEnvironmentError::Activity(
            ActivityError::Cancelled { .. }
        ))
    ));
    assert_eq!(fake.state.lock().expect("lock").merge_calls, 0);
}

#[tokio::test]
async fn start_entity_workflows_cancels_cleanly_when_activity_context_is_cancelled() {
    let fake = FakePorts::default();
    let ports = UpdateServicePorts::new(fake.clone(), fake.clone());

    let env = ActivityEnvironment::builder()
        .register_activities(
            UpdateServiceActivities::new(ports).with_heartbeat_interval(Duration::from_millis(15)),
        )
        .build();
    env.cancel();

    let start_outcome = env
        .run(
            UpdateServiceActivities::start_entity_workflows,
            StartEntityWorkflowsInput {
                anime_id: Uuid::now_v7(),
                episode_ids: vec!["ep-1".to_owned(), "ep-2".to_owned()],
                start_synopsis: false,
                trigger: RefreshTrigger::User,
                priority: NonZeroU32::new(1),
            },
        )
        .await;
    assert!(matches!(
        start_outcome,
        Err(ActivityEnvironmentError::Activity(
            ActivityError::Cancelled { .. }
        ))
    ));
    assert_eq!(fake.state.lock().expect("lock").started_batches.len(), 0);
}

#[tokio::test]
async fn noop_ports_reject_all_operations() {
    let ports = UpdateServicePorts::noop();
    let anime_id = Uuid::now_v7();
    let key = RefreshPageKey {
        provider: Provider::Mal,
        entity_type: EntityType::Anime,
        address: "1".to_owned(),
    };
    assert!(ports.catalogue().list_entity_pages(anime_id).await.is_err());
    assert!(
        ports
            .catalogue()
            .fetch_and_store_page(key.clone(), Duration::ZERO)
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue()
            .record_page_failure(key, SystemTime::now())
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue()
            .merge_anime_attempt(anime_id, 1)
            .await
            .is_err()
    );
    assert!(
        ports
            .catalogue()
            .record_refresh(anime_id, RefreshOutcome::Succeeded, 7)
            .await
            .is_err()
    );
    assert!(
        ports
            .starter()
            .start_entity_workflows(StartEntityWorkflowsInput {
                anime_id,
                episode_ids: Vec::new(),
                start_synopsis: false,
                trigger: RefreshTrigger::Sweep,
                priority: NonZeroU32::new(3),
            })
            .await
            .is_err()
    );
}

#[tokio::test]
async fn refresh_anime_workflow_refreshes_due_pages_retries_aborted_and_starts_downstream()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.aborted_remaining = 2;
        guard.has_changed_synopses = true;
        guard.changed_episode_ids = vec!["ep-1".to_owned()];
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::Kitsu, "2", true, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-happy",
        fake.clone(),
        fast_refresh_settings(2),
        &[Provider::Mal],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        trigger: RefreshTrigger::User,
                        priority: NonZeroU32::new(1),
                        pages_relinked: false,
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
                    .priority(build_priority(NonZeroU32::new(1), RefreshTrigger::User))
                    .build(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.anime_id == anime_id);
    ensure!(outcome.pages_refreshed == 1);
    ensure!(outcome.pages_failed == 0);
    ensure!(outcome.merges_run == 1);
    ensure!(outcome.episode_workflows_started == 1);
    ensure!(outcome.synopsis_workflow_started);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);

    let state = fake.state.lock().expect("lock");
    ensure!(state.merge_calls == 3);
    ensure!(state.merge_versions_seen == vec![2, 2, 2]);
    ensure!(state.recorded_outcomes == vec![(anime_id, RefreshOutcome::Succeeded, 7)]);
    ensure!(state.started_batches.len() == 2);
    ensure!(state.started_batches[0].episode_ids == vec!["ep-1"]);
    ensure!(!state.started_batches[0].start_synopsis);
    ensure!(state.started_batches[1].start_synopsis);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_default_input_uses_worker_merge_version_and_sweep_remerges_without_fetching()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.pages = ListEntityPagesOutput {
            pages: vec![build_page_summary(
                Provider::Mal,
                "1",
                false,
                Some(i64::MAX / 2),
            )],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-worker-settings",
        fake.clone(),
        RefreshSettings::new(14, 3, 0, 0),
        &[],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.pages_refreshed == 0);
    ensure!(outcome.merges_run == 1);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);
    let state = fake.state.lock().expect("lock");
    ensure!(state.fetch_calls.is_empty());
    ensure!(state.merge_versions_seen == vec![3]);
    ensure!(state.recorded_outcomes == vec![(anime_id, RefreshOutcome::Succeeded, 14)]);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_remerge_trigger_skips_due_pages_and_merges_stored_pages()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.pages = ListEntityPagesOutput {
            pages: vec![build_page_summary(Provider::Mal, "1", false, None)],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-remerge",
        fake.clone(),
        fast_refresh_settings(2),
        &[],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        trigger: RefreshTrigger::Remerge,
                        priority: None,
                        pages_relinked: false,
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.pages_refreshed == 0);
    ensure!(outcome.merges_run == 1);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);
    ensure!(fake.state.lock().expect("lock").fetch_calls.is_empty());
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_signals_during_record_refresh_and_synopsis_trigger_another_cycle()
-> anyhow::Result<()> {
    let record_entered = Arc::new(Notify::new());
    let synopsis_entered = Arc::new(Notify::new());
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.has_changed_synopses = true;
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::Kitsu, "2", false, Some(i64::MAX / 2)),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
        guard.synopsis_delay = Duration::from_millis(600);
        guard.synopsis_entered = Some(synopsis_entered.clone());
        guard.record_refresh_delay = Duration::from_millis(600);
        guard.record_refresh_entered = Some(record_entered.clone());
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-final-signals",
        fake.clone(),
        fast_refresh_settings(1),
        &[Provider::Mal, Provider::Kitsu],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        trigger: RefreshTrigger::User,
                        priority: NonZeroU32::new(1),
                        pages_relinked: false,
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            synopsis_entered.notified().await;
            handle
                .signal(
                    RefreshAnimeWorkflow::merge_requested,
                    (),
                    WorkflowSignalOptions::default(),
                )
                .await?;
            record_entered.notified().await;
            handle
                .signal(
                    RefreshAnimeWorkflow::fetch_requested,
                    FetchRequestedSignal {
                        providers: vec![Provider::Kitsu],
                        max_age: Duration::ZERO,
                        priority: NonZeroU32::new(1),
                    },
                    WorkflowSignalOptions::default(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.merges_run >= 2);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);
    let state = fake.state.lock().expect("lock");
    ensure!(
        state
            .fetch_calls
            .iter()
            .any(|(key, _)| key.provider == Provider::Kitsu)
    );
    ensure!(state.recorded_outcomes.len() >= 2);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_handles_early_merge_wait_condition_and_blocked_page()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_delays
            .insert(Provider::AniList, Duration::from_millis(1_800));
        guard
            .page_errors
            .insert(Provider::AniList, PageErrorMode::Blocked);
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::AniList, "2", false, None),
            ],
            entity_merge_version: Some(2),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-early",
        fake.clone(),
        RefreshSettings::new(7, 2, 1, 1),
        &[Provider::Mal, Provider::AniList],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        trigger: RefreshTrigger::Sweep,
                        priority: NonZeroU32::new(3),
                        pages_relinked: false,
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
                    .build(),
                )
                .await?;
            tokio::time::sleep(Duration::from_millis(1_200)).await;
            handle
                .signal(
                    RefreshAnimeWorkflow::fetch_requested,
                    FetchRequestedSignal {
                        providers: vec![Provider::Mal],
                        max_age: Duration::ZERO,
                        priority: NonZeroU32::new(1),
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
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.final_outcome == RefreshOutcome::PagesFailed);
    ensure!(outcome.pages_failed == 1);
    ensure!(outcome.merges_run >= 2);
    let state = fake.state.lock().expect("lock");
    ensure!(state.recorded_page_failures.len() == 1);
    ensure!(state.recorded_page_failures[0].provider == Provider::AniList);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_records_page_failures_for_non_retryable_and_timeout()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_errors
            .insert(Provider::AniList, PageErrorMode::NonRetryable);
        guard
            .page_delays
            .insert(Provider::Mal, Duration::from_secs(5));
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::AniList, "1", false, None),
                build_page_summary(Provider::Mal, "2", false, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-page-failures",
        fake.clone(),
        fast_refresh_settings(1).with_refresh_page_timeout(1),
        &[Provider::AniList, Provider::Mal],
        Some(Duration::from_millis(100)),
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.pages_failed == 2);
    ensure!(outcome.final_outcome == RefreshOutcome::PagesFailed);
    let state = fake.state.lock().expect("lock");
    ensure!(state.recorded_page_failures.len() == 2);
    ensure!(
        state
            .recorded_page_failures
            .iter()
            .any(|key| key.provider == Provider::AniList)
    );
    ensure!(
        state
            .recorded_page_failures
            .iter()
            .any(|key| key.provider == Provider::Mal)
    );
    Ok(())
}

async fn assert_workflow_failure_records_failed(
    build_id: &str,
    setup: impl FnOnce(&mut FakeState),
) -> anyhow::Result<()> {
    let fake = FakePorts::default();
    setup(&mut fake.state.lock().expect("lock"));
    let anime_id = Uuid::now_v7();
    let failed = run_with_workers(
        build_id,
        fake.clone(),
        fast_refresh_settings(2),
        &[],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        trigger: RefreshTrigger::User,
                        priority: NonZeroU32::new(1),
                        pages_relinked: true,
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: Result<RefreshAnimeOutcome, _> = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await?;
            Ok(result.is_err())
        },
    )
    .await?;

    ensure!(failed, "{build_id} must fail workflow");
    let state = fake.state.lock().expect("lock");
    ensure!(state.recorded_outcomes.len() == 1);
    ensure!(state.recorded_outcomes[0].1 == RefreshOutcome::Failed);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_records_failed_when_list_pages_fails() -> anyhow::Result<()> {
    assert_workflow_failure_records_failed("test-fail-list", |guard| {
        guard.list_pages_fail = true;
    })
    .await
}

#[tokio::test]
async fn refresh_anime_workflow_records_failed_when_merge_fails_non_retryably() -> anyhow::Result<()>
{
    assert_workflow_failure_records_failed("test-fail-merge", |guard| {
        guard.merge_fail_non_retryable = true;
        guard.pages.entity_merge_version = Some(0);
    })
    .await
}

#[tokio::test]
async fn refresh_anime_workflow_records_failed_when_synopsis_start_fails() -> anyhow::Result<()> {
    assert_workflow_failure_records_failed("test-fail-synopsis", |guard| {
        guard.has_changed_synopses = true;
        guard.synopsis_start_fail = true;
        guard.pages.entity_merge_version = Some(0);
    })
    .await
}

#[tokio::test]
async fn refresh_anime_workflow_propagates_error_when_cleanup_record_refresh_also_fails()
-> anyhow::Result<()> {
    assert_workflow_failure_records_failed("test-fail-cleanup", |guard| {
        guard.list_pages_fail = true;
        guard.record_refresh_fail = true;
    })
    .await
}

#[tokio::test]
async fn refresh_anime_workflow_cancellation_cancels_in_flight_pages_and_records_failed()
-> anyhow::Result<()> {
    let mal_started = Arc::new(Notify::new());
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_delays
            .insert(Provider::Mal, Duration::from_secs(30));
        guard
            .page_started
            .insert(Provider::Mal, mal_started.clone());
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Kitsu, "1", false, None),
                build_page_summary(Provider::Mal, "2", false, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let cancelled = run_with_workers(
        "test-cancel-wf",
        fake.clone(),
        RefreshSettings::new(7, 1, 60, 60),
        &[Provider::Kitsu, Provider::Mal],
        Some(Duration::from_millis(50)),
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            mal_started.notified().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
            handle.cancel(WorkflowCancelOptions::default()).await?;
            let result: Result<RefreshAnimeOutcome, _> = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await?;
            Ok(result.is_err())
        },
    )
    .await?;

    ensure!(cancelled);
    let state = fake.state.lock().expect("lock");
    ensure!(state.merge_calls == 0);
    ensure!(state.recorded_page_failures.is_empty());
    ensure!(state.recorded_outcomes == vec![(anime_id, RefreshOutcome::Failed, 7)]);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_coalesces_late_pages_and_cancels_unused_timer() -> anyhow::Result<()>
{
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_delays
            .insert(Provider::Kitsu, Duration::from_millis(1_400));
        guard
            .page_delays
            .insert(Provider::AniList, Duration::from_millis(1_700));
        guard.page_changes.insert(Provider::AniList, false);
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::Kitsu, "2", false, None),
                build_page_summary(Provider::AniList, "3", false, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-coalesce",
        fake.clone(),
        RefreshSettings::new(7, 1, 1, 5),
        &[Provider::Mal, Provider::Kitsu, Provider::AniList],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.pages_refreshed == 3);
    ensure!(outcome.merges_run == 2);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_record_refresh_failure_does_not_double_record() -> anyhow::Result<()>
{
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.record_refresh_fail = true;
        guard.pages = ListEntityPagesOutput {
            pages: Vec::new(),
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let failed = run_with_workers(
        "test-record-fail",
        fake.clone(),
        fast_refresh_settings(1),
        &[],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: Result<RefreshAnimeOutcome, _> = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await?;
            Ok(result.is_err())
        },
    )
    .await?;

    ensure!(failed);
    ensure!(fake.state.lock().expect("lock").recorded_outcomes.len() == 1);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_mid_batch_merge_failure_cancels_remaining_pages()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard.merge_fail_non_retryable = true;
        guard
            .page_delays
            .insert(Provider::AniList, Duration::from_secs(10));
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::AniList, "2", false, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let failed = run_with_workers(
        "test-mid-batch-fail",
        fake.clone(),
        RefreshSettings::new(7, 1, 1, 1),
        &[Provider::Mal, Provider::AniList],
        Some(Duration::from_millis(50)),
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: Result<RefreshAnimeOutcome, _> = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await?;
            Ok(result.is_err())
        },
    )
    .await?;

    ensure!(failed);
    let state = fake.state.lock().expect("lock");
    ensure!(state.recorded_outcomes == vec![(anime_id, RefreshOutcome::Failed, 7)]);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_mid_batch_merge_signal_cancels_timer_and_tracks_reported_failure()
-> anyhow::Result<()> {
    let kitsu_started = Arc::new(Notify::new());
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_delays
            .insert(Provider::Kitsu, Duration::from_millis(800));
        guard
            .page_started
            .insert(Provider::Kitsu, kitsu_started.clone());
        guard.page_reported_failed.insert(Provider::Kitsu, true);
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::Mal, "1", false, None),
                build_page_summary(Provider::Kitsu, "2", false, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let outcome = run_with_workers(
        "test-mid-batch-signal",
        fake.clone(),
        RefreshSettings::new(7, 1, 60, 60),
        &[Provider::Mal, Provider::Kitsu],
        None,
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            kitsu_started.notified().await;
            tokio::time::sleep(Duration::from_millis(150)).await;
            handle
                .signal(
                    RefreshAnimeWorkflow::merge_requested,
                    (),
                    WorkflowSignalOptions::default(),
                )
                .await?;
            let result: RefreshAnimeOutcome = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            Ok(result)
        },
    )
    .await?;

    ensure!(outcome.merges_run == 1);
    ensure!(outcome.pages_failed == 1);
    ensure!(outcome.final_outcome == RefreshOutcome::PagesFailed);
    Ok(())
}

#[tokio::test]
async fn refresh_anime_workflow_record_page_failure_error_cancels_remaining_pages_and_records_failed()
-> anyhow::Result<()> {
    let fake = FakePorts::default();
    {
        let mut guard = fake.state.lock().expect("lock");
        guard
            .page_errors
            .insert(Provider::AniList, PageErrorMode::NonRetryable);
        guard.record_page_failure_fail = true;
        guard
            .page_delays
            .insert(Provider::Mal, Duration::from_secs(10));
        guard.pages = ListEntityPagesOutput {
            pages: vec![
                build_page_summary(Provider::AniList, "1", false, None),
                build_page_summary(Provider::Mal, "2", false, None),
            ],
            entity_merge_version: Some(1),
            refresh_settings: RefreshSettings::default(),
        };
    }

    let anime_id = Uuid::now_v7();
    let failed = run_with_workers(
        "test-record-page-fail",
        fake.clone(),
        fast_refresh_settings(1),
        &[Provider::AniList, Provider::Mal],
        Some(Duration::from_millis(50)),
        |client| async move {
            let handle = client
                .start_workflow(
                    RefreshAnimeWorkflow::run,
                    RefreshAnimeInput {
                        anime_id,
                        ..RefreshAnimeInput::default()
                    },
                    WorkflowStartOptions::new(
                        UPDATE_SERVICE_TASK_QUEUE,
                        build_refresh_anime_workflow_id(anime_id),
                    )
                    .build(),
                )
                .await?;
            let result: Result<RefreshAnimeOutcome, _> = tokio::time::timeout(
                TEST_WAIT,
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await?;
            Ok(result.is_err())
        },
    )
    .await?;

    ensure!(failed);
    let state = fake.state.lock().expect("lock");
    ensure!(state.recorded_outcomes == vec![(anime_id, RefreshOutcome::Failed, 7)]);
    Ok(())
}
