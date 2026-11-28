//! Integration tests for `GrpcCatalogueAdapter`, `build_*_client`, and `TemporalWorkflowStarter`.

#![expect(
    clippy::too_many_lines,
    reason = "integration test covers full gRPC roundtrip assertions"
)]

mod support;

use std::error::Error as _;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use echora_domain::{EntityType, PageKey, Provider};
use support::{MockCatalogueState, start_configurable_mock_grpc_server};
use temporal_workflows::testing::start_local_environment;
use temporal_workflows::{
    CatalogueDataPort, MergeAttemptOutcome, RefreshOutcome, RefreshTrigger,
    StartEntityWorkflowsInput, WorkflowStarterPort,
};
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::{Code, Status};
use update_service::postgres_service_proto::v1::EntityPage;
use update_service::shared_proto::v1::{EntityType as ProtoEntityType, PageKey as ProtoPageKey};
use update_service::{
    FETCH_ERROR_METADATA_KEY, FETCH_ERROR_PROVIDER_BLOCKED, GrpcCatalogueAdapter,
    TemporalWorkflowStarter, build_enrichment_client, build_postgres_client, map_grpc_status,
};
use uuid::Uuid;

#[test]
fn map_grpc_status_classifies_permanent_blocked_and_transient_codes_and_keeps_source() {
    for code in [
        Code::InvalidArgument,
        Code::NotFound,
        Code::PermissionDenied,
        Code::ResourceExhausted,
        Code::FailedPrecondition,
        Code::Unimplemented,
    ] {
        let err = map_grpc_status(Status::new(code, "permanent"));
        assert!(err.is_non_retryable(), "{code:?} must be non-retryable");
        let source = err
            .source()
            .and_then(|src| src.downcast_ref::<Status>())
            .expect("PortError must retain tonic::Status source");
        assert_eq!(source.code(), code);
    }

    for code in [Code::Unavailable, Code::DeadlineExceeded, Code::Internal] {
        let err = map_grpc_status(Status::new(code, "transient"));
        assert!(err.is_retryable(), "{code:?} must be retryable");
        let source = err
            .source()
            .and_then(|src| src.downcast_ref::<Status>())
            .expect("PortError must retain tonic::Status source");
        assert_eq!(source.code(), code);
    }

    let mut blocked_status = Status::unavailable("HTTP 423");
    blocked_status.metadata_mut().insert(
        FETCH_ERROR_METADATA_KEY,
        MetadataValue::from_static(FETCH_ERROR_PROVIDER_BLOCKED),
    );
    let blocked_err = map_grpc_status(blocked_status);
    assert!(blocked_err.is_provider_blocked());
    assert!(blocked_err.source().is_some());
}

#[tokio::test]
async fn grpc_catalogue_adapter_covers_happy_blocked_transient_aborted_and_page_sources()
-> anyhow::Result<()> {
    let state = Arc::new(Mutex::new(MockCatalogueState {
        pages: vec![
            EntityPage {
                page_key: Some(ProtoPageKey {
                    provider: "mal".to_owned(),
                    entity_type: ProtoEntityType::Anime as i32,
                    address: "1".to_owned(),
                }),
                is_partial_source: false,
                fetched_at: None,
                is_dead: false,
                fetch_failed_at: None,
                fetch_failures: 0,
            },
            EntityPage {
                page_key: Some(ProtoPageKey {
                    provider: "kitsu".to_owned(),
                    entity_type: ProtoEntityType::Episode as i32,
                    address: "e1".to_owned(),
                }),
                is_partial_source: false,
                fetched_at: None,
                is_dead: false,
                fetch_failed_at: None,
                fetch_failures: 0,
            },
            EntityPage {
                page_key: Some(ProtoPageKey {
                    provider: "anilist".to_owned(),
                    entity_type: ProtoEntityType::Character as i32,
                    address: "3".to_owned(),
                }),
                is_partial_source: true,
                fetched_at: None,
                is_dead: false,
                fetch_failed_at: None,
                fetch_failures: 0,
            },
        ],
        merge_version: Some(3),
        ..MockCatalogueState::default()
    }));

    let endpoint = start_configurable_mock_grpc_server(state.clone()).await?;
    let pg_channel = Channel::from_shared(endpoint.clone())?.connect().await?;
    let enr_channel = Channel::from_shared(endpoint)?.connect().await?;

    let adapter = GrpcCatalogueAdapter::new(
        build_postgres_client(pg_channel),
        build_enrichment_client(enr_channel),
    );
    let anime_id = Uuid::now_v7();

    let listed = adapter.list_entity_pages(anime_id).await?;
    assert_eq!(listed.pages.len(), 3);
    assert_eq!(listed.entity_merge_version, Some(3));
    assert!(format!("{adapter:?}").contains("GrpcCatalogueAdapter"));

    state.lock().expect("lock").invalid_page_provider = true;
    let invalid_err = adapter
        .list_entity_pages(anime_id)
        .await
        .expect_err("unknown provider must fail non-retryably");
    assert!(invalid_err.is_non_retryable());
    state.lock().expect("lock").invalid_page_provider = false;

    state.lock().expect("lock").invalid_page_entity_type = true;
    let bad_entity_err = adapter
        .list_entity_pages(anime_id)
        .await
        .expect_err("unspecified entity_type must fail non-retryably");
    assert!(bad_entity_err.is_non_retryable());
    state.lock().expect("lock").invalid_page_entity_type = false;

    state.lock().expect("lock").missing_page_key = true;
    let missing_key_err = adapter
        .list_entity_pages(anime_id)
        .await
        .expect_err("missing page_key must fail non-retryably");
    assert!(missing_key_err.is_non_retryable());
    state.lock().expect("lock").missing_page_key = false;

    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "1").expect("valid page key");
    let refreshed = adapter
        .fetch_and_store_page(page_key.clone(), Duration::from_secs(3600))
        .await?;
    assert!(refreshed.has_changed);
    assert!(!refreshed.has_failed);
    assert!(adapter.max_message_bytes_seen() > 0);

    state.lock().expect("lock").fail_fetch_page_transient = true;
    let transient_err = adapter
        .fetch_and_store_page(page_key.clone(), Duration::from_secs(3600))
        .await
        .expect_err("transient error must propagate for Temporal retries");
    assert!(transient_err.is_retryable());
    state.lock().expect("lock").fail_fetch_page_transient = false;

    state.lock().expect("lock").fail_fetch_page_blocked = true;
    let blocked_err = adapter
        .fetch_and_store_page(page_key.clone(), Duration::from_secs(3600))
        .await
        .expect_err("blocked error must surface as provider_blocked");
    assert!(blocked_err.is_provider_blocked());
    state.lock().expect("lock").fail_fetch_page_blocked = false;

    let failed_at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    for (entity_type, address) in [
        (EntityType::Anime, "1"),
        (EntityType::Episode, "1:1"),
        (EntityType::Character, "1"),
    ] {
        adapter
            .record_page_failure(
                PageKey::new(Provider::Mal, entity_type, address).expect("valid page key"),
                failed_at,
            )
            .await?;
    }
    {
        let guard = state.lock().expect("lock");
        let last_stored = guard.stored_pages.last().expect("recorded failure page");
        assert_eq!(last_stored.responses.len(), 0);
        assert!(!last_stored.is_not_found);
        assert_eq!(
            last_stored.fetched_at.as_ref().map(|ts| ts.seconds),
            Some(1_700_000_000)
        );
    }

    let saved = adapter.merge_anime_attempt(anime_id, 2).await?;
    match saved {
        MergeAttemptOutcome::Saved(output) => {
            assert!(output.has_changed);
            assert!(!output.has_changed_synopses);
            assert_eq!(output.changed_episode_ids, vec!["ep-10"]);
        }
        MergeAttemptOutcome::VersionConflict(msg) => panic!("unexpected conflict: {msg}"),
    }
    assert_eq!(state.lock().expect("lock").saved_page_sources.len(), 1);
    assert_eq!(state.lock().expect("lock").saved_merge_versions, vec![2]);

    state.lock().expect("lock").aborted_save = true;
    let conflicted = adapter.merge_anime_attempt(anime_id, 2).await?;
    assert!(matches!(
        conflicted,
        MergeAttemptOutcome::VersionConflict(_)
    ));
    state.lock().expect("lock").aborted_save = false;

    state.lock().expect("lock").fail_save_anime_invalid = true;
    let invalid_save = adapter
        .merge_anime_attempt(anime_id, 0)
        .await
        .expect_err("invalid save_anime must fail non-retryably");
    assert!(invalid_save.is_non_retryable());
    state.lock().expect("lock").fail_save_anime_invalid = false;

    for outcome in [
        RefreshOutcome::Succeeded,
        RefreshOutcome::PagesFailed,
        RefreshOutcome::Failed,
    ] {
        adapter.record_refresh(anime_id, outcome, 7).await?;
    }
    assert_eq!(state.lock().expect("lock").recorded.len(), 3);

    state.lock().expect("lock").empty_fetch_page = true;
    let missing_page_err = adapter
        .fetch_and_store_page(page_key.clone(), Duration::ZERO)
        .await
        .expect_err("missing page must fail non-retryably");
    assert!(missing_page_err.is_non_retryable());
    state.lock().expect("lock").empty_fetch_page = false;

    state.lock().expect("lock").fail_enrichment_invalid = true;
    let bad_merge_err = adapter
        .merge_anime_attempt(anime_id, 1)
        .await
        .expect_err("invalid_argument merge must fail non-retryably");
    assert!(bad_merge_err.is_non_retryable());
    state.lock().expect("lock").fail_enrichment_invalid = false;

    state.lock().expect("lock").fail_postgres_not_found = true;
    assert!(adapter.list_entity_pages(anime_id).await.is_err());
    assert!(
        adapter
            .record_page_failure(page_key, failed_at)
            .await
            .is_err()
    );
    assert!(adapter.merge_anime_attempt(anime_id, 1).await.is_err());
    assert!(
        adapter
            .record_refresh(anime_id, RefreshOutcome::Succeeded, 7)
            .await
            .is_err()
    );

    Ok(())
}

#[tokio::test]
async fn temporal_workflow_starter_starts_and_deduplicates_downstream_workflows()
-> anyhow::Result<()> {
    let environment = start_local_environment(None).await?;
    let starter = TemporalWorkflowStarter::new(environment.client().clone());
    assert!(format!("{starter:?}").contains("TemporalWorkflowStarter"));
    let anime_id = Uuid::now_v7();

    let input = StartEntityWorkflowsInput {
        anime_id,
        episode_ids: vec!["ep-1".to_owned(), "ep-2".to_owned(), "ep-3".to_owned()],
        start_synopsis: true,
        trigger: RefreshTrigger::User,
        priority: NonZeroU32::new(1),
    };
    let started_count = starter.start_entity_workflows(input.clone()).await?;
    assert_eq!(started_count, 4);

    let deduplicated_count = starter.start_entity_workflows(input).await?;
    assert_eq!(deduplicated_count, 4);

    let empty_count = starter
        .start_entity_workflows(StartEntityWorkflowsInput {
            anime_id: Uuid::now_v7(),
            episode_ids: Vec::new(),
            start_synopsis: false,
            trigger: RefreshTrigger::Sweep,
            priority: None,
        })
        .await?;
    assert_eq!(empty_count, 0);

    environment.shutdown().await?;

    let stopped_err = starter
        .start_entity_workflows(StartEntityWorkflowsInput {
            anime_id: Uuid::now_v7(),
            episode_ids: vec!["ep-after-stop".to_owned()],
            start_synopsis: false,
            trigger: RefreshTrigger::Sweep,
            priority: None,
        })
        .await
        .expect_err("starting workflow after server shutdown must fail retryably");
    assert!(stopped_err.is_retryable());
    Ok(())
}
