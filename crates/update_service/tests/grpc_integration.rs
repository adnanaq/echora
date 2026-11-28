//! Integration tests for `GrpcCatalogueAdapter`, `build_*_client`, and `TemporalWorkflowStarter`.

#![expect(
    clippy::expect_used,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    reason = "integration test mock flags and full gRPC roundtrip assertions"
)]

use std::sync::{Arc, Mutex};

use temporal_workflows::testing::start_local_environment;
use temporal_workflows::{
    CatalogueDataPort, EntityType, MergeAttemptOutcome, Provider, RecordRefreshInput,
    RefreshOutcome, RefreshPageKey, RefreshTrigger, StartEntityWorkflowsInput, WorkflowStarterPort,
};
use tokio::net::TcpListener;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Server};
use tonic::{Code, Request, Response, Status};
use update_service::enrichment_service_proto::v1::enrichment_service_server::{
    EnrichmentService, EnrichmentServiceServer,
};
use update_service::enrichment_service_proto::v1::{
    FetchPageRequest, FetchPageResponse, HealthRequest, HealthResponse, MergeAnimeRequest,
    MergeAnimeResponse, MergeCharacterRequest, MergeCharacterResponse, MergeEpisodeRequest,
    MergeEpisodeResponse, RunPipelineRequest, RunPipelineResponse,
};
use update_service::postgres_service_proto::v1::postgres_service_server::{
    PostgresService, PostgresServiceServer,
};
use update_service::postgres_service_proto::v1::{
    EntityPage, GetAnimeResponse, GetCharacterResponse, GetEntityRequest, GetEpisodeResponse,
    ListDuePagesRequest, ListDuePagesResponse, ListDueRequest, ListDueResponse,
    ListEntityPagesRequest, ListEntityPagesResponse, ListEntityVersionsRequest,
    ListEntityVersionsResponse, ListEpisodesResponse, ListRequestedMergesRequest,
    ListRequestedMergesResponse, ListScoreAnimeRequest, ListScoreAnimeResponse,
    ListUpcomingEpisodesRequest, ListUpcomingEpisodesResponse, LoadAnimeRequest, LoadAnimeResponse,
    LoadCharacterRequest, LoadCharacterResponse, LoadEpisodeRequest, LoadEpisodeResponse,
    MarkEpisodeAiredRequest, MarkEpisodeAiredResponse, RecordRefreshRequest, RecordRefreshResponse,
    SaveAnimeRequest, SaveAnimeResponse, SaveCharacterRequest, SaveCharacterResponse,
    SaveEpisodeRequest, SaveEpisodeResponse, StorePageRequest, StorePageResponse,
};
use update_service::shared_proto::v1::{
    EntityType as ProtoEntityType, PageKey, PageResponse, PageSource, StoredPage,
};
use update_service::{
    FETCH_ERROR_METADATA_KEY, FETCH_ERROR_PROVIDER_BLOCKED, GrpcCatalogueAdapter,
    TemporalWorkflowStarter, build_enrichment_client, build_postgres_client, map_grpc_status,
};

#[derive(Default)]
struct MockGrpcState {
    aborted_save: bool,
    fail_fetch_page_transient: bool,
    fail_fetch_page_blocked: bool,
    empty_fetch_page: bool,
    fail_postgres_not_found: bool,
    fail_enrichment_invalid: bool,
    invalid_page_provider: bool,
    saved_page_sources: Vec<PageSource>,
    stored_pages: Vec<StoredPage>,
    recorded: Vec<(String, i32)>,
}

#[derive(Clone, Default)]
struct MockServices {
    state: Arc<Mutex<MockGrpcState>>,
}

#[tonic::async_trait]
impl PostgresService for MockServices {
    async fn list_entity_pages(
        &self,
        _request: Request<ListEntityPagesRequest>,
    ) -> Result<Response<ListEntityPagesResponse>, Status> {
        let (fail_not_found, invalid_provider) = {
            let guard = self.state.lock().expect("lock");
            (guard.fail_postgres_not_found, guard.invalid_page_provider)
        };
        if fail_not_found {
            return Err(Status::not_found("missing anime"));
        }
        let provider = if invalid_provider { "unknown" } else { "mal" };
        Ok(Response::new(ListEntityPagesResponse {
            pages: vec![
                EntityPage {
                    page_key: Some(PageKey {
                        provider: provider.to_owned(),
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
                    page_key: Some(PageKey {
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
                    page_key: Some(PageKey {
                        provider: "anilist".to_owned(),
                        entity_type: ProtoEntityType::Character as i32,
                        address: "c1".to_owned(),
                    }),
                    is_partial_source: true,
                    fetched_at: None,
                    is_dead: false,
                    fetch_failed_at: None,
                    fetch_failures: 0,
                },
            ],
            merge_version: Some(3),
        }))
    }

    async fn load_anime(
        &self,
        _request: Request<LoadAnimeRequest>,
    ) -> Result<Response<LoadAnimeResponse>, Status> {
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing anime"));
        }
        Ok(Response::new(LoadAnimeResponse {
            version: 1,
            source_pages: vec![StoredPage {
                page_key: Some(PageKey {
                    provider: "mal".to_owned(),
                    entity_type: ProtoEntityType::Anime as i32,
                    address: "1".to_owned(),
                }),
                fetched_at: None,
                is_not_found: false,
                is_dead: false,
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "text/html".to_owned(),
                    body: b"<html>anime</html>".to_vec(),
                }],
            }],
            partial_source_pages: Vec::new(),
            corrections: Vec::new(),
            known_episodes: Vec::new(),
        }))
    }

    async fn store_page(
        &self,
        request: Request<StorePageRequest>,
    ) -> Result<Response<StorePageResponse>, Status> {
        let mut guard = self.state.lock().expect("lock");
        if guard.fail_postgres_not_found {
            return Err(Status::not_found("missing page"));
        }
        if let Some(page) = request.into_inner().page {
            guard.stored_pages.push(page);
        }
        Ok(Response::new(StorePageResponse { has_changed: true }))
    }

    async fn save_anime(
        &self,
        request: Request<SaveAnimeRequest>,
    ) -> Result<Response<SaveAnimeResponse>, Status> {
        let mut guard = self.state.lock().expect("lock");
        if guard.fail_postgres_not_found {
            return Err(Status::not_found("missing save target"));
        }
        if guard.aborted_save {
            return Err(Status::aborted("version conflict"));
        }
        guard.saved_page_sources = request.into_inner().page_sources;
        Ok(Response::new(SaveAnimeResponse {
            has_changed: true,
            version: 2,
            created_episode_ids: vec!["ep-10".to_owned()],
            updated_episode_ids: Vec::new(),
        }))
    }

    async fn record_refresh(
        &self,
        request: Request<RecordRefreshRequest>,
    ) -> Result<Response<RecordRefreshResponse>, Status> {
        let mut guard = self.state.lock().expect("lock");
        if guard.fail_postgres_not_found {
            return Err(Status::not_found("missing entity"));
        }
        let req = request.into_inner();
        guard.recorded.push((req.entity_id, req.outcome));
        Ok(Response::new(RecordRefreshResponse::default()))
    }

    async fn load_episode(
        &self,
        _request: Request<LoadEpisodeRequest>,
    ) -> Result<Response<LoadEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn save_episode(
        &self,
        _request: Request<SaveEpisodeRequest>,
    ) -> Result<Response<SaveEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn mark_episode_aired(
        &self,
        _request: Request<MarkEpisodeAiredRequest>,
    ) -> Result<Response<MarkEpisodeAiredResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn load_character(
        &self,
        _request: Request<LoadCharacterRequest>,
    ) -> Result<Response<LoadCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn save_character(
        &self,
        _request: Request<SaveCharacterRequest>,
    ) -> Result<Response<SaveCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_anime(
        &self,
        _request: Request<ListDueRequest>,
    ) -> Result<Response<ListDueResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_episodes(
        &self,
        _request: Request<ListDueRequest>,
    ) -> Result<Response<ListDueResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_characters(
        &self,
        _request: Request<ListDueRequest>,
    ) -> Result<Response<ListDueResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_upcoming_episodes(
        &self,
        _request: Request<ListUpcomingEpisodesRequest>,
    ) -> Result<Response<ListUpcomingEpisodesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_score_anime(
        &self,
        _request: Request<ListScoreAnimeRequest>,
    ) -> Result<Response<ListScoreAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_requested_merges(
        &self,
        _request: Request<ListRequestedMergesRequest>,
    ) -> Result<Response<ListRequestedMergesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn get_anime(
        &self,
        _request: Request<GetEntityRequest>,
    ) -> Result<Response<GetAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn get_episode(
        &self,
        _request: Request<GetEntityRequest>,
    ) -> Result<Response<GetEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_episodes(
        &self,
        _request: Request<GetEntityRequest>,
    ) -> Result<Response<ListEpisodesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn get_character(
        &self,
        _request: Request<GetEntityRequest>,
    ) -> Result<Response<GetCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_entity_versions(
        &self,
        _request: Request<ListEntityVersionsRequest>,
    ) -> Result<Response<ListEntityVersionsResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_pages(
        &self,
        _request: Request<ListDuePagesRequest>,
    ) -> Result<Response<ListDuePagesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
}

#[tonic::async_trait]
impl EnrichmentService for MockServices {
    async fn fetch_page(
        &self,
        _request: Request<FetchPageRequest>,
    ) -> Result<Response<FetchPageResponse>, Status> {
        let (fail_transient, fail_blocked, empty_fetch_page) = {
            let guard = self.state.lock().expect("lock");
            (
                guard.fail_fetch_page_transient,
                guard.fail_fetch_page_blocked,
                guard.empty_fetch_page,
            )
        };
        if fail_blocked {
            let mut status = Status::unavailable("blocked by WAF");
            status.metadata_mut().insert(
                FETCH_ERROR_METADATA_KEY,
                MetadataValue::from_static(FETCH_ERROR_PROVIDER_BLOCKED),
            );
            return Err(status);
        }
        if fail_transient {
            return Err(Status::unavailable("provider timeout"));
        }
        if empty_fetch_page {
            return Ok(Response::new(FetchPageResponse { page: None }));
        }
        Ok(Response::new(FetchPageResponse {
            page: Some(StoredPage {
                page_key: Some(PageKey {
                    provider: "mal".to_owned(),
                    entity_type: ProtoEntityType::Anime as i32,
                    address: "1".to_owned(),
                }),
                fetched_at: None,
                is_not_found: false,
                is_dead: false,
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "text/html".to_owned(),
                    body: b"<html>fetched</html>".to_vec(),
                }],
            }),
        }))
    }

    async fn merge_anime(
        &self,
        _request: Request<MergeAnimeRequest>,
    ) -> Result<Response<MergeAnimeResponse>, Status> {
        if self.state.lock().expect("lock").fail_enrichment_invalid {
            return Err(Status::invalid_argument("bad merge"));
        }
        Ok(Response::new(MergeAnimeResponse {
            anime: None,
            episodes: Vec::new(),
            relation_claims: Vec::new(),
            provider_synopses_hash: "syn-hash".to_owned(),
            page_sources: vec![PageSource {
                page_key: Some(PageKey {
                    provider: "mal".to_owned(),
                    entity_type: ProtoEntityType::Anime as i32,
                    address: "1".to_owned(),
                }),
                source: "https://myanimelist.net/anime/1/Cowboy_Bebop".to_owned(),
                title: "Cowboy Bebop".to_owned(),
            }],
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn run_pipeline(
        &self,
        _request: Request<RunPipelineRequest>,
    ) -> Result<Response<RunPipelineResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn merge_episode(
        &self,
        _request: Request<MergeEpisodeRequest>,
    ) -> Result<Response<MergeEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn merge_character(
        &self,
        _request: Request<MergeCharacterRequest>,
    ) -> Result<Response<MergeCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
}

#[test]
fn map_grpc_status_classifies_permanent_blocked_and_transient_codes() {
    for code in [
        Code::InvalidArgument,
        Code::NotFound,
        Code::PermissionDenied,
        Code::ResourceExhausted,
        Code::FailedPrecondition,
        Code::Unimplemented,
    ] {
        assert!(
            map_grpc_status(&Status::new(code, "permanent")).is_non_retryable(),
            "{code:?} must be non-retryable"
        );
    }

    for code in [Code::Unavailable, Code::DeadlineExceeded, Code::Internal] {
        assert!(
            map_grpc_status(&Status::new(code, "transient")).is_retryable(),
            "{code:?} must be retryable"
        );
    }

    let mut blocked_status = Status::unavailable("HTTP 423");
    blocked_status.metadata_mut().insert(
        FETCH_ERROR_METADATA_KEY,
        MetadataValue::from_static(FETCH_ERROR_PROVIDER_BLOCKED),
    );
    assert!(map_grpc_status(&blocked_status).is_provider_blocked());
}

#[tokio::test]
async fn grpc_catalogue_adapter_covers_happy_blocked_transient_aborted_and_page_sources()
-> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let mock = MockServices::default();

    let server_mock = mock.clone();
    let incoming = futures::stream::unfold(listener, |listener| async move {
        let accepted = listener.accept().await.map(|(stream, _addr)| stream);
        Some((accepted, listener))
    });
    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(
                PostgresServiceServer::new(server_mock.clone())
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .add_service(
                EnrichmentServiceServer::new(server_mock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(incoming)
            .await;
    });

    let endpoint = format!("http://{addr}");
    let pg_channel = Channel::from_shared(endpoint.clone())?.connect().await?;
    let enr_channel = Channel::from_shared(endpoint)?.connect().await?;

    let adapter = GrpcCatalogueAdapter::new(
        build_postgres_client(pg_channel),
        build_enrichment_client(enr_channel),
    );

    let listed = adapter.list_entity_pages("a1".to_owned()).await?;
    assert_eq!(listed.pages.len(), 3);
    assert_eq!(listed.entity_merge_version, Some(3));

    mock.state.lock().expect("lock").invalid_page_provider = true;
    let invalid_err = adapter
        .list_entity_pages("a1".to_owned())
        .await
        .expect_err("unknown provider must fail non-retryably");
    assert!(invalid_err.is_non_retryable());
    mock.state.lock().expect("lock").invalid_page_provider = false;

    let page_key = RefreshPageKey {
        provider: Provider::Mal,
        entity_type: EntityType::Anime,
        address: "1".to_owned(),
    };
    let refreshed = adapter.fetch_and_store_page(page_key.clone(), 3600).await?;
    assert!(refreshed.has_changed);
    assert!(!refreshed.has_failed);
    assert!(adapter.max_message_bytes_seen() > 0);

    mock.state.lock().expect("lock").fail_fetch_page_transient = true;
    let transient_err = adapter
        .fetch_and_store_page(page_key.clone(), 3600)
        .await
        .expect_err("transient error must propagate for Temporal retries");
    assert!(transient_err.is_retryable());
    mock.state.lock().expect("lock").fail_fetch_page_transient = false;

    mock.state.lock().expect("lock").fail_fetch_page_blocked = true;
    let blocked_err = adapter
        .fetch_and_store_page(page_key.clone(), 3600)
        .await
        .expect_err("blocked error must surface as provider_blocked");
    assert!(blocked_err.is_provider_blocked());
    mock.state.lock().expect("lock").fail_fetch_page_blocked = false;

    adapter.record_page_failure(page_key.clone()).await?;
    {
        let guard = mock.state.lock().expect("lock");
        let last_stored = guard.stored_pages.last().expect("recorded failure page");
        assert_eq!(last_stored.responses.len(), 0);
        assert!(!last_stored.is_not_found);
        assert!(last_stored.fetched_at.is_some());
    }

    let saved = adapter.merge_anime_attempt("a1".to_owned(), 2).await?;
    match saved {
        MergeAttemptOutcome::Saved(output) => {
            assert!(output.has_changed);
            assert!(!output.has_changed_synopses);
            assert_eq!(output.changed_episode_ids, vec!["ep-10"]);
        }
        MergeAttemptOutcome::VersionConflict(msg) => panic!("unexpected conflict: {msg}"),
    }
    assert_eq!(mock.state.lock().expect("lock").saved_page_sources.len(), 1);

    mock.state.lock().expect("lock").aborted_save = true;
    let conflicted = adapter.merge_anime_attempt("a1".to_owned(), 2).await?;
    assert!(matches!(
        conflicted,
        MergeAttemptOutcome::VersionConflict(_)
    ));
    mock.state.lock().expect("lock").aborted_save = false;

    for outcome in [
        RefreshOutcome::Succeeded,
        RefreshOutcome::PagesFailed,
        RefreshOutcome::Failed,
    ] {
        adapter
            .record_refresh(RecordRefreshInput {
                anime_id: "a1".to_owned(),
                outcome,
                refresh_after_days: 7,
            })
            .await?;
    }
    assert_eq!(mock.state.lock().expect("lock").recorded.len(), 3);

    mock.state.lock().expect("lock").empty_fetch_page = true;
    let missing_page_err = adapter
        .fetch_and_store_page(page_key.clone(), 0)
        .await
        .expect_err("missing page must fail non-retryably");
    assert!(missing_page_err.is_non_retryable());
    mock.state.lock().expect("lock").empty_fetch_page = false;

    mock.state.lock().expect("lock").fail_enrichment_invalid = true;
    let bad_merge_err = adapter
        .merge_anime_attempt("a1".to_owned(), 1)
        .await
        .expect_err("invalid_argument merge must fail non-retryably");
    assert!(bad_merge_err.is_non_retryable());
    mock.state.lock().expect("lock").fail_enrichment_invalid = false;

    mock.state.lock().expect("lock").fail_postgres_not_found = true;
    assert!(adapter.list_entity_pages("a1".to_owned()).await.is_err());
    assert!(adapter.record_page_failure(page_key).await.is_err());
    assert!(
        adapter
            .merge_anime_attempt("a1".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        adapter
            .record_refresh(RecordRefreshInput {
                anime_id: "a1".to_owned(),
                outcome: RefreshOutcome::Succeeded,
                refresh_after_days: 7,
            })
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

    let input = StartEntityWorkflowsInput {
        anime_id: "anime-xyz".to_owned(),
        episode_ids: vec!["ep-1".to_owned(), "ep-2".to_owned(), "ep-3".to_owned()],
        start_synopsis: true,
        trigger: RefreshTrigger::User,
        priority: 1,
    };
    let started_count = starter.start_entity_workflows(input.clone()).await?;
    assert_eq!(started_count, 4);

    // Calling again hits UseExisting conflict policy cleanly without erroring.
    let deduplicated_count = starter.start_entity_workflows(input).await?;
    assert_eq!(deduplicated_count, 4);

    let empty_count = starter
        .start_entity_workflows(StartEntityWorkflowsInput {
            anime_id: "anime-empty".to_owned(),
            episode_ids: Vec::new(),
            start_synopsis: false,
            trigger: RefreshTrigger::Sweep,
            priority: 0,
        })
        .await?;
    assert_eq!(empty_count, 0);

    environment.shutdown().await?;
    Ok(())
}
