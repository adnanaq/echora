//! Integration tests for `GrpcCatalogueAdapter`, `build_*_client`, and `TemporalWorkflowStarter`.

#![allow(
    clippy::expect_used,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    reason = "integration test mock flags and full gRPC roundtrip assertions"
)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use temporal_workflows::{
    CatalogueDataPort, CharacterFetchTarget, FetchRequestedSignal, MergeAttemptOutcome,
    REFRESH_TRIGGER_SEARCH_ATTRIBUTE, RecordRefreshInput, RefreshEntityKind, RefreshOutcomeKind,
    RefreshPageKey, StartEntityWorkflowsInput, WorkflowStarterPort,
    build_refresh_character_workflow_id, build_refresh_episode_workflow_id,
    build_write_synopsis_workflow_id,
};
use temporalio_client::grpc::{OperatorService as _, WorkflowService as _};
use temporalio_client::{Client, ClientOptions, Connection, ConnectionOptions};
use temporalio_common::protos::temporal::api::enums::v1::IndexedValueType;
use temporalio_common::protos::temporal::api::operatorservice::v1::AddSearchAttributesRequest;
use temporalio_common::protos::temporal::api::workflowservice::v1::RegisterNamespaceRequest;
use tokio::net::TcpListener;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};
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
use update_service::shared_proto::v1::{EntityType, PageKey, PageResponse, StoredPage};
use update_service::{
    GrpcCatalogueAdapter, TemporalWorkflowStarter, build_enrichment_client, build_postgres_client,
    map_grpc_status,
};
use url::Url;

#[derive(Default)]
struct MockGrpcState {
    aborted_save: bool,
    fail_fetch_page: bool,
    empty_fetch_page: bool,
    fail_postgres_not_found: bool,
    fail_enrichment_not_found: bool,
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
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing anime"));
        }
        Ok(Response::new(ListEntityPagesResponse {
            pages: vec![
                EntityPage {
                    page_key: Some(PageKey {
                        provider: "mal".to_owned(),
                        entity_type: EntityType::Anime as i32,
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
                        provider: "mal".to_owned(),
                        entity_type: EntityType::Character as i32,
                        address: "c1".to_owned(),
                    }),
                    is_partial_source: false,
                    fetched_at: None,
                    is_dead: false,
                    fetch_failed_at: None,
                    fetch_failures: 0,
                },
                EntityPage {
                    page_key: Some(PageKey {
                        provider: "mal".to_owned(),
                        entity_type: EntityType::Episode as i32,
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
                        provider: "mal".to_owned(),
                        entity_type: EntityType::Unspecified as i32,
                        address: "u1".to_owned(),
                    }),
                    is_partial_source: false,
                    fetched_at: None,
                    is_dead: false,
                    fetch_failed_at: None,
                    fetch_failures: 0,
                },
            ],
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
                    entity_type: EntityType::Anime as i32,
                    address: "1".to_owned(),
                }),
                fetched_at: None,
                is_not_found: false,
                is_dead: false,
                responses: vec![PageResponse::default()],
            }],
            partial_source_pages: vec![],
            corrections: vec![],
            known_episodes: vec![],
        }))
    }

    async fn store_page(
        &self,
        _request: Request<StorePageRequest>,
    ) -> Result<Response<StorePageResponse>, Status> {
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing page"));
        }
        Ok(Response::new(StorePageResponse { has_changed: true }))
    }

    async fn save_anime(
        &self,
        _request: Request<SaveAnimeRequest>,
    ) -> Result<Response<SaveAnimeResponse>, Status> {
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing save target"));
        }
        if self.state.lock().expect("lock").aborted_save {
            return Err(Status::aborted("version conflict"));
        }
        Ok(Response::new(SaveAnimeResponse {
            has_changed: true,
            version: 2,
            created_episode_ids: vec!["ep-10".to_owned()],
            updated_episode_ids: vec![],
        }))
    }

    async fn record_refresh(
        &self,
        request: Request<RecordRefreshRequest>,
    ) -> Result<Response<RecordRefreshResponse>, Status> {
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing entity"));
        }
        let req = request.into_inner();
        self.state
            .lock()
            .expect("lock")
            .recorded
            .push((req.entity_id, req.outcome));
        Ok(Response::new(RecordRefreshResponse::default()))
    }

    async fn load_episode(
        &self,
        _request: Request<LoadEpisodeRequest>,
    ) -> Result<Response<LoadEpisodeResponse>, Status> {
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing episode"));
        }
        Ok(Response::new(LoadEpisodeResponse {
            version: 1,
            anime_id: "a1".to_owned(),
            pages: vec![],
            corrections: vec![],
            list_entries: vec![],
        }))
    }

    async fn save_episode(
        &self,
        _request: Request<SaveEpisodeRequest>,
    ) -> Result<Response<SaveEpisodeResponse>, Status> {
        if self.state.lock().expect("lock").aborted_save {
            return Err(Status::aborted("episode version conflict"));
        }
        Ok(Response::new(SaveEpisodeResponse {
            has_changed: true,
            version: 2,
        }))
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
        if self.state.lock().expect("lock").fail_postgres_not_found {
            return Err(Status::not_found("missing character"));
        }
        Ok(Response::new(LoadCharacterResponse {
            version: 1,
            pages: vec![],
            corrections: vec![],
        }))
    }

    async fn save_character(
        &self,
        _request: Request<SaveCharacterRequest>,
    ) -> Result<Response<SaveCharacterResponse>, Status> {
        if self.state.lock().expect("lock").aborted_save {
            return Err(Status::aborted("character version conflict"));
        }
        Ok(Response::new(SaveCharacterResponse {
            has_changed: true,
            version: 2,
        }))
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
        let (fail_fetch_page, empty_fetch_page) = {
            let guard = self.state.lock().expect("lock");
            (guard.fail_fetch_page, guard.empty_fetch_page)
        };
        if fail_fetch_page {
            return Err(Status::unavailable("provider timeout"));
        }
        if empty_fetch_page {
            return Ok(Response::new(FetchPageResponse { page: None }));
        }
        Ok(Response::new(FetchPageResponse {
            page: Some(StoredPage {
                page_key: Some(PageKey {
                    provider: "mal".to_owned(),
                    entity_type: EntityType::Anime as i32,
                    address: "1".to_owned(),
                }),
                fetched_at: None,
                is_not_found: false,
                is_dead: false,
                responses: vec![PageResponse::default()],
            }),
        }))
    }

    async fn merge_anime(
        &self,
        _request: Request<MergeAnimeRequest>,
    ) -> Result<Response<MergeAnimeResponse>, Status> {
        if self.state.lock().expect("lock").fail_enrichment_not_found {
            return Err(Status::invalid_argument("bad merge"));
        }
        Ok(Response::new(MergeAnimeResponse {
            anime: None,
            episodes: vec![],
            relation_claims: vec![],
            provider_synopses_hash: "syn-hash".to_owned(),
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
        if self.state.lock().expect("lock").fail_enrichment_not_found {
            return Err(Status::invalid_argument("bad episode merge"));
        }
        Ok(Response::new(MergeEpisodeResponse { episode: None }))
    }

    async fn merge_character(
        &self,
        _request: Request<MergeCharacterRequest>,
    ) -> Result<Response<MergeCharacterResponse>, Status> {
        if self.state.lock().expect("lock").fail_enrichment_not_found {
            return Err(Status::invalid_argument("bad character merge"));
        }
        Ok(Response::new(MergeCharacterResponse { character: None }))
    }
}

#[tokio::test]
async fn grpc_catalogue_adapter_covers_happy_failed_and_aborted_paths() -> anyhow::Result<()> {
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

    let listed = adapter
        .list_entity_pages(RefreshEntityKind::Anime, "a1".to_owned())
        .await?;
    assert_eq!(listed.pages.len(), 4);
    assert_eq!(listed.entity_merge_version, None);

    let page_key = RefreshPageKey {
        provider: "mal".to_owned(),
        entity_type: "anime".to_owned(),
        address: "1".to_owned(),
    };
    let refreshed = adapter.refresh_page(page_key.clone(), 3600).await?;
    assert!(refreshed.has_changed);
    assert!(!refreshed.has_failed);

    mock.state.lock().expect("lock").fail_fetch_page = true;
    let failed_fetch = adapter.refresh_page(page_key, 3600).await?;
    assert!(!failed_fetch.has_changed);
    assert!(failed_fetch.has_failed);

    let saved = adapter
        .merge_entity_attempt(RefreshEntityKind::Anime, "a1".to_owned(), 2)
        .await?;
    match saved {
        MergeAttemptOutcome::Saved(out) => {
            assert!(out.has_changed);
            assert!(out.has_changed_synopses);
            assert_eq!(out.changed_episode_ids, vec!["ep-10"]);
        }
        MergeAttemptOutcome::VersionConflict(msg) => panic!("unexpected conflict: {msg}"),
    }

    let ep_saved = adapter
        .merge_entity_attempt(RefreshEntityKind::Episode, "ep-1".to_owned(), 2)
        .await?;
    assert!(matches!(ep_saved, MergeAttemptOutcome::Saved(_)));
    let ch_saved = adapter
        .merge_entity_attempt(RefreshEntityKind::Character, "ch-1".to_owned(), 2)
        .await?;
    assert!(matches!(ch_saved, MergeAttemptOutcome::Saved(_)));

    mock.state.lock().expect("lock").aborted_save = true;
    let conflicted = adapter
        .merge_entity_attempt(RefreshEntityKind::Anime, "a1".to_owned(), 2)
        .await?;
    assert!(matches!(
        conflicted,
        MergeAttemptOutcome::VersionConflict(_)
    ));
    let ep_conflicted = adapter
        .merge_entity_attempt(RefreshEntityKind::Episode, "ep-1".to_owned(), 2)
        .await?;
    assert!(matches!(
        ep_conflicted,
        MergeAttemptOutcome::VersionConflict(_)
    ));
    let ch_conflicted = adapter
        .merge_entity_attempt(RefreshEntityKind::Character, "ch-1".to_owned(), 2)
        .await?;
    assert!(matches!(
        ch_conflicted,
        MergeAttemptOutcome::VersionConflict(_)
    ));

    adapter
        .record_refresh(RecordRefreshInput {
            entity_kind: RefreshEntityKind::Anime,
            entity_id: "a1".to_owned(),
            outcome: RefreshOutcomeKind::Succeeded,
            refresh_after_days: 7,
        })
        .await?;
    adapter
        .record_refresh(RecordRefreshInput {
            entity_kind: RefreshEntityKind::Episode,
            entity_id: "ep-1".to_owned(),
            outcome: RefreshOutcomeKind::PagesFailed,
            refresh_after_days: 7,
        })
        .await?;
    adapter
        .record_refresh(RecordRefreshInput {
            entity_kind: RefreshEntityKind::Character,
            entity_id: "ch-1".to_owned(),
            outcome: RefreshOutcomeKind::Failed,
            refresh_after_days: 7,
        })
        .await?;
    assert_eq!(mock.state.lock().expect("lock").recorded.len(), 3);

    // Cover non-anime EntityType variants and map_grpc_status paths
    assert_eq!(
        map_grpc_status(&Status::not_found("x")).kind,
        temporal_workflows::PortErrorKind::NonRetryable
    );
    assert_eq!(
        map_grpc_status(&Status::unavailable("x")).kind,
        temporal_workflows::PortErrorKind::Retryable
    );
    let listed_all = adapter
        .list_entity_pages(RefreshEntityKind::Episode, "ep-1".to_owned())
        .await?;
    assert_eq!(listed_all.pages.len(), 4);
    for entity_type in ["character", "episode", "other"] {
        let out = adapter
            .refresh_page(
                RefreshPageKey {
                    provider: "mal".to_owned(),
                    entity_type: entity_type.to_owned(),
                    address: "1".to_owned(),
                },
                0,
            )
            .await?;
        assert!(out.has_failed || out.has_changed);
    }

    mock.state.lock().expect("lock").fail_fetch_page = false;
    mock.state.lock().expect("lock").empty_fetch_page = true;
    assert!(
        adapter
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
    mock.state.lock().expect("lock").empty_fetch_page = false;

    mock.state.lock().expect("lock").fail_enrichment_not_found = true;
    assert!(
        adapter
            .merge_entity_attempt(RefreshEntityKind::Anime, "a1".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        adapter
            .merge_entity_attempt(RefreshEntityKind::Episode, "ep-1".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        adapter
            .merge_entity_attempt(RefreshEntityKind::Character, "ch-1".to_owned(), 1)
            .await
            .is_err()
    );
    mock.state.lock().expect("lock").fail_enrichment_not_found = false;

    mock.state.lock().expect("lock").fail_postgres_not_found = true;
    assert!(
        adapter
            .list_entity_pages(RefreshEntityKind::Anime, "a1".to_owned())
            .await
            .is_err()
    );
    assert!(
        adapter
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
        adapter
            .merge_entity_attempt(RefreshEntityKind::Anime, "a1".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        adapter
            .merge_entity_attempt(RefreshEntityKind::Episode, "ep-1".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        adapter
            .merge_entity_attempt(RefreshEntityKind::Character, "ch-1".to_owned(), 1)
            .await
            .is_err()
    );
    assert!(
        adapter
            .record_refresh(RecordRefreshInput {
                entity_kind: RefreshEntityKind::Anime,
                entity_id: "a1".to_owned(),
                outcome: RefreshOutcomeKind::Succeeded,
                refresh_after_days: 7,
            })
            .await
            .is_err()
    );
    mock.state.lock().expect("lock").fail_postgres_not_found = false;
    mock.state.lock().expect("lock").aborted_save = false;

    Ok(())
}

#[tokio::test]
async fn grpc_catalogue_adapter_routes_split_fetch_and_merge_endpoints() -> anyhow::Result<()> {
    let pg_listener = TcpListener::bind("127.0.0.1:0").await?;
    let fetch_listener = TcpListener::bind("127.0.0.1:0").await?;
    let merge_listener = TcpListener::bind("127.0.0.1:0").await?;
    let pg_addr = pg_listener.local_addr()?;
    let fetch_addr = fetch_listener.local_addr()?;
    let merge_addr = merge_listener.local_addr()?;

    let pg_mock = MockServices::default();
    let fetch_mock = MockServices::default();
    let merge_mock = MockServices::default();

    let spawn_incoming = |listener: TcpListener| {
        futures::stream::unfold(listener, |listener| async move {
            let accepted = listener.accept().await.map(|(stream, _addr)| stream);
            Some((accepted, listener))
        })
    };

    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(
                PostgresServiceServer::new(pg_mock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(spawn_incoming(pg_listener))
            .await;
    });
    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(
                EnrichmentServiceServer::new(fetch_mock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(spawn_incoming(fetch_listener))
            .await;
    });
    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(
                EnrichmentServiceServer::new(merge_mock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(spawn_incoming(merge_listener))
            .await;
    });

    let pg_ch = Channel::from_shared(format!("http://{pg_addr}"))?
        .connect()
        .await?;
    let fetch_ch = Channel::from_shared(format!("http://{fetch_addr}"))?
        .connect()
        .await?;
    let merge_ch = Channel::from_shared(format!("http://{merge_addr}"))?
        .connect()
        .await?;

    let adapter = GrpcCatalogueAdapter::with_split_enrichment(
        build_postgres_client(pg_ch),
        build_enrichment_client(fetch_ch),
        build_enrichment_client(merge_ch),
    );
    let page_out = adapter
        .refresh_page(
            RefreshPageKey {
                provider: "mal".to_owned(),
                entity_type: "anime".to_owned(),
                address: "1".to_owned(),
            },
            0,
        )
        .await?;
    assert!(page_out.has_changed);

    let merge_out = adapter
        .merge_entity_attempt(RefreshEntityKind::Anime, "a1".to_owned(), 1)
        .await?;
    assert!(matches!(merge_out, MergeAttemptOutcome::Saved(_)));
    Ok(())
}

#[tokio::test]
async fn temporal_workflow_starter_starts_and_deduplicates_downstream_workflows()
-> anyhow::Result<()> {
    let temporal_addr =
        std::env::var("TEMPORAL_ADDRESS").unwrap_or_else(|_| "http://localhost:7233".to_owned());
    let mut connection =
        Connection::connect(ConnectionOptions::new(Url::parse(&temporal_addr)?).build()).await?;
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let test_namespace = format!("test-echo111-starter-{nanos}");
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
    tokio::time::sleep(Duration::from_millis(200)).await;
    connection
        .add_search_attributes(temporalio_client::tonic::Request::new(
            AddSearchAttributesRequest {
                namespace: test_namespace.clone(),
                search_attributes: std::collections::HashMap::from([(
                    REFRESH_TRIGGER_SEARCH_ATTRIBUTE.to_owned(),
                    IndexedValueType::Keyword as i32,
                )]),
            },
        ))
        .await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    let client = Client::new(connection, ClientOptions::new(&test_namespace).build())?;
    let starter = TemporalWorkflowStarter::new(client.clone());

    let input = StartEntityWorkflowsInput {
        anime_id: "anime-xyz".to_owned(),
        episode_ids: vec!["ep-1".to_owned(), "ep-2".to_owned()],
        character_ids: vec!["char-1".to_owned()],
        character_fetch_requests: vec![CharacterFetchTarget {
            character_id: "char-2".to_owned(),
            fetch: FetchRequestedSignal {
                providers: vec!["anilist".to_owned()],
                max_age_seconds: 0,
                priority: 2,
            },
        }],
        start_synopsis: true,
        trigger: "user".to_owned(),
        priority: 1,
    };
    let started_count = starter.start_entity_workflows(input.clone()).await?;
    assert_eq!(started_count, 5);

    // Calling again hits UseExisting conflict policy cleanly without erroring
    let deduped = starter.start_entity_workflows(input).await?;
    assert_eq!(deduped, 5);

    // Also cover default trigger ("") and priority (0)
    let defaulted = starter
        .start_entity_workflows(StartEntityWorkflowsInput {
            anime_id: "anime-def".to_owned(),
            trigger: String::new(),
            priority: 0,
            ..StartEntityWorkflowsInput::default()
        })
        .await?;
    assert_eq!(defaulted, 0);

    assert_eq!(
        build_refresh_episode_workflow_id("ep-1"),
        "refresh-episode-ep-1"
    );
    assert_eq!(
        build_refresh_character_workflow_id("char-1"),
        "refresh-character-char-1"
    );
    assert_eq!(
        build_write_synopsis_workflow_id("anime-xyz"),
        "synopsis-anime-anime-xyz"
    );

    Ok(())
}
