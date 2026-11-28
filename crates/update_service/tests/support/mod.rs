//! Shared test doubles and helpers for `update_service` integration tests.

#![expect(
    clippy::expect_used,
    clippy::struct_excessive_bools,
    reason = "integration test mock flags and state locks"
)]
#![expect(
    dead_code,
    reason = "each integration test binary uses a subset of the shared mock helpers"
)]

use std::net::TcpListener;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, ensure};
use temporal_workflows::{
    RefreshAnimeInput, RefreshAnimeOutcome, RefreshAnimeWorkflow, RefreshOutcome, RefreshTrigger,
    UPDATE_SERVICE_TASK_QUEUE, build_refresh_anime_workflow_id,
};
use temporalio_client::{Client, WorkflowGetResultOptions, WorkflowStartOptions};
use tonic::metadata::MetadataValue;
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
use update_service::shared_proto::v1::{
    EntityType as ProtoEntityType, PageKey, PageResponse, PageSource, StoredPage,
};
use update_service::{FETCH_ERROR_METADATA_KEY, FETCH_ERROR_PROVIDER_BLOCKED};
use uuid::Uuid;

pub struct MockCatalogueState {
    pub pages: Vec<EntityPage>,
    pub merge_version: Option<i32>,
    pub aborted_save: bool,
    pub fail_save_anime_invalid: bool,
    pub fail_fetch_page_transient: bool,
    pub fail_fetch_page_blocked: bool,
    pub empty_fetch_page: bool,
    pub fail_postgres_not_found: bool,
    pub fail_enrichment_invalid: bool,
    pub invalid_page_provider: bool,
    pub invalid_page_entity_type: bool,
    pub missing_page_key: bool,
    pub saved_page_sources: Vec<PageSource>,
    pub saved_merge_versions: Vec<i32>,
    pub stored_pages: Vec<StoredPage>,
    pub recorded: Vec<(String, i32)>,
}

impl Default for MockCatalogueState {
    fn default() -> Self {
        Self {
            pages: Vec::new(),
            merge_version: Some(1),
            aborted_save: false,
            fail_save_anime_invalid: false,
            fail_fetch_page_transient: false,
            fail_fetch_page_blocked: false,
            empty_fetch_page: false,
            fail_postgres_not_found: false,
            fail_enrichment_invalid: false,
            invalid_page_provider: false,
            invalid_page_entity_type: false,
            missing_page_key: false,
            saved_page_sources: Vec::new(),
            saved_merge_versions: Vec::new(),
            stored_pages: Vec::new(),
            recorded: Vec::new(),
        }
    }
}

#[derive(Clone, Default)]
struct ConfigurableCatalogueMock {
    state: Arc<Mutex<MockCatalogueState>>,
}

#[tonic::async_trait]
impl PostgresService for ConfigurableCatalogueMock {
    async fn list_entity_pages(
        &self,
        _request: Request<ListEntityPagesRequest>,
    ) -> Result<Response<ListEntityPagesResponse>, Status> {
        let (
            fail_not_found,
            invalid_provider,
            invalid_entity_type,
            missing_key,
            mut pages,
            merge_version,
        ) = {
            let guard = self.state.lock().expect("lock");
            (
                guard.fail_postgres_not_found,
                guard.invalid_page_provider,
                guard.invalid_page_entity_type,
                guard.missing_page_key,
                guard.pages.clone(),
                guard.merge_version,
            )
        };
        if fail_not_found {
            return Err(Status::not_found("missing anime"));
        }
        if missing_key && let Some(first) = pages.first_mut() {
            first.page_key = None;
        }
        if invalid_provider
            && let Some(first) = pages.first_mut()
            && let Some(key) = first.page_key.as_mut()
        {
            "unknown".clone_into(&mut key.provider);
        }
        if invalid_entity_type
            && let Some(first) = pages.first_mut()
            && let Some(key) = first.page_key.as_mut()
        {
            key.entity_type = ProtoEntityType::Unspecified as i32;
        }
        Ok(Response::new(ListEntityPagesResponse {
            pages,
            merge_version,
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
        if guard.fail_save_anime_invalid {
            return Err(Status::invalid_argument("invalid merge_version"));
        }
        let inner = request.into_inner();
        guard.saved_merge_versions.push(inner.merge_version);
        guard.saved_page_sources = inner.page_sources;
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
impl EnrichmentService for ConfigurableCatalogueMock {
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

/// Finds an unused ephemeral TCP port on loopback.
pub fn find_free_port() -> anyhow::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// Starts a configurable in-process gzip gRPC server backed by `state`.
pub async fn start_configurable_mock_grpc_server(
    state: Arc<Mutex<MockCatalogueState>>,
) -> anyhow::Result<String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let mock = ConfigurableCatalogueMock { state };
    let incoming = futures::stream::unfold(listener, |listener| async move {
        let accepted = listener.accept().await.map(|(stream, _addr)| stream);
        Some((accepted, listener))
    });
    tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(
                PostgresServiceServer::new(mock.clone())
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .add_service(
                EnrichmentServiceServer::new(mock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(incoming)
            .await;
    });
    Ok(format!("http://{addr}"))
}

/// Starts an in-process gzip gRPC server answering `ListEntityPages` and `RecordRefresh`.
pub async fn start_mock_grpc_server() -> anyhow::Result<String> {
    start_configurable_mock_grpc_server(Arc::new(Mutex::new(MockCatalogueState::default()))).await
}

/// Starts a no-op `RefreshAnimeWorkflow` against the minimal mock and asserts `Succeeded`.
pub async fn complete_refresh_workflow(
    client: &Client,
    anime_id: Uuid,
    timeout: Duration,
) -> anyhow::Result<()> {
    let workflow_id = build_refresh_anime_workflow_id(anime_id);
    let outcome: RefreshAnimeOutcome = tokio::time::timeout(timeout, async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id,
                    trigger: RefreshTrigger::User,
                    priority: NonZeroU32::new(1),
                    pages_relinked: false,
                },
                WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id).build(),
            )
            .await?;
        handle
            .get_result(WorkflowGetResultOptions::default())
            .await
            .map_err(anyhow::Error::from)
    })
    .await
    .context("workers never completed RefreshAnimeWorkflow")??;
    ensure!(outcome.anime_id == anime_id);
    ensure!(outcome.final_outcome == RefreshOutcome::Succeeded);
    Ok(())
}
