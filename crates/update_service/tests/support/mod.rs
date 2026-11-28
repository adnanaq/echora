//! Shared test doubles and helpers for `update_service` integration tests.

use std::net::TcpListener;
use std::time::Duration;

use anyhow::{Context as _, ensure};
use temporal_workflows::{
    RefreshAnimeInput, RefreshAnimeOutcome, RefreshAnimeWorkflow, RefreshOutcome, RefreshTrigger,
    UPDATE_SERVICE_TASK_QUEUE, build_refresh_anime_workflow_id,
};
use temporalio_client::{Client, WorkflowGetResultOptions, WorkflowStartOptions};
use tonic::{Response, Status};
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
    GetAnimeResponse, GetCharacterResponse, GetEntityRequest, GetEpisodeResponse,
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

#[derive(Clone, Default)]
struct MinimalCatalogueMock;

#[tonic::async_trait]
impl PostgresService for MinimalCatalogueMock {
    async fn list_entity_pages(
        &self,
        _request: tonic::Request<ListEntityPagesRequest>,
    ) -> Result<Response<ListEntityPagesResponse>, Status> {
        Ok(Response::new(ListEntityPagesResponse {
            pages: Vec::new(),
            merge_version: Some(1),
        }))
    }

    async fn record_refresh(
        &self,
        _request: tonic::Request<RecordRefreshRequest>,
    ) -> Result<Response<RecordRefreshResponse>, Status> {
        Ok(Response::new(RecordRefreshResponse::default()))
    }

    async fn store_page(
        &self,
        _request: tonic::Request<StorePageRequest>,
    ) -> Result<Response<StorePageResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn load_anime(
        &self,
        _request: tonic::Request<LoadAnimeRequest>,
    ) -> Result<Response<LoadAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn save_anime(
        &self,
        _request: tonic::Request<SaveAnimeRequest>,
    ) -> Result<Response<SaveAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn load_episode(
        &self,
        _request: tonic::Request<LoadEpisodeRequest>,
    ) -> Result<Response<LoadEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn save_episode(
        &self,
        _request: tonic::Request<SaveEpisodeRequest>,
    ) -> Result<Response<SaveEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn mark_episode_aired(
        &self,
        _request: tonic::Request<MarkEpisodeAiredRequest>,
    ) -> Result<Response<MarkEpisodeAiredResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn load_character(
        &self,
        _request: tonic::Request<LoadCharacterRequest>,
    ) -> Result<Response<LoadCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn save_character(
        &self,
        _request: tonic::Request<SaveCharacterRequest>,
    ) -> Result<Response<SaveCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_anime(
        &self,
        _request: tonic::Request<ListDueRequest>,
    ) -> Result<Response<ListDueResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_episodes(
        &self,
        _request: tonic::Request<ListDueRequest>,
    ) -> Result<Response<ListDueResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_characters(
        &self,
        _request: tonic::Request<ListDueRequest>,
    ) -> Result<Response<ListDueResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_upcoming_episodes(
        &self,
        _request: tonic::Request<ListUpcomingEpisodesRequest>,
    ) -> Result<Response<ListUpcomingEpisodesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_score_anime(
        &self,
        _request: tonic::Request<ListScoreAnimeRequest>,
    ) -> Result<Response<ListScoreAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_requested_merges(
        &self,
        _request: tonic::Request<ListRequestedMergesRequest>,
    ) -> Result<Response<ListRequestedMergesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn get_anime(
        &self,
        _request: tonic::Request<GetEntityRequest>,
    ) -> Result<Response<GetAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn get_episode(
        &self,
        _request: tonic::Request<GetEntityRequest>,
    ) -> Result<Response<GetEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_episodes(
        &self,
        _request: tonic::Request<GetEntityRequest>,
    ) -> Result<Response<ListEpisodesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn get_character(
        &self,
        _request: tonic::Request<GetEntityRequest>,
    ) -> Result<Response<GetCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_entity_versions(
        &self,
        _request: tonic::Request<ListEntityVersionsRequest>,
    ) -> Result<Response<ListEntityVersionsResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn list_due_pages(
        &self,
        _request: tonic::Request<ListDuePagesRequest>,
    ) -> Result<Response<ListDuePagesResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
}

#[tonic::async_trait]
impl EnrichmentService for MinimalCatalogueMock {
    async fn fetch_page(
        &self,
        _request: tonic::Request<FetchPageRequest>,
    ) -> Result<Response<FetchPageResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn merge_anime(
        &self,
        _request: tonic::Request<MergeAnimeRequest>,
    ) -> Result<Response<MergeAnimeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn health(
        &self,
        _request: tonic::Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn run_pipeline(
        &self,
        _request: tonic::Request<RunPipelineRequest>,
    ) -> Result<Response<RunPipelineResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn merge_episode(
        &self,
        _request: tonic::Request<MergeEpisodeRequest>,
    ) -> Result<Response<MergeEpisodeResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }

    async fn merge_character(
        &self,
        _request: tonic::Request<MergeCharacterRequest>,
    ) -> Result<Response<MergeCharacterResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
}

/// Finds an unused ephemeral TCP port on loopback.
pub fn find_free_port() -> anyhow::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// Starts an in-process gzip gRPC server answering `ListEntityPages` and `RecordRefresh`.
pub async fn start_mock_grpc_server() -> anyhow::Result<String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let incoming = futures::stream::unfold(listener, |listener| async move {
        let accepted = listener.accept().await.map(|(stream, _addr)| stream);
        Some((accepted, listener))
    });
    tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(
                PostgresServiceServer::new(MinimalCatalogueMock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .add_service(
                EnrichmentServiceServer::new(MinimalCatalogueMock)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(incoming)
            .await;
    });
    Ok(format!("http://{addr}"))
}

/// Starts a no-op `RefreshAnimeWorkflow` against the minimal mock and asserts `Succeeded`.
pub async fn complete_refresh_workflow(
    client: &Client,
    anime_id: &str,
    timeout: Duration,
) -> anyhow::Result<()> {
    let workflow_id = build_refresh_anime_workflow_id(anime_id);
    let outcome: RefreshAnimeOutcome = tokio::time::timeout(timeout, async {
        let handle = client
            .start_workflow(
                RefreshAnimeWorkflow::run,
                RefreshAnimeInput {
                    anime_id: anime_id.to_owned(),
                    trigger: RefreshTrigger::User,
                    priority: 1,
                    refresh_after_days: 7,
                    merge_version: 1,
                    merge_first_wait_seconds: Some(0),
                    merge_coalesce_wait_seconds: Some(0),
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
