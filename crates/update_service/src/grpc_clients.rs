//! gRPC service clients (`64 MiB` + gzip) and Temporal workflow starter adapters.

use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

pub use echora_protos::{enrichment_service_proto, postgres_service_proto, shared_proto};
use enrichment_service_proto::v1::enrichment_service_client::EnrichmentServiceClient;
use enrichment_service_proto::v1::{FetchPageRequest, MergeAnimeRequest};
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt as _, TryStreamExt as _};
use opentelemetry::global;
use opentelemetry::metrics::Gauge;
use postgres_service_proto::v1::postgres_service_client::PostgresServiceClient;
use postgres_service_proto::v1::{
    ListEntityPagesRequest, LoadAnimeRequest, RecordRefreshRequest,
    RefreshOutcome as ProtoRefreshOutcome, SaveAnimeRequest, StorePageRequest,
};
use prost::Message as _;
use serde::Serialize;
use shared_proto::v1::{EntityType as ProtoEntityType, PageKey as ProtoPageKey, StoredPage};
use temporal_workflows::{
    CatalogueDataPort, EntityPageSummary, EntityType, ListEntityPagesOutput, MergeAnimeOutput,
    MergeAttemptOutcome, PortError, Provider, REFRESH_TRIGGER_SEARCH_ATTRIBUTE, RefreshOutcome,
    RefreshPageKey, RefreshPageOutput, RefreshSettings, RefreshTrigger, SECONDS_PER_DAY,
    StartEntityWorkflowsInput, UPDATE_SERVICE_TASK_QUEUE, WorkflowStarterPort, build_priority,
    build_refresh_episode_workflow_id, build_write_synopsis_workflow_id,
};
use temporalio_client::{Client, UntypedWorkflow, WorkflowIdConflictPolicy, WorkflowStartOptions};
use temporalio_common::Priority;
use temporalio_common::data_converters::{
    GenericPayloadConverter as _, PayloadConverter, RawValue, SerializationContext,
    SerializationContextData, WorkflowSerializationContext,
};
use temporalio_common::search_attributes::{SearchAttributeKey, SearchAttributes};
use tonic::Code;
use tonic::codec::CompressionEncoding;
use tonic::transport::Channel;
use uuid::Uuid;

/// Maximum gRPC message size (64 MiB) configured on Update Service clients.
///
/// Matches `postgres_service` and `enrichment_service` limits in
/// `docs/architecture/update_service.md` so large multi-megabyte provider pages pass cleanly.
pub const MAX_GRPC_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Trailing gRPC metadata key set by `EnrichmentService.FetchPage` on provider-specific errors.
///
/// Must stay in sync with `FETCH_ERROR_METADATA_KEY` in
/// `apps/enrichment_service/src/enrichment_service/routes/fetch_page.py`.
pub const FETCH_ERROR_METADATA_KEY: &str = "x-echora-fetch-error";

/// Metadata value indicating `polite_http` rejected the fetch because the provider is blocked.
///
/// Must stay in sync with `FETCH_ERROR_PROVIDER_BLOCKED` in
/// `apps/enrichment_service/src/enrichment_service/routes/fetch_page.py`.
pub const FETCH_ERROR_PROVIDER_BLOCKED: &str = "provider_blocked";

/// Maximum concurrent `start_workflow` RPCs issued inside one `start_entity_workflows` batch.
///
/// Bounds pressure on the Temporal frontend while letting 1,100+ episode series (such as
/// One Piece) complete well within the 60-second activity timeout.
const MAX_CONCURRENT_WORKFLOW_STARTS: usize = 16;

/// Builds a `PostgresServiceClient` configured with gzip compression and 64 MiB limits.
///
/// # Examples
///
/// ```no_run
/// # use tonic::transport::Channel;
/// # async fn example() -> Result<(), tonic::transport::Error> {
/// let channel = Channel::from_static("http://localhost:8004").connect_lazy();
/// let _client = update_service::build_postgres_client(channel);
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn build_postgres_client(channel: Channel) -> PostgresServiceClient<Channel> {
    PostgresServiceClient::new(channel)
        .send_compressed(CompressionEncoding::Gzip)
        .accept_compressed(CompressionEncoding::Gzip)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
}

/// Builds an `EnrichmentServiceClient` configured with gzip compression and 64 MiB limits.
///
/// # Examples
///
/// ```no_run
/// # use tonic::transport::Channel;
/// # async fn example() -> Result<(), tonic::transport::Error> {
/// let channel = Channel::from_static("http://localhost:8002").connect_lazy();
/// let _client = update_service::build_enrichment_client(channel);
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn build_enrichment_client(channel: Channel) -> EnrichmentServiceClient<Channel> {
    EnrichmentServiceClient::new(channel)
        .send_compressed(CompressionEncoding::Gzip)
        .accept_compressed(CompressionEncoding::Gzip)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
}

/// Maps a `tonic::Status` into a [`PortError`] preserving it as the error source.
///
/// Permanent codes (`INVALID_ARGUMENT`, `NOT_FOUND`, `PERMISSION_DENIED`,
/// `RESOURCE_EXHAUSTED`, `FAILED_PRECONDITION`, `UNIMPLEMENTED`) map to
/// [`PortError::non_retryable`]. `UNAVAILABLE` with trailing metadata
/// `x-echora-fetch-error: provider_blocked` maps to [`PortError::provider_blocked`].
/// All other statuses remain retryable.
///
/// # Examples
///
/// ```
/// use std::error::Error as _;
///
/// use tonic::Status;
/// use update_service::map_grpc_status;
///
/// let err = map_grpc_status(Status::not_found("missing"));
/// assert!(err.is_non_retryable());
/// assert!(err.source().is_some());
/// assert!(map_grpc_status(Status::unavailable("busy")).is_retryable());
/// ```
#[must_use]
pub fn map_grpc_status(status: tonic::Status) -> PortError {
    let message = status.to_string();
    if status.code() == Code::Unavailable
        && status
            .metadata()
            .get(FETCH_ERROR_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            == Some(FETCH_ERROR_PROVIDER_BLOCKED)
    {
        return PortError::provider_blocked(message).with_source(status);
    }

    match status.code() {
        Code::InvalidArgument
        | Code::NotFound
        | Code::PermissionDenied
        | Code::ResourceExhausted
        | Code::FailedPrecondition
        | Code::Unimplemented => PortError::non_retryable(message).with_source(status),
        Code::Ok
        | Code::Cancelled
        | Code::Unknown
        | Code::DeadlineExceeded
        | Code::AlreadyExists
        | Code::Aborted
        | Code::OutOfRange
        | Code::Internal
        | Code::Unavailable
        | Code::DataLoss
        | Code::Unauthenticated => PortError::retryable(message).with_source(status),
    }
}

/// gRPC-backed implementation of [`CatalogueDataPort`] talking to `postgres_service` and
/// `enrichment_service`.
#[derive(Clone)]
pub struct GrpcCatalogueAdapter {
    postgres: PostgresServiceClient<Channel>,
    enrichment: EnrichmentServiceClient<Channel>,
    max_message_bytes_seen: Arc<AtomicU64>,
    max_message_gauge: Gauge<u64>,
}

impl GrpcCatalogueAdapter {
    /// Creates an adapter from pre-configured `postgres` and `enrichment` gRPC clients.
    #[must_use]
    pub fn new(
        postgres: PostgresServiceClient<Channel>,
        enrichment: EnrichmentServiceClient<Channel>,
    ) -> Self {
        let meter = global::meter("update-service");
        let max_message_gauge = meter
            .u64_gauge("update_service.grpc.max_message_bytes")
            .with_description("Largest encoded gRPC message carrying page bodies, in bytes")
            .build();
        Self {
            postgres,
            enrichment,
            max_message_bytes_seen: Arc::new(AtomicU64::new(0)),
            max_message_gauge,
        }
    }

    /// Returns the largest encoded gRPC payload size observed by this adapter in bytes.
    #[must_use]
    pub fn max_message_bytes_seen(&self) -> u64 {
        self.max_message_bytes_seen.load(Ordering::Relaxed)
    }

    /// Records the largest encoded message size seen by this client as an OpenTelemetry gauge.
    fn record_message_size(&self, bytes: usize) {
        let size = u64::try_from(bytes).unwrap_or(u64::MAX);
        let previous = self
            .max_message_bytes_seen
            .fetch_max(size, Ordering::Relaxed);
        self.max_message_gauge.record(previous.max(size), &[]);
    }
}

impl fmt::Debug for GrpcCatalogueAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrpcCatalogueAdapter")
            .field("max_message_bytes_seen", &self.max_message_bytes_seen())
            .finish_non_exhaustive()
    }
}

/// Converts a [`RefreshPageKey`] into a proto `PageKey`.
fn to_proto_page_key(key: &RefreshPageKey) -> ProtoPageKey {
    let entity_type = match key.entity_type {
        EntityType::Anime => ProtoEntityType::Anime as i32,
        EntityType::Character => ProtoEntityType::Character as i32,
        EntityType::Episode => ProtoEntityType::Episode as i32,
    };
    ProtoPageKey {
        provider: key.provider.slug().to_owned(),
        entity_type,
        address: key.address.clone(),
    }
}

/// Converts a proto `PageKey` into a strongly typed [`RefreshPageKey`].
fn from_proto_page_key(key: ProtoPageKey) -> Result<RefreshPageKey, PortError> {
    let provider = Provider::from_slug(&key.provider).ok_or_else(|| {
        PortError::non_retryable(format!("unknown provider in PageKey: {:?}", key.provider))
    })?;
    let entity_type = match ProtoEntityType::try_from(key.entity_type) {
        Ok(ProtoEntityType::Anime) => EntityType::Anime,
        Ok(ProtoEntityType::Character) => EntityType::Character,
        Ok(ProtoEntityType::Episode) => EntityType::Episode,
        Ok(ProtoEntityType::Unspecified) | Err(_) => {
            return Err(PortError::non_retryable(format!(
                "invalid entity_type in PageKey: {}",
                key.entity_type
            )));
        }
    };
    Ok(RefreshPageKey {
        provider,
        entity_type,
        address: key.address,
    })
}

impl CatalogueDataPort for GrpcCatalogueAdapter {
    fn list_entity_pages(
        &self,
        anime_id: Uuid,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>> {
        let mut postgres = self.postgres.clone();
        Box::pin(async move {
            let response = postgres
                .list_entity_pages(ListEntityPagesRequest {
                    entity_type: ProtoEntityType::Anime as i32,
                    entity_id: anime_id.to_string(),
                })
                .await
                .map_err(map_grpc_status)?
                .into_inner();

            let mut pages = Vec::with_capacity(response.pages.len());
            for page in response.pages {
                let raw_key = page.page_key.ok_or_else(|| {
                    PortError::non_retryable("ListEntityPages response item missing page_key")
                })?;
                pages.push(EntityPageSummary {
                    page_key: from_proto_page_key(raw_key)?,
                    is_partial_source: page.is_partial_source,
                    fetched_at_epoch_seconds: page.fetched_at.map(|ts| ts.seconds),
                    is_dead: page.is_dead,
                    fetch_failed_at_epoch_seconds: page.fetch_failed_at.map(|ts| ts.seconds),
                    fetch_failures: page.fetch_failures,
                });
            }

            Ok(ListEntityPagesOutput {
                pages,
                entity_merge_version: response.merge_version,
                refresh_settings: RefreshSettings::default(),
            })
        })
    }

    fn fetch_and_store_page(
        &self,
        page_key: RefreshPageKey,
        max_age: Duration,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>> {
        let mut enrichment = self.enrichment.clone();
        let mut postgres = self.postgres.clone();
        let adapter = self.clone();
        Box::pin(async move {
            let proto_key = to_proto_page_key(&page_key);
            let fetched_page = enrichment
                .fetch_page(FetchPageRequest {
                    page_key: Some(proto_key),
                    max_age: Some(prost_types::Duration::try_from(max_age).unwrap_or_default()),
                })
                .await
                .map_err(map_grpc_status)?
                .into_inner()
                .page
                .ok_or_else(|| PortError::non_retryable("FetchPage response missing page"))?;

            adapter.record_message_size(fetched_page.encoded_len());
            let has_failed = fetched_page.responses.is_empty() && !fetched_page.is_not_found;
            let store_response = postgres
                .store_page(StorePageRequest {
                    page: Some(fetched_page),
                })
                .await
                .map_err(map_grpc_status)?
                .into_inner();

            Ok(RefreshPageOutput {
                page_key,
                has_changed: store_response.has_changed,
                has_failed,
            })
        })
    }

    fn record_page_failure(
        &self,
        page_key: RefreshPageKey,
        failed_at: SystemTime,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        let mut postgres = self.postgres.clone();
        Box::pin(async move {
            postgres
                .store_page(StorePageRequest {
                    page: Some(StoredPage {
                        page_key: Some(to_proto_page_key(&page_key)),
                        fetched_at: Some(prost_types::Timestamp::from(failed_at)),
                        is_not_found: false,
                        is_dead: false,
                        responses: Vec::new(),
                    }),
                })
                .await
                .map_err(map_grpc_status)?;
            Ok(())
        })
    }

    fn merge_anime_attempt(
        &self,
        anime_id: Uuid,
        merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>> {
        let mut postgres = self.postgres.clone();
        let mut enrichment = self.enrichment.clone();
        let adapter = self.clone();
        let anime_id_text = anime_id.to_string();
        Box::pin(async move {
            let loaded = postgres
                .load_anime(LoadAnimeRequest {
                    anime_id: anime_id_text.clone(),
                })
                .await
                .map_err(map_grpc_status)?
                .into_inner();
            adapter.record_message_size(loaded.encoded_len());

            let merged = enrichment
                .merge_anime(MergeAnimeRequest {
                    anime_id: anime_id_text.clone(),
                    source_pages: loaded.source_pages,
                    partial_source_pages: loaded.partial_source_pages,
                    corrections: loaded.corrections,
                    generated_texts: Vec::new(),
                    known_episodes: loaded.known_episodes,
                })
                .await
                .map_err(map_grpc_status)?
                .into_inner();
            adapter.record_message_size(merged.encoded_len());

            let save_request = SaveAnimeRequest {
                anime_id: anime_id_text,
                expected_version: loaded.version,
                anime: merged.anime,
                merge_version,
                relation_claims: merged.relation_claims,
                episodes: merged.episodes,
                page_sources: merged.page_sources,
            };
            adapter.record_message_size(save_request.encoded_len());

            match postgres.save_anime(save_request).await {
                Ok(response) => {
                    let saved = response.into_inner();
                    let mut changed_episode_ids = saved.created_episode_ids;
                    changed_episode_ids.extend(saved.updated_episode_ids);
                    Ok(MergeAttemptOutcome::Saved(MergeAnimeOutput {
                        has_changed: saved.has_changed,
                        changed_episode_ids,
                        // TODO(ECHO-115): compare `merged.provider_synopses_hash` against the
                        // stored synopsis input hash once `LoadAnimeResponse.generated_texts` lands.
                        has_changed_synopses: false,
                    }))
                }
                Err(status) if status.code() == Code::Aborted => {
                    Ok(MergeAttemptOutcome::VersionConflict(status.to_string()))
                }
                Err(status) => Err(map_grpc_status(status)),
            }
        })
    }

    fn record_refresh(
        &self,
        anime_id: Uuid,
        outcome: RefreshOutcome,
        refresh_after_days: i64,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        let mut postgres = self.postgres.clone();
        Box::pin(async move {
            let outcome = match outcome {
                RefreshOutcome::Succeeded => ProtoRefreshOutcome::Succeeded as i32,
                RefreshOutcome::PagesFailed => ProtoRefreshOutcome::PagesFailed as i32,
                RefreshOutcome::Failed => ProtoRefreshOutcome::Failed as i32,
            };
            let refresh_seconds = u64::try_from(refresh_after_days.max(1))
                .unwrap_or(1)
                .saturating_mul(u64::try_from(SECONDS_PER_DAY).unwrap_or(86_400));
            let refresh_after =
                prost_types::Duration::try_from(Duration::from_secs(refresh_seconds))
                    .unwrap_or_default();
            postgres
                .record_refresh(RecordRefreshRequest {
                    entity_type: ProtoEntityType::Anime as i32,
                    entity_id: anime_id.to_string(),
                    outcome,
                    refresh_after: Some(refresh_after),
                })
                .await
                .map_err(map_grpc_status)?;
            Ok(())
        })
    }
}

/// Top-level Temporal workflow starter using `USE_EXISTING` and inherited `RefreshTrigger`.
#[derive(Clone)]
pub struct TemporalWorkflowStarter {
    client: Client,
}

impl TemporalWorkflowStarter {
    /// Creates a starter backed by a connected Temporal `client`.
    #[must_use]
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

impl fmt::Debug for TemporalWorkflowStarter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemporalWorkflowStarter")
            .field("namespace", &self.client.options().namespace)
            .finish_non_exhaustive()
    }
}

#[derive(Serialize)]
struct DownstreamWorkflowInput {
    entity_id: String,
    trigger: RefreshTrigger,
    priority: u32,
}

/// Encodes a serializable workflow input using Temporal's standard [`PayloadConverter`].
fn encode_workflow_input<T: Serialize + 'static>(value: &T) -> Result<RawValue, PortError> {
    let converter = PayloadConverter::default();
    let context_data = SerializationContextData::Workflow(WorkflowSerializationContext::new());
    let context = SerializationContext::new(&context_data, &converter);
    let payload = converter.to_payload(&context, value).map_err(|err| {
        PortError::non_retryable(format!("failed to encode workflow input: {err}"))
    })?;
    Ok(RawValue {
        payloads: vec![payload],
    })
}

/// Starts one top-level Temporal workflow with `USE_EXISTING`, priority, and `RefreshTrigger`.
async fn start_top_level_workflow(
    client: &Client,
    workflow_type: &str,
    workflow_id: String,
    priority: Priority,
    search_attributes: SearchAttributes,
    input: &DownstreamWorkflowInput,
) -> Result<(), PortError> {
    let raw_input = encode_workflow_input(input)?;
    let options = WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, workflow_id)
        .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
        .priority(priority)
        .search_attributes(search_attributes)
        .build();
    client
        .start_workflow(UntypedWorkflow::new(workflow_type), raw_input, options)
        .await
        .map_err(|err| PortError::retryable(err.to_string()))?;
    Ok(())
}

impl WorkflowStarterPort for TemporalWorkflowStarter {
    fn start_entity_workflows(
        &self,
        input: StartEntityWorkflowsInput,
    ) -> BoxFuture<'_, Result<usize, PortError>> {
        let client = self.client.clone();
        Box::pin(async move {
            let priority = build_priority(input.priority, input.trigger);
            let trigger_key = SearchAttributeKey::keyword(REFRESH_TRIGGER_SEARCH_ATTRIBUTE);
            let search_attributes =
                SearchAttributes::new([trigger_key.value_set(input.trigger.as_str().to_owned())]);

            let episode_count = input.episode_ids.len();
            let trigger = input.trigger;
            let raw_priority = input.priority.map_or(0, NonZeroU32::get);

            stream::iter(input.episode_ids)
                .map(|episode_id| {
                    let client = client.clone();
                    let priority = priority.clone();
                    let search_attributes = search_attributes.clone();
                    async move {
                        let workflow_id = build_refresh_episode_workflow_id(&episode_id);
                        let downstream_input = DownstreamWorkflowInput {
                            entity_id: episode_id,
                            trigger,
                            priority: raw_priority,
                        };
                        start_top_level_workflow(
                            &client,
                            "RefreshEpisodeWorkflow",
                            workflow_id,
                            priority,
                            search_attributes,
                            &downstream_input,
                        )
                        .await
                    }
                })
                .buffer_unordered(MAX_CONCURRENT_WORKFLOW_STARTS)
                .try_collect::<Vec<()>>()
                .await?;

            let mut started = episode_count;
            if input.start_synopsis {
                let workflow_id = build_write_synopsis_workflow_id(input.anime_id);
                let downstream_input = DownstreamWorkflowInput {
                    entity_id: input.anime_id.to_string(),
                    trigger,
                    priority: raw_priority,
                };
                start_top_level_workflow(
                    &client,
                    "WriteSynopsisWorkflow",
                    workflow_id,
                    priority,
                    search_attributes,
                    &downstream_input,
                )
                .await?;
                started += 1;
            }

            Ok(started)
        })
    }
}
