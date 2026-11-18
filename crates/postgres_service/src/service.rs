//! gRPC service handlers and error mapping for `postgres_service`.

// Rust guideline compliant 2026-10-09

use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use opentelemetry::global;
use opentelemetry::metrics::Gauge;
use postgres_db::{
    PageKeyInput, PageKind, PageResponseInput, PgPool, StorePageInput, load_anime, save_anime,
    store_page,
};
use prost::Message as _;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::anime_mapping::build_save_anime_input;

#[allow(clippy::all, clippy::pedantic, missing_docs, dead_code)]
pub mod shared_proto {
    pub mod v1 {
        tonic::include_proto!("shared_proto.v1");
    }
}

#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod postgres_service_proto {
    pub mod v1 {
        tonic::include_proto!("postgres_service.v1");
    }
}

use postgres_service_proto::v1::postgres_service_server::PostgresService;
use postgres_service_proto::v1::{
    LoadAnimeRequest, LoadAnimeResponse, SaveAnimeRequest, SaveAnimeResponse, StorePageRequest,
    StorePageResponse,
};
use shared_proto::v1::{
    FieldCorrection as ProtoFieldCorrection, PageKey as ProtoPageKey, PageKind as ProtoPageKind,
    PageResponse as ProtoPageResponse, StoredPage as ProtoStoredPage,
};

/// Maximum gRPC message size (64 MiB) for RPCs carrying raw provider page bodies.
pub const MAX_PAGE_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Maps a `postgres_db::Error` into a `tonic::Status` in one place.
#[must_use]
pub fn map_db_error(err: &postgres_db::Error) -> Status {
    if err.is_version_conflict() {
        Status::aborted(err.to_string())
    } else if err.is_not_found() {
        Status::not_found(err.to_string())
    } else if err.is_invalid_argument() {
        Status::invalid_argument(err.to_string())
    } else {
        Status::internal(err.to_string())
    }
}

/// Implements `PostgresService` gRPC RPCs backed by `postgres_db`.
#[derive(Debug)]
pub struct PostgresGrpcService {
    pool: PgPool,
    max_message_bytes_seen: AtomicU64,
    max_message_gauge: Gauge<u64>,
}

impl PostgresGrpcService {
    /// Creates a new `PostgresGrpcService` backed by `pool`.
    #[must_use]
    pub fn new(pool: PgPool) -> Self {
        let meter = global::meter("postgres-service");
        let max_message_gauge = meter
            .u64_gauge("postgres_service.grpc.max_message_bytes")
            .with_description("Largest encoded StorePage gRPC message size seen")
            .build();
        Self {
            pool,
            max_message_bytes_seen: AtomicU64::new(0),
            max_message_gauge,
        }
    }

    fn record_message_size(&self, bytes: u64) {
        let prev = self
            .max_message_bytes_seen
            .fetch_max(bytes, Ordering::Relaxed);
        let max_seen = prev.max(bytes);
        self.max_message_gauge.record(max_seen, &[]);
    }
}

#[tonic::async_trait]
impl PostgresService for PostgresGrpcService {
    async fn store_page(
        &self,
        request: Request<StorePageRequest>,
    ) -> Result<Response<StorePageResponse>, Status> {
        let req = request.into_inner();
        self.record_message_size(req.encoded_len() as u64);

        let page = req
            .page
            .ok_or_else(|| Status::invalid_argument("page must be provided"))?;
        let proto_key = page
            .page_key
            .ok_or_else(|| Status::invalid_argument("page.page_key must be provided"))?;

        let page_kind = match ProtoPageKind::try_from(proto_key.page_kind) {
            Ok(ProtoPageKind::Anime) => PageKind::Anime,
            Ok(ProtoPageKind::Episode) => PageKind::Episode,
            Ok(ProtoPageKind::Character) => PageKind::Character,
            Ok(ProtoPageKind::Unspecified) | Err(_) => {
                return Err(Status::invalid_argument(
                    "page.page_key.page_kind must be specified",
                ));
            }
        };

        let fetched_ts = page
            .fetched_at
            .ok_or_else(|| Status::invalid_argument("page.fetched_at must be provided"))?;
        let fetched_at = DateTime::<Utc>::from_timestamp(
            fetched_ts.seconds,
            u32::try_from(fetched_ts.nanos)
                .map_err(|_| Status::invalid_argument("page.fetched_at.nanos is invalid"))?,
        )
        .ok_or_else(|| Status::invalid_argument("page.fetched_at is out of range"))?;

        let dead_page_review = is_dead_page_review_enabled(&proto_key.provider);
        let responses = page
            .responses
            .into_iter()
            .map(|resp| PageResponseInput {
                request_name: resp.request_name,
                content_type: resp.content_type,
                body: resp.body,
            })
            .collect();

        let input = StorePageInput {
            page_key: PageKeyInput {
                provider: proto_key.provider,
                page_kind,
                address: proto_key.address,
            },
            fetched_at,
            is_dead: page.is_dead,
            responses,
            is_not_found: page.is_not_found,
            dead_page_review,
        };

        let outcome = store_page(&self.pool, input)
            .await
            .map_err(|err| map_db_error(&err))?;

        Ok(Response::new(StorePageResponse {
            has_changed: outcome.has_changed,
        }))
    }

    async fn load_anime(
        &self,
        request: Request<LoadAnimeRequest>,
    ) -> Result<Response<LoadAnimeResponse>, Status> {
        let anime_id = Uuid::parse_str(&request.into_inner().anime_id)
            .map_err(|err| Status::invalid_argument(format!("invalid anime_id UUID: {err}")))?;

        let outcome = load_anime(&self.pool, anime_id)
            .await
            .map_err(|err| map_db_error(&err))?;

        let source_pages = outcome
            .source_pages
            .into_iter()
            .map(|page| ProtoStoredPage {
                page_key: Some(to_proto_page_key(page.page_key)),
                fetched_at: Some(to_proto_timestamp(page.fetched_at)),
                is_dead: page.is_dead,
                responses: page
                    .responses
                    .into_iter()
                    .map(|resp| ProtoPageResponse {
                        request_name: resp.request_name,
                        content_type: resp.content_type,
                        body: resp.body,
                    })
                    .collect(),
                is_not_found: page.is_not_found,
            })
            .collect();

        let partial_source_pages = outcome
            .partial_source_pages
            .into_iter()
            .map(to_proto_page_key)
            .collect();

        let corrections = outcome
            .corrections
            .into_iter()
            .map(|corr| ProtoFieldCorrection {
                field_path: corr.field_path,
                value: Some(prost_types::Any {
                    type_url: corr.value_message,
                    value: corr.value,
                }),
                approved_by: corr.approved_by,
                approved_at: Some(to_proto_timestamp(corr.approved_at)),
            })
            .collect();

        let response = LoadAnimeResponse {
            version: outcome.version,
            source_pages,
            partial_source_pages,
            corrections,
        };
        self.record_message_size(response.encoded_len() as u64);

        Ok(Response::new(response))
    }

    async fn save_anime(
        &self,
        request: Request<SaveAnimeRequest>,
    ) -> Result<Response<SaveAnimeResponse>, Status> {
        let traceparent = request
            .metadata()
            .get("traceparent")
            .and_then(|val| val.to_str().ok())
            .map(str::to_owned);
        let req = request.into_inner();
        let anime = req
            .anime
            .ok_or_else(|| Status::invalid_argument("anime must be provided"))?;

        let input = build_save_anime_input(
            &req.anime_id,
            req.expected_version,
            req.merge_version,
            anime,
            traceparent,
        )?;

        let outcome = save_anime(&self.pool, &input)
            .await
            .map_err(|err| map_db_error(&err))?;

        Ok(Response::new(SaveAnimeResponse {
            has_changed: outcome.has_changed,
            version: outcome.version,
        }))
    }
}

fn to_proto_page_key(key: PageKeyInput) -> ProtoPageKey {
    let page_kind = match key.page_kind {
        PageKind::Anime => ProtoPageKind::Anime,
        PageKind::Episode => ProtoPageKind::Episode,
        PageKind::Character => ProtoPageKind::Character,
    };
    ProtoPageKey {
        provider: key.provider,
        page_kind: page_kind as i32,
        address: key.address,
    }
}

fn to_proto_timestamp(ts: DateTime<Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: ts.timestamp(),
        nanos: i32::try_from(ts.timestamp_subsec_nanos()).unwrap_or(0),
    }
}

fn is_dead_page_review_enabled(provider: &str) -> bool {
    let key = format!("DEAD_PAGE_REVIEW_{}", provider.to_ascii_uppercase());
    match std::env::var(key) {
        Ok(val) => !matches!(
            val.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no"
        ),
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_db_error_maps_error_kinds_to_expected_grpc_codes() {
        assert_eq!(
            map_db_error(&postgres_db::Error::version_conflict("stale")).code(),
            tonic::Code::Aborted
        );
        assert_eq!(
            map_db_error(&postgres_db::Error::not_found()).code(),
            tonic::Code::NotFound
        );
        assert_eq!(
            map_db_error(&postgres_db::Error::invalid_argument("bad address")).code(),
            tonic::Code::InvalidArgument
        );
    }
}
