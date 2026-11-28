//! Temporal activity implementations for `update-service` and `provider-<slug>` queues.

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
#[allow(
    clippy::unused_async_trait_impl,
    reason = "temporalio_macros copies each async #[activity] into a generated trait impl"
)]
mod definitions {
    use std::future::Future;
    use std::sync::Arc;
    use std::time::Duration;

    use temporalio_macros::activities;
    use temporalio_sdk::activities::{ActivityContext, ActivityError, ActivityInfo};
    use tracing::{Level, event};

    use crate::ports::{MergeAttemptOutcome, PortError, UpdateServicePorts};
    use crate::refresh_anime_types::{
        ListEntityPagesOutput, MergeAnimeInput, MergeAnimeOutput, RecordRefreshInput,
        RefreshPageInput, RefreshPageOutput, StartEntityWorkflowsInput,
    };

    /// Maximum `LoadAnime` -> `MergeAnime` -> `SaveAnime` retries inside one `merge_anime` attempt.
    ///
    /// Matches `docs/architecture/update_service.md`: optimistic concurrency conflicts resolve
    /// quickly and should not consume the Temporal retry policy's 5-attempt budget.
    pub const MAX_ABORTED_RETRIES_PER_ATTEMPT: usize = 10;

    /// Interval between `refresh_page` heartbeats while `FetchPage` + `StorePage` run.
    ///
    /// Set to one third of the 60-second heartbeat timeout so a healthy fetch always delivers
    /// multiple heartbeats while a crashed worker is detected within a minute.
    pub const REFRESH_PAGE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);

    /// Core Update Service activities running on the `update-service` task queue.
    #[derive(Debug)]
    pub struct UpdateServiceActivities {
        ports: UpdateServicePorts,
        heartbeat_interval: Duration,
    }

    impl UpdateServiceActivities {
        /// Creates core Update Service activities backed by `ports` with a 20-second heartbeat.
        #[must_use]
        pub fn new(ports: UpdateServicePorts) -> Self {
            Self {
                ports,
                heartbeat_interval: REFRESH_PAGE_HEARTBEAT_INTERVAL,
            }
        }

        /// Overrides the heartbeat interval for fast tests.
        #[cfg(feature = "test-util")]
        #[must_use]
        pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
            self.heartbeat_interval = interval;
            self
        }
    }

    /// Provider page-fetch activities running on each `provider-<slug>` task queue.
    #[derive(Debug)]
    pub struct ProviderActivities {
        ports: UpdateServicePorts,
        heartbeat_interval: Duration,
    }

    impl ProviderActivities {
        /// Creates provider activities backed by `ports` with the 20-second heartbeat.
        #[must_use]
        pub fn new(ports: UpdateServicePorts) -> Self {
            Self {
                ports,
                heartbeat_interval: REFRESH_PAGE_HEARTBEAT_INTERVAL,
            }
        }

        /// Overrides the heartbeat interval for fast tests.
        #[cfg(feature = "test-util")]
        #[must_use]
        pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
            self.heartbeat_interval = interval;
            self
        }
    }

    /// Returns `true` when this attempt is the final retry allowed by the activity's policy.
    fn is_final_attempt(info: &ActivityInfo) -> bool {
        let max_attempts = info
            .retry_policy
            .as_ref()
            .map_or(0, temporalio_common::RetryPolicy::maximum_attempts);
        max_attempts > 0 && info.attempt >= max_attempts
    }

    /// Drives the future returned by `start_operation` while heartbeating and watching cancellation.
    ///
    /// Returns [`ActivityError::Cancelled`] before calling `start_operation` if the context is
    /// already cancelled, so cancelled retries never start a new RPC.
    async fn run_with_heartbeats<T, F>(
        ctx: &ActivityContext,
        heartbeat_interval: Duration,
        heartbeat_detail: &str,
        start_operation: impl FnOnce() -> F,
    ) -> Result<T, ActivityError>
    where
        F: Future<Output = Result<T, PortError>>,
    {
        if ctx.is_cancelled() {
            return Err(ActivityError::Cancelled { details: None });
        }

        let mut heartbeat_ticker = tokio::time::interval(heartbeat_interval);
        heartbeat_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Consume the immediate first tick so heartbeats fire at each interval.
        heartbeat_ticker.tick().await;

        let operation = start_operation();
        tokio::pin!(operation);

        loop {
            tokio::select! {
                biased;
                () = ctx.cancelled() => {
                    return Err(ActivityError::Cancelled { details: None });
                }
                outcome = &mut operation => {
                    return outcome.map_err(PortError::into_activity_error);
                }
                _ = heartbeat_ticker.tick() => {
                    if let Err(err) = ctx.record_heartbeat(heartbeat_detail.to_owned()).await {
                        event!(
                            name: "update_service.activity.heartbeat_failed",
                            Level::DEBUG,
                            heartbeat.detail = heartbeat_detail,
                            error.message = %err,
                            "failed to encode activity heartbeat {{heartbeat.detail}}: {{error.message}}",
                        );
                    }
                }
            }
        }
    }

    #[activities]
    impl UpdateServiceActivities {
        /// Lists an entity's linked pages, fetch state, and stored `merge_version`.
        ///
        /// # Errors
        ///
        /// Returns a non-retryable [`ActivityError`] on invalid arguments or missing entities,
        /// or a retryable [`ActivityError`] on transient transport/server failures.
        #[activity]
        pub async fn list_pages(
            self: Arc<Self>,
            _ctx: ActivityContext,
            anime_id: String,
        ) -> Result<ListEntityPagesOutput, ActivityError> {
            self.ports
                .catalogue()
                .list_entity_pages(anime_id)
                .await
                .map_err(PortError::into_activity_error)
        }

        /// Loads, merges, and saves an anime, retrying `ABORTED` conflicts up to 10 times.
        ///
        /// Emits heartbeats every 20 seconds and watches `ctx.cancelled()` across all retry
        /// attempts so workflow cancellations and graceful worker shutdowns stop promptly.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] when Temporal cancels the activity, a retryable
        /// [`ActivityError`] after 10 consecutive version conflicts or on transient gRPC failures,
        /// or a non-retryable error on permanent gRPC statuses.
        #[activity]
        pub async fn merge_anime(
            self: Arc<Self>,
            ctx: ActivityContext,
            input: MergeAnimeInput,
        ) -> Result<MergeAnimeOutput, ActivityError> {
            for attempt in 1..=MAX_ABORTED_RETRIES_PER_ATTEMPT {
                let heartbeat_detail = format!("merge:{}:{attempt}", input.anime_id);
                match run_with_heartbeats(&ctx, self.heartbeat_interval, &heartbeat_detail, || {
                    self.ports
                        .catalogue()
                        .merge_anime_attempt(input.anime_id.clone(), input.merge_version)
                })
                .await?
                {
                    MergeAttemptOutcome::Saved(output) => return Ok(output),
                    MergeAttemptOutcome::VersionConflict(reason) => {
                        event!(
                            name: "update_service.merge_anime.version_conflict",
                            Level::DEBUG,
                            anime.id = %input.anime_id,
                            merge.attempt = attempt,
                            conflict.reason = %reason,
                            "retrying merge_anime for {{anime.id}} after ABORTED attempt {{merge.attempt}}: {{conflict.reason}}",
                        );
                    }
                }
            }
            Err(PortError::retryable(format!(
                "merge_anime exceeded {MAX_ABORTED_RETRIES_PER_ATTEMPT} ABORTED retries for {}",
                input.anime_id
            ))
            .into_activity_error())
        }

        /// Starts top-level downstream workflows (`RefreshEpisodeWorkflow` / `WriteSynopsisWorkflow`).
        ///
        /// Emits heartbeats every 20 seconds and watches `ctx.cancelled()` while starting large
        /// episode batches.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] on cancellation, or an [`ActivityError`] if the
        /// Temporal client fails to start any workflow.
        #[activity]
        pub async fn start_entity_workflows(
            self: Arc<Self>,
            ctx: ActivityContext,
            input: StartEntityWorkflowsInput,
        ) -> Result<usize, ActivityError> {
            let heartbeat_detail = format!("start:{}:{}", input.anime_id, input.episode_ids.len());
            run_with_heartbeats(&ctx, self.heartbeat_interval, &heartbeat_detail, || {
                self.ports.starter().start_entity_workflows(input)
            })
            .await
        }

        /// Records the final refresh outcome via `PostgresService.RecordRefresh`.
        ///
        /// # Errors
        ///
        /// Returns a non-retryable [`ActivityError`] when the entity is missing or arguments
        /// are invalid, or a retryable [`ActivityError`] on transient failures.
        #[activity]
        pub async fn record_refresh(
            self: Arc<Self>,
            _ctx: ActivityContext,
            input: RecordRefreshInput,
        ) -> Result<(), ActivityError> {
            self.ports
                .catalogue()
                .record_refresh(input)
                .await
                .map_err(PortError::into_activity_error)
        }
    }

    #[activities]
    impl ProviderActivities {
        /// Fetches and stores one provider page while heartbeating every 20 seconds.
        ///
        /// Transient fetch failures are returned to Temporal for exponential retry up to the
        /// policy's maximum attempts. On a provider block (`ServiceBlockedError`) or on the
        /// final allowed attempt, the failure is recorded through `StorePage` and returned as
        /// `has_failed = true` so the refresh continues with the remaining pages.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] when Temporal cancels the activity, a
        /// non-retryable [`ActivityError`] on permanent gRPC statuses, or a retryable
        /// [`ActivityError`] while retry attempts remain.
        #[activity]
        pub async fn refresh_page(
            self: Arc<Self>,
            ctx: ActivityContext,
            input: RefreshPageInput,
        ) -> Result<RefreshPageOutput, ActivityError> {
            let page_key = input.page_key;
            let heartbeat_detail = format!("{}:{}", page_key.provider, page_key.address);
            let fetch_outcome =
                run_with_heartbeats(&ctx, self.heartbeat_interval, &heartbeat_detail, || {
                    self.ports
                        .catalogue()
                        .fetch_and_store_page(page_key.clone(), input.max_age_seconds)
                })
                .await;

            match fetch_outcome {
                Ok(output) => Ok(output),
                Err(ActivityError::Cancelled { details }) => {
                    Err(ActivityError::Cancelled { details })
                }
                Err(activity_error) => {
                    let port_error = activity_error.downcast_port_error();
                    let should_record_failure = port_error.is_some_and(|err| {
                        err.is_provider_blocked()
                            || (err.is_retryable() && is_final_attempt(ctx.info()))
                    });
                    if !should_record_failure {
                        return Err(activity_error);
                    }
                    event!(
                        name: "update_service.refresh_page.recording_failure",
                        Level::WARN,
                        provider = %page_key.provider,
                        page.address = %page_key.address,
                        activity.attempt = ctx.info().attempt,
                        "recording page fetch failure for {{provider}}:{{page.address}} on attempt {{activity.attempt}}",
                    );
                    run_with_heartbeats(&ctx, self.heartbeat_interval, &heartbeat_detail, || {
                        self.ports.catalogue().record_page_failure(page_key.clone())
                    })
                    .await?;
                    Ok(RefreshPageOutput {
                        page_key,
                        has_changed: false,
                        has_failed: true,
                    })
                }
            }
        }
    }

    trait ActivityErrorExt {
        fn downcast_port_error(&self) -> Option<&PortError>;
    }

    impl ActivityErrorExt for ActivityError {
        fn downcast_port_error(&self) -> Option<&PortError> {
            match self {
                Self::Application(failure) => failure.source_error().downcast_ref::<PortError>(),
                Self::Cancelled { .. } | Self::WillCompleteAsync | _ => None,
            }
        }
    }
}

#[doc(inline)]
pub use definitions::{
    MAX_ABORTED_RETRIES_PER_ATTEMPT, ProviderActivities, REFRESH_PAGE_HEARTBEAT_INTERVAL,
    UpdateServiceActivities,
};
