//! Temporal workflow definitions for `RefreshEpisodeWorkflow` and `RefreshCharacterWorkflow`.

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
mod definitions {
    use temporalio_macros::{workflow, workflow_methods};
    use temporalio_sdk::{
        SyncWorkflowContext, WorkflowContext, WorkflowContextView, WorkflowResult,
    };

    use crate::refresh_anime_types::{
        FetchRequestedSignal, RefreshEntityInput, RefreshEntityKind, RefreshEntityOutcome,
    };
    use crate::refresh_anime_workflow::{
        EntityRefreshWorkflowState, RefreshCoreState, run_entity_refresh,
    };

    /// Durable workflow refreshing one episode (`refresh-episode-<episode_id>`).
    #[workflow]
    pub struct RefreshEpisodeWorkflow {
        core: RefreshCoreState,
    }

    impl EntityRefreshWorkflowState for RefreshEpisodeWorkflow {
        const ENTITY_KIND: RefreshEntityKind = RefreshEntityKind::Episode;

        fn core_state(&self) -> &RefreshCoreState {
            &self.core
        }

        fn core_state_mut(&mut self) -> &mut RefreshCoreState {
            &mut self.core
        }
    }

    #[workflow_methods]
    impl RefreshEpisodeWorkflow {
        /// Initializes `RefreshEpisodeWorkflow` from `input`.
        #[init]
        fn new(_ctx: &WorkflowContextView, input: RefreshEntityInput) -> Self {
            Self {
                core: RefreshCoreState::new(input),
            }
        }

        /// Handles a `fetch_requested` signal, enqueueing page refreshes and raising priority.
        #[signal]
        pub fn fetch_requested(
            &mut self,
            _ctx: &mut SyncWorkflowContext<Self>,
            signal: FetchRequestedSignal,
        ) {
            self.core.apply_fetch_requested(signal);
        }

        /// Handles a `merge_requested` signal by setting the merge flag checked before exit.
        #[signal]
        pub fn merge_requested(&mut self, _ctx: &mut SyncWorkflowContext<Self>) {
            self.core.apply_merge_requested();
        }

        /// Executes the episode refresh lifecycle and returns its outcome summary.
        ///
        /// # Errors
        ///
        /// Returns a workflow failure if `list_pages`, `merge_episode`, or `record_refresh` fails.
        #[run]
        pub async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<RefreshEntityOutcome> {
            Ok(run_entity_refresh(ctx).await?.outcome)
        }
    }

    /// Durable workflow refreshing one character (`refresh-character-<character_id>`).
    #[workflow]
    pub struct RefreshCharacterWorkflow {
        core: RefreshCoreState,
    }

    impl EntityRefreshWorkflowState for RefreshCharacterWorkflow {
        const ENTITY_KIND: RefreshEntityKind = RefreshEntityKind::Character;

        fn core_state(&self) -> &RefreshCoreState {
            &self.core
        }

        fn core_state_mut(&mut self) -> &mut RefreshCoreState {
            &mut self.core
        }
    }

    #[workflow_methods]
    impl RefreshCharacterWorkflow {
        /// Initializes `RefreshCharacterWorkflow` from `input`.
        #[init]
        fn new(_ctx: &WorkflowContextView, input: RefreshEntityInput) -> Self {
            Self {
                core: RefreshCoreState::new(input),
            }
        }

        /// Handles a `fetch_requested` signal, enqueueing page refreshes and raising priority.
        #[signal]
        pub fn fetch_requested(
            &mut self,
            _ctx: &mut SyncWorkflowContext<Self>,
            signal: FetchRequestedSignal,
        ) {
            self.core.apply_fetch_requested(signal);
        }

        /// Handles a `merge_requested` signal by setting the merge flag checked before exit.
        #[signal]
        pub fn merge_requested(&mut self, _ctx: &mut SyncWorkflowContext<Self>) {
            self.core.apply_merge_requested();
        }

        /// Executes the character refresh lifecycle and returns its outcome summary.
        ///
        /// # Errors
        ///
        /// Returns a workflow failure if `list_pages`, `merge_character`, or `record_refresh` fails.
        #[run]
        pub async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<RefreshEntityOutcome> {
            Ok(run_entity_refresh(ctx).await?.outcome)
        }
    }
}

#[doc(inline)]
pub use definitions::{RefreshCharacterWorkflow, RefreshEpisodeWorkflow};
