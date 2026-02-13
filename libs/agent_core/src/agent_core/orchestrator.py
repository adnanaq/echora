"""Core orchestration loop for staged agentic-RAG search."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from atomic_agents import AtomicAgent
from langfuse import observe

from agent_core.context import RetrievedContextProvider
from agent_core.orchestration_validation import normalize_step_for_turn
from agent_core.retrieval import (
    GraphNotAvailableError,
    PostgresGraphExecutor,
    QdrantExecutor,
)
from agent_core.schemas import (
    AnswerInput,
    AnswerOutput,
    GraphResult,
    RewriteInput,
    RewriteOutput,
    SearchAIEvidence,
    SourceSelectionInput,
    SourceSelectionOutput,
    SufficiencyInput,
    SufficiencyOutput,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Runs the staged bounded loop for ``SearchAI`` requests."""

    def __init__(
        self,
        rewrite: AtomicAgent[RewriteInput, RewriteOutput],
        source_selector: AtomicAgent[SourceSelectionInput, SourceSelectionOutput],
        answer: AtomicAgent[AnswerInput, AnswerOutput],
        sufficiency: AtomicAgent[SufficiencyInput, SufficiencyOutput],
        qdrant: QdrantExecutor,
        graph: PostgresGraphExecutor | None = None,
        *,
        max_turns_default: int = 4,
        qdrant_limit: int = 10,
        qdrant_context_top_k: int = 3,
        qdrant_min_score_text: float = 0.6,
        qdrant_min_score_image: float = 0.45,
        qdrant_min_score_multivector: float = 0.35,
    ) -> None:
        """Initializes staged dependencies and safety limits.

        Args:
            rewrite: Atomic rewrite agent producing ``RewriteOutput``.
            source_selector: Atomic source-selector producing ``SourceSelectionOutput``.
            answer: Atomic answer agent producing ``AnswerOutput``.
            sufficiency: Atomic sufficiency agent producing ``SufficiencyOutput``.
            qdrant: Qdrant retrieval executor.
            graph: Optional Postgres graph executor.
            max_turns_default: Default loop turn cap when request does not provide one.
            qdrant_limit: Max Qdrant hits per retrieval step.
            qdrant_context_top_k: Max threshold-qualified rows injected into context.
            qdrant_min_score_text: Min score gate for text-only Qdrant search.
            qdrant_min_score_image: Min score gate for image-only Qdrant search.
            qdrant_min_score_multivector: Min score gate for fused text+image search.
        """
        self._rewrite = rewrite
        self._source_selector = source_selector
        self._answer = answer
        self._sufficiency = sufficiency
        self._qdrant = qdrant
        self._graph = graph or PostgresGraphExecutor()
        self._max_turns_default = max_turns_default
        self._qdrant_limit = qdrant_limit
        self._qdrant_context_top_k = qdrant_context_top_k
        self._qdrant_min_score_text = qdrant_min_score_text
        self._qdrant_min_score_image = qdrant_min_score_image
        self._qdrant_min_score_multivector = qdrant_min_score_multivector
        self._retrieved = RetrievedContextProvider()

        self._source_selector.register_context_provider("retrieved", self._retrieved)
        self._answer.register_context_provider("retrieved", self._retrieved)
        self._sufficiency.register_context_provider("retrieved", self._retrieved)

    @observe()
    async def run_search_ai(
        self,
        query: str,
        image_query: str | None = None,
        max_turns: int | None = None,
        request_id: str | None = None,
    ) -> AnswerOutput:
        """Executes the staged rewrite/select/retrieve/answer/sufficiency loop.

        Args:
            query: User text query.
            image_query: Optional image query (data URL or raw base64 string).
            max_turns: Optional override for the loop turn cap.
            request_id: Optional request correlation id for stage-level logging.

        Returns:
            Final response model suitable for the gRPC layer.
        """
        warnings: list[str] = []

        turns = max_turns or self._max_turns_default
        req = request_id or "n/a"
        last_summary: str | None = None
        last_step: SourceSelectionOutput | None = None
        last_draft: AnswerOutput | None = None
        any_hits = False
        attempted_actions: list[Literal["qdrant_search", "pg_graph"]] = []
        last_search_similarity_score: float | None = None
        rewritten_query_stagnation = 0

        rewrite: RewriteOutput = await self._call_rewrite_agent(
            RewriteInput(user_query=query, has_image_query=bool(image_query))
        )
        rewritten_query = (rewrite.rewritten_query or "").strip() or query
        requires_graph_traversal = rewrite.requires_graph_traversal
        self._retrieved.add_card(
            title="Rewrite",
            body=f"rewritten_query={rewritten_query}\nneeds_external_context={rewrite.needs_external_context}\nrequires_graph_traversal={requires_graph_traversal}",
            data={
                "rewritten_query": rewritten_query,
                "needs_external_context": rewrite.needs_external_context,
                "requires_graph_traversal": requires_graph_traversal,
            },
        )

        # Fast path for general knowledge questions (e.g., "What is anime?").
        # Drafts from LLM knowledge without retrieval; falls back to retrieval if insufficient.
        if not rewrite.needs_external_context:
            draft = await self._call_answer_agent(
                AnswerInput(user_query=query, rewritten_query=rewritten_query)
            )
            last_draft = draft
            report: SufficiencyOutput = await self._call_sufficient_agent(
                SufficiencyInput(
                    user_query=query,
                    draft_answer=draft.answer,
                    last_search_similarity_score=None,
                    attempted_actions=[],
                )
            )
            if report.sufficient:
                # Direct answer sufficient; return without retrieval.
                evidence = draft.evidence.model_copy(
                    update={"termination_reason": "direct_answer_sufficient"}
                )
                return draft.model_copy(
                    update={
                        "evidence": evidence,
                        "warnings": [*warnings, *draft.warnings],
                    }
                )

            # Direct answer insufficient; switch to retrieval loop.
            warnings.append(
                "Direct-answer path was insufficient; switching to retrieval loop."
            )
            rewritten_query = await self._rewrite_with_feedback(
                query=query,
                image_query=image_query,
                current_rewritten_query=rewritten_query,
                missing_information=self._rewrite_feedback_from_sufficiency(report),
                last_retrieval_summary=last_summary,
            )
            self._retrieved.add_card(
                title="Rewrite refinement",
                body=f"rewritten_query={rewritten_query}",
                data={"rewritten_query": rewritten_query},
            )

        for turn in range(1, turns + 1):
            step_input = SourceSelectionInput(
                user_query=query,
                rewritten_query=rewritten_query,
                has_image_query=bool(image_query),
                turn=turn,
                last_action=last_step.action if last_step is not None else None,
                last_summary=last_summary,
                warnings=warnings[-5:],
                attempted_actions=list(attempted_actions or [])[-4:],
                requires_graph_traversal=requires_graph_traversal,
            )
            step: SourceSelectionOutput = await self._call_selector_agent(step_input)
            normalized_step, step_warning = normalize_step_for_turn(
                step=step,
                rewritten_query=rewritten_query,
                request_image_query=image_query,
            )
            if step_warning:
                warnings.append(step_warning)
                continue

            if normalized_step is None:
                continue
            step = normalized_step
            last_step = step

            if step.action == "qdrant_search":
                attempted_actions.append("qdrant_search")

                (
                    last_summary,
                    top_score,
                    qualified_rows,
                ) = await self._process_qdrant_retrieval(
                    step=step,
                    retrieved=self._retrieved,
                )
                last_search_similarity_score = top_score
                if qualified_rows:
                    any_hits = True

                # Calculate threshold for feedback messages
                search_mode = self._search_mode(
                    text_query=step.search_intent.query if step.search_intent else None,
                    image_query=step.search_intent.image_query
                    if step.search_intent
                    else None,
                )
                score_threshold = self._score_threshold(search_mode)

                if top_score is None or top_score < score_threshold:
                    warnings.append(
                        f"Top similarity score {top_score if top_score is not None else 'n/a'} "
                        f"is below threshold {score_threshold:.3f} ({search_mode}); retrying."
                    )
                    rewritten_query = await self._rewrite_with_feedback(
                        query=query,
                        image_query=image_query,
                        current_rewritten_query=rewritten_query,
                        missing_information=[
                            "Need a query phrasing that retrieves higher-confidence matches.",
                        ],
                        last_retrieval_summary=last_summary,
                    )
                    continue
                if not qualified_rows:
                    warnings.append("No Qdrant rows met the score threshold; retrying.")
                    rewritten_query = await self._rewrite_with_feedback(
                        query=query,
                        image_query=image_query,
                        current_rewritten_query=rewritten_query,
                        missing_information=[
                            "Need terms that better match indexed payload titles/aliases.",
                        ],
                        last_retrieval_summary=last_summary,
                    )
                    continue

            elif step.action == "pg_graph":
                attempted_actions.append("pg_graph")

                last_summary, graph_success = await self._process_graph_retrieval(
                    step=step,
                    retrieved=self._retrieved,
                    request_id=req,
                    turn=turn,
                )
                if graph_success:
                    any_hits = True
                else:
                    warnings.append(
                        "PostgreSQL graph executor not available yet; answering without graph traversal."
                    )

            else:
                # Unreachable because ``normalize_step_for_turn`` gates action values.
                continue

            draft: AnswerOutput = await self._call_answer_agent(
                AnswerInput(user_query=query, rewritten_query=rewritten_query)
            )
            last_draft = draft
            self._retrieved.add_card(
                title="Draft answer",
                body=(draft.answer or "")[:500],
                data={
                    "num_sources": draft.evidence.num_sources,
                    "result_entities": len(draft.result_entities),
                    "source_entities": len(draft.source_entities),
                },
            )

            suff: SufficiencyOutput = await self._call_sufficient_agent(
                SufficiencyInput(
                    user_query=query,
                    draft_answer=draft.answer,
                    last_search_similarity_score=last_search_similarity_score,
                    attempted_actions=[a for a in attempted_actions],
                )
            )
            self._retrieved.add_card(
                title="Sufficiency",
                body=f"sufficient={suff.sufficient} missing={suff.missing}",
                data={"sufficient": suff.sufficient, "missing": suff.missing},
            )

            if suff.sufficient:
                evidence_update: dict[str, Any] = {
                    "termination_reason": "sufficient_with_retrieval"
                }
                if last_search_similarity_score is not None:
                    evidence_update["search_similarity_score"] = (
                        last_search_similarity_score
                    )
                evidence = draft.evidence.model_copy(update=evidence_update)
                return draft.model_copy(
                    update={
                        "evidence": evidence,
                        "warnings": [*warnings, *draft.warnings],
                    }
                )

            # Check if sufficiency explicitly requested graph traversal
            if suff.need_graph_traversal and "pg_graph" not in attempted_actions:
                warnings.append(
                    "Sufficiency detected relationship query needs pg_graph; "
                    "requesting graph traversal on next turn."
                )
                # Add hint to rewrite feedback to emphasize graph need
                rewrite_feedback = [
                    *self._rewrite_feedback_from_sufficiency(suff),
                    "Query requires graph traversal for relationships/comparisons.",
                ]
            else:
                rewrite_feedback = self._rewrite_feedback_from_sufficiency(suff)
            rewritten_candidate = await self._rewrite_with_feedback(
                query=query,
                image_query=image_query,
                current_rewritten_query=rewritten_query,
                missing_information=rewrite_feedback,
                last_retrieval_summary=last_summary,
            )
            if rewritten_candidate == rewritten_query:
                rewritten_query_stagnation += 1
                warnings.append("Rewrite feedback did not change rewritten query.")
            else:
                rewritten_query_stagnation = 0
                rewritten_query = rewritten_candidate
                self._retrieved.add_card(
                    title="Rewrite refinement",
                    body=f"rewritten_query={rewritten_query}",
                    data={
                        "rewritten_query": rewritten_query,
                        "missing_information": rewrite_feedback,
                    },
                )

            if rewritten_query_stagnation >= 2 and len(attempted_actions) >= 2:
                warnings.append(
                    "Repeated insufficient loop detected; selector should choose an alternate retrieval lane."
                )

        # Hard stop: use last draft as best effort if present.
        if last_draft is not None:
            evidence_update: dict[str, Any] = {
                "termination_reason": "max_turns_best_effort"
            }
            if last_search_similarity_score is not None:
                evidence_update["search_similarity_score"] = (
                    last_search_similarity_score
                )
            evidence = last_draft.evidence.model_copy(update=evidence_update)
            return last_draft.model_copy(
                update={
                    "evidence": evidence,
                    "warnings": [
                        *warnings,
                        f"Reached max_turns={turns}.",
                        *last_draft.warnings,
                    ],
                }
            )

        # No-match is a business outcome, not a system error.
        evidence = SearchAIEvidence(
            termination_reason="no_match_after_max_turns",
            search_similarity_score=last_search_similarity_score or 0.0,
            num_sources=0,
        )

        answer = "I couldn't find a confident match for this query yet."
        warning_code = "NO_MATCH_AFTER_MAX_TURNS"
        if any_hits:
            answer = "I found partial signals, but not enough confident evidence to answer this reliably yet."
            warning_code = "BEST_EFFORT_PARTIAL_AFTER_MAX_TURNS"

        return AnswerOutput(
            answer=answer,
            source_entities=[],
            result_entities=[],
            evidence=evidence,
            warnings=[*warnings, warning_code],
        )

    async def _process_qdrant_retrieval(
        self,
        *,
        step: SourceSelectionOutput,
        retrieved: RetrievedContextProvider,
    ) -> tuple[str, float | None, list[dict[str, Any]]]:
        """Executes Qdrant search and returns summary, top score, and qualified rows.

        Args:
            step: Source selection output with search intent.
            retrieved: Context provider for logging.

        Returns:
            Tuple of (summary, top_score, qualified_rows).
        """
        assert step.search_intent is not None

        result = await self._qdrant.search(step.search_intent, limit=self._qdrant_limit)
        search_mode = self._search_mode(
            text_query=step.search_intent.query,
            image_query=step.search_intent.image_query,
        )
        score_threshold = self._score_threshold(search_mode)
        top_score = (
            self._extract_similarity_score(result.raw_data[0])
            if result.raw_data
            else None
        )

        qualified_rows: list[dict[str, Any]] = []
        for row in result.raw_data:
            score = self._extract_similarity_score(row)
            if score is not None and score >= score_threshold:
                qualified_rows.append(row)
            if len(qualified_rows) >= self._qdrant_context_top_k:
                break

        retrieved.add_card(
            title=f"Qdrant search ({step.search_intent.entity_type.value})",
            body=self._qdrant_context_body(
                summary=result.summary,
                search_mode=search_mode,
                threshold=score_threshold,
                top_score=top_score,
                qualified_rows=qualified_rows,
            ),
            data={
                "count": result.count,
                "search_mode": search_mode,
                "score_threshold": score_threshold,
                "top_score": top_score,
                "qualified_count": len(qualified_rows),
            },
        )

        return result.summary, top_score, qualified_rows

    async def _process_graph_retrieval(
        self,
        *,
        step: SourceSelectionOutput,
        retrieved: RetrievedContextProvider,
        request_id: str,
        turn: int,
    ) -> tuple[str, bool]:
        """Executes graph traversal and returns summary and success flag.

        Args:
            step: Source selection output with graph intent.
            retrieved: Context provider for logging.
            request_id: Request correlation ID.
            turn: Current turn number.

        Returns:
            Tuple of (summary, success).
        """
        assert step.graph_intent is not None

        try:
            graph: GraphResult = await self._graph.execute(step.graph_intent)
            retrieved.add_card(
                title="Postgres graph", body=graph.summary, data={"count": graph.count}
            )
            return graph.summary, True
        except GraphNotAvailableError:
            logger.warning(
                "agent.search_ai.pg_graph.unavailable request_id=%s turn=%d",
                request_id,
                turn,
            )
            retrieved.add_card(title="Postgres graph", body="Not available.", data={})
            return "Graph step skipped (not available).", False

    async def _rewrite_with_feedback(
        self,
        *,
        query: str,
        image_query: str | None,
        current_rewritten_query: str,
        missing_information: list[str],
        last_retrieval_summary: str | None = None,
    ) -> str:
        """Runs rewrite stage using sufficiency/retrieval feedback and returns next query.
        Args:
            query: Original user query.
            image_query: Optional image query payload.
            current_rewritten_query: Current query to refine.
            missing_information: List of missing facts/constraints from feedback.
            last_retrieval_summary: Optional summary from last retrieval.
        Returns:
            Refined query string.
        """
        rewrite_input = RewriteInput(
            user_query=query,
            has_image_query=bool(image_query),
            current_rewritten_query=current_rewritten_query,
            missing_information=missing_information,
            last_retrieval_summary=last_retrieval_summary,
        )
        rewrite = await self._call_rewrite_agent(rewrite_input)
        return (
            (rewrite.rewritten_query or "").strip() or current_rewritten_query or query
        )

    @staticmethod
    def _rewrite_feedback_from_sufficiency(sufficiency: SufficiencyOutput) -> list[str]:
        """Converts sufficiency output into compact rewrite feedback."""
        feedback = list(sufficiency.missing[:4])
        if sufficiency.rationale:
            feedback.append(sufficiency.rationale)
        if not feedback:
            feedback.append("Improve retrieval precision for the user request.")
        return feedback

    @staticmethod
    def _search_mode(*, text_query: str | None, image_query: str | None) -> str:
        """Classifies retrieval modality for score-gating thresholds."""
        if text_query and image_query:
            return "multivector"
        if image_query:
            return "image"
        return "text"

    def _score_threshold(self, mode: str) -> float:
        """Returns the score threshold configured for the selected search mode."""
        if mode == "multivector":
            return self._qdrant_min_score_multivector
        if mode == "image":
            return self._qdrant_min_score_image
        return self._qdrant_min_score_text

    @staticmethod
    def _extract_similarity_score(row: dict[str, Any]) -> float | None:
        """Extracts a numeric similarity score from one Qdrant result row."""
        score = row.get("similarity_score")
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def _qdrant_context_body(
        self,
        *,
        summary: str,
        search_mode: str,
        threshold: float,
        top_score: float | None,
        qualified_rows: list[dict[str, Any]],
    ) -> str:
        """Builds the retrieved-context body with full threshold-qualified payload rows."""
        header = [
            f"search_mode={search_mode}",
            f"score_threshold={threshold:.3f}",
            f"top_score={top_score if top_score is not None else 'n/a'}",
            f"qualified_rows={len(qualified_rows)} (top_k={self._qdrant_context_top_k})",
            "summary:",
            summary,
            "qualified_payload_rows:",
            json.dumps(qualified_rows, ensure_ascii=False, default=str),
        ]
        return "\n".join(header)

    @observe()
    async def _call_rewrite_agent(self, input: RewriteInput) -> RewriteOutput:
        return await self._rewrite.run_async(input)

    @observe()
    async def _call_answer_agent(self, input: AnswerInput) -> AnswerOutput:
        return await self._answer.run_async(input)

    @observe()
    async def _call_sufficient_agent(
        self, input: SufficiencyInput
    ) -> SufficiencyOutput:
        return await self._sufficiency.run_async(input)

    @observe()
    async def _call_selector_agent(
        self, input: SourceSelectionInput
    ) -> SourceSelectionOutput:
        return await self._source_selector.run_async(input)
