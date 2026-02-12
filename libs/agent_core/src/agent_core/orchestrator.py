"""Core orchestration loop for staged agentic-RAG search."""

from __future__ import annotations

import json
import logging
from typing import Any

from atomic_agents import AtomicAgent

from agent_core.context import RetrievedContextProvider
from agent_core.orchestration_inputs import (
    build_answer_stage_input,
    build_rewrite_stage_input,
    build_source_selection_stage_input,
    build_sufficiency_stage_input,
)
from agent_core.orchestration_validation import normalize_step_for_turn
from agent_core.retrieval import GraphNotAvailableError, PostgresGraphExecutor, QdrantExecutor
from agent_core.schemas import (
    AgentResponse,
    AnswerInput,
    AnswerOutput,
    GraphResult,
    RewriteInput,
    SourceSelectionInput,
    SourceSelectionOutput,
    RewriteOutput,
    RetrievalResult,
    SufficiencyInput,
    SufficiencyOutput,
)

from langfuse import observe

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

    @observe()
    async def run_search_ai(
        self,
        query: str,
        image_query: str | None = None,
        max_turns: int | None = None,
        request_id: str | None = None,
    ) -> AgentResponse:
        """Executes the staged rewrite/select/retrieve/answer/sufficiency loop.

        Args:
            query: User text query.
            image_query: Optional image query (data URL or raw base64 string).
            max_turns: Optional override for the loop turn cap.
            request_id: Optional request correlation id for stage-level logging.

        Returns:
            Final ``AgentResponse`` suitable for the gRPC layer.
        """
        retrieved = self._get_retrieved_provider()
        warnings: list[str] = []

        turns = max_turns or self._max_turns_default
        req = request_id or "n/a"
        logger.info(
            "agent.search_ai.start request_id=%s query=%r has_image=%s max_turns=%s",
            req,
            query[:160],
            bool(image_query),
            turns,
        )
        last_qdrant_result: RetrievalResult | None = None
        last_summary: str | None = None
        last_step: SourceSelectionOutput | None = None
        last_draft: AnswerOutput | None = None
        any_hits = False
        max_hit_count = 0
        attempted_actions: list[str] = []
        last_search_similarity_score: float | None = None

        rewrite: RewriteOutput = await self._call_rewrite_agent(
            build_rewrite_stage_input(query=query, image_query=image_query)
        )
        rewritten_query = (rewrite.rewritten_query or "").strip() or query
        retrieved.add_card(
            title="Rewrite",
            body=f"rewritten_query={rewritten_query}\nneeds_external_context={rewrite.needs_external_context}",
            data={
                "rewritten_query": rewritten_query,
                "needs_external_context": rewrite.needs_external_context,
            },
        )

        # Fast path: if rewrite says retrieval is not required, draft and validate directly.
        if not rewrite.needs_external_context:
            draft = await self._call_answer_agent(
                build_answer_stage_input(query=query, rewritten_query=rewritten_query)
            )
            last_draft = draft
            report: SufficiencyOutput = await self._call_sufficient_agent(
                build_sufficiency_stage_input(
                    query=query,
                    draft_answer=draft.answer,
                )
            )
            if report.sufficient:
                return self._finalize_from_draft(draft, warnings)

            warnings.append("Direct-answer path was insufficient; switching to retrieval loop.")


        for turn in range(1, turns + 1):
            step_input = build_source_selection_stage_input(
                query=query,
                rewritten_query=rewritten_query,
                image_query=image_query,
                turn=turn,
                last_step=last_step,
                last_summary=last_summary,
                warnings=warnings,
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
                assert step.search_intent is not None  # validated in normalize_step_for_turn
                attempted_actions.append("qdrant_search")
                
                result = await self._qdrant.search(step.search_intent, limit=self._qdrant_limit)
                last_qdrant_result = result
                last_summary = result.summary
                if result.count > 0:
                    any_hits = True
                    max_hit_count = max(max_hit_count, result.count)
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
                last_search_similarity_score = top_score
                qualified_rows: list[dict[str, Any]] = []
                for row in result.raw_data:
                    score = self._extract_similarity_score(row)
                    if score is not None and score >= score_threshold:
                        qualified_rows.append(row)
                    if len(qualified_rows) >= self._qdrant_context_top_k:
                        break
                top_hits = [
                    str(r.get("title") or r.get("name") or r.get("id") or "item")
                    for r in result.raw_data[:5]
                ]

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
                if top_score is None or top_score < score_threshold:
                    warnings.append(
                        (
                            f"Top similarity score {top_score if top_score is not None else 'n/a'} "
                            f"is below threshold {score_threshold:.3f} ({search_mode}); retrying."
                        )
                    )
                    continue
                if not qualified_rows:
                    warnings.append("No Qdrant rows met the score threshold; retrying.")
                    continue

            elif step.action == "pg_graph":
                assert step.graph_intent is not None  # validated in normalize_step_for_turn
                attempted_actions.append("pg_graph")
                
                try:
                    graph: GraphResult = await self._graph.execute(step.graph_intent)
                    last_summary = graph.summary
                    if graph.count > 0:
                        any_hits = True
                        max_hit_count = max(max_hit_count, graph.count)

                    retrieved.add_card(title="Postgres graph", body=graph.summary, data={"count": graph.count})
                except GraphNotAvailableError:
                    warnings.append("PostgreSQL graph executor not available yet; answering without graph traversal.")
                    logger.warning(
                        "agent.search_ai.pg_graph.unavailable request_id=%s turn=%d",
                        req,
                        turn,
                    )
                    last_summary = "Graph step skipped (not available)."
                    retrieved.add_card(title="Postgres graph", body="Not available.", data={})

            else:
                # Unreachable because ``normalize_step_for_turn`` gates action values.
                continue

            draft: AnswerOutput = await self._call_answer_agent(
                build_answer_stage_input(query=query, rewritten_query=rewritten_query)
            )
            last_draft = draft
            retrieved.add_card(
                title="Draft answer",
                body=(draft.answer or "")[:500],
                data={
                    "llm_confidence": draft.llm_confidence,
                    "result_entities": len(draft.result_entities),
                    "source_entities": len(draft.source_entities),
                },
            )

            suff: SufficiencyOutput = await self._call_sufficient_agent(
                build_sufficiency_stage_input(
                    query=query,
                    draft_answer=draft.answer,
                )
            )
            retrieved.add_card(
                title="Sufficiency",
                body=f"sufficient={suff.sufficient} missing={suff.missing}",
                data={"sufficient": suff.sufficient, "missing": suff.missing},
            )
           
            if suff.sufficient:
                return self._finalize_from_draft(
                    draft,
                    warnings,
                    search_similarity_score=last_search_similarity_score,
                )

        # Hard stop: use last draft as best effort if present.
        if last_draft is not None:
            best_effort = self._finalize_from_draft(
                last_draft,
                [*warnings, f"Reached max_turns={turns}."],
                search_similarity_score=last_search_similarity_score,
            )
            if not best_effort.result_entities and last_qdrant_result is not None:
                best_effort.result_entities = self._qdrant.extract_entity_refs(last_qdrant_result.raw_data)
            if not best_effort.source_entities:
                best_effort.source_entities = list(best_effort.result_entities)
            if not best_effort.warnings:
                best_effort.warnings = [f"Reached max_turns={turns}."]
            return best_effort

        # No-match is a business outcome, not a system error.
        evidence: dict[str, Any] = {"termination_reason": "no_match_after_max_turns"}
        if last_summary:
            evidence["last_summary"] = last_summary
        if last_search_similarity_score is not None:
            evidence["search_similarity_score"] = last_search_similarity_score

        answer = "I couldn’t find a confident match for this query yet."
        warning_code = "NO_MATCH_AFTER_MAX_TURNS"
        if any_hits:
            answer = "I found partial signals, but not enough confident evidence to answer this reliably yet."
            warning_code = "BEST_EFFORT_PARTIAL_AFTER_MAX_TURNS"

        return AgentResponse(
            answer=answer,
            source_entities=[],
            result_entities=[],
            evidence=evidence,
            warnings=[*warnings, warning_code],
        )

    def _get_retrieved_provider(self) -> RetrievedContextProvider:
        """Returns the dynamic retrieved-context provider from source selector state."""
        # Agents are built with the same context provider instance. Pull from one of them.
        provider = self._source_selector.get_context_provider("retrieved")
        if not isinstance(provider, RetrievedContextProvider):
            raise TypeError(
                "Expected RetrievedContextProvider for key 'retrieved', "
                f"got: {type(provider)!r}"
            )
        return provider
    
    @observe()
    async def _call_rewrite_agent(self, input):
        return await self._rewrite.run_async(input)

    @observe()
    async def _call_answer_agent(self, input):
        return await self._answer.run_async(input)

    @observe()
    async def _call_sufficient_agent(self, input):
        return await self._sufficiency.run_async(input)

    @observe()
    async def _call_selector_agent(self, input):
        return await self._source_selector.run_async(input)

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

    @staticmethod
    def _finalize_from_draft(
        draft: AnswerOutput,
        warnings: list[str],
        search_similarity_score: float | None = None,
    ) -> AgentResponse:
        """Converts a draft answer into final response format.

        Args:
            draft: Candidate answer draft from the answer stage.
            warnings: Orchestrator-level warnings to prepend.
            search_similarity_score: Optional top search similarity score from retrieval.

        Returns:
            Final ``AgentResponse`` preserving draft fields.
        """
        evidence = dict(draft.evidence)
        evidence["llm_confidence"] = max(0.0, min(1.0, float(draft.llm_confidence)))
        if search_similarity_score is not None:
            bounded = max(0.0, min(1.0, float(search_similarity_score)))
            evidence["search_similarity_score"] = bounded
        return AgentResponse(
            answer=draft.answer,
            source_entities=list(draft.source_entities),
            result_entities=list(draft.result_entities),
            evidence=evidence,
            warnings=[*warnings, *draft.warnings],
        )
