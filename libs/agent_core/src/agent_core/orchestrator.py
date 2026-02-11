"""Core orchestration loop for planner, executors, and sufficiency checks."""

from __future__ import annotations

import logging
from typing import Any

from atomic_agents import BasicChatInputSchema

from agent_core.context import RetrievedContextProvider
from agent_core.retrieval import GraphNotAvailableError, PostgresGraphExecutor, QdrantExecutor
from agent_core.schemas import AgentResponse, GraphResult, NextStep, RetrievalResult, SufficiencyReport

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Runs the bounded agent loop for ``SearchAI`` requests."""

    def __init__(
        self,
        planner,
        sufficiency,
        qdrant: QdrantExecutor,
        graph: PostgresGraphExecutor | None = None,
        *,
        max_turns_default: int = 4,
        qdrant_limit: int = 10,
    ) -> None:
        """Initializes orchestrator dependencies and safety limits.

        Args:
            planner: Atomic planner agent producing ``NextStep``.
            sufficiency: Atomic sufficiency agent producing ``SufficiencyReport``.
            qdrant: Qdrant retrieval executor.
            graph: Optional Postgres graph executor.
            max_turns_default: Default loop turn cap when request does not provide one.
            qdrant_limit: Max Qdrant hits per retrieval step.
        """
        self._planner = planner
        self._sufficiency = sufficiency
        self._qdrant = qdrant
        self._graph = graph or PostgresGraphExecutor()
        self._max_turns_default = max_turns_default
        self._qdrant_limit = qdrant_limit

    async def run_search_ai(
        self,
        query: str,
        image_query: str | None = None,
        max_turns: int | None = None,
    ) -> AgentResponse:
        """Executes the planner/executor/sufficiency loop for one query.

        Args:
            query: User text query.
            image_query: Optional image query (URL, data URL, or base64).
            max_turns: Optional override for the loop turn cap.

        Returns:
            Final ``AgentResponse`` suitable for the gRPC layer.
        """
        retrieved = self._get_retrieved_provider()
        warnings: list[str] = []

        turns = max_turns or self._max_turns_default
        last_qdrant_result: RetrievalResult | None = None
        last_summary: str | None = None
        last_step: NextStep | None = None
        any_hits = False
        max_hit_count = 0
        attempted_actions: list[str] = []

        for turn in range(1, turns + 1):
            prompt = self._planner_prompt(
                query=query,
                image_query=image_query,
                turn=turn,
                last_step=last_step,
                last_summary=last_summary,
                warnings=warnings,
            )
            step: NextStep = await self._planner.run_async(BasicChatInputSchema(chat_message=prompt))
            last_step = step

            if step.action == "final":
                if step.final is None:
                    warnings.append("Planner emitted final without a final AgentResponse; continuing.")
                    continue

                # Guard: planner is not allowed to finalize unless sufficiency is true.
                suff: SufficiencyReport = await self._sufficiency.run_async(
                    BasicChatInputSchema(
                        chat_message=f"User query: {query}\n\nDecide if Retrieved Context is sufficient to answer."
                    )
                )
                retrieved.add_card(
                    title="Sufficiency (guard)",
                    body=f"sufficient={suff.sufficient} missing={suff.missing}",
                    data={"sufficient": suff.sufficient, "missing": suff.missing},
                )
                if suff.sufficient:
                    step.final.warnings = [*warnings, *step.final.warnings]
                    return step.final

                warnings.append("Planner attempted to finalize before sufficiency was true; continuing.")
                continue

            if step.action == "qdrant_search":
                if step.search_intent is None:
                    warnings.append("Planner emitted qdrant_search without search_intent; skipping.")
                    continue
                attempted_actions.append("qdrant_search")
                # If an image was provided at request level, make it available to the executor even if
                # the planner forgot to copy it into the intent. We avoid overriding id-lookups.
                if (
                    image_query
                    and not step.search_intent.image_query
                    and not (isinstance(step.search_intent.filters.get("id"), list) and step.search_intent.filters.get("id"))
                ):
                    step.search_intent.image_query = image_query
                if query and step.search_intent.query is None and not step.search_intent.filters.get("id"):
                    step.search_intent.query = query
                result = await self._qdrant.search(step.search_intent, limit=self._qdrant_limit)
                last_qdrant_result = result
                last_summary = result.summary
                if result.count > 0:
                    any_hits = True
                    max_hit_count = max(max_hit_count, result.count)
                retrieved.add_card(
                    title=f"Qdrant search ({step.search_intent.entity_type.value})",
                    body=result.summary,
                    data={"count": result.count},
                )

            elif step.action == "pg_graph":
                if step.graph_intent is None:
                    warnings.append("Planner emitted pg_graph without graph_intent; skipping.")
                    continue
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
                    last_summary = "Graph step skipped (not available)."
                    retrieved.add_card(title="Postgres graph", body="Not available.", data={})
            else:
                warnings.append(f"Unknown action '{step.action}'; skipping.")
                continue

            suff: SufficiencyReport = await self._sufficiency.run_async(
                BasicChatInputSchema(
                    chat_message=f"User query: {query}\n\nDecide if Retrieved Context is sufficient to answer."
                )
            )
            retrieved.add_card(
                title="Sufficiency",
                body=f"sufficient={suff.sufficient} missing={suff.missing}",
                data={"sufficient": suff.sufficient, "missing": suff.missing},
            )
            if suff.sufficient:
                # Ask planner to finalize with current context.
                final_step: NextStep = await self._planner.run_async(
                    BasicChatInputSchema(
                        chat_message=f"User query: {query}\n\nFinalize the answer now using Retrieved Context. Include best result_entities/source_entities you used."
                    )
                )
                if final_step.action == "final" and final_step.final is not None:
                    final_step.final.warnings = [*warnings, *final_step.final.warnings]
                    return final_step.final

                warnings.append("Sufficiency was true but planner did not finalize; continuing.")

        # Hard stop: ask the planner to finalize with current context.
        final_step: NextStep = await self._planner.run_async(
            BasicChatInputSchema(
                chat_message=f"User query: {query}\n\nWe reached the turn limit ({turns}). Finalize the best-effort answer now using Retrieved Context. Be explicit about missing data."
            )
        )
        if final_step.action == "final" and final_step.final is not None:
            final_step.final.warnings = [*warnings, *final_step.final.warnings, f"Reached max_turns={turns}."]
            # Best-effort: if planner didn't provide result entities, we can at least attach the last qdrant refs.
            if not final_step.final.result_entities and last_qdrant_result is not None:
                final_step.final.result_entities = self._qdrant.extract_entity_refs(last_qdrant_result.raw_data)
            return final_step.final

        # No-match is a business outcome, not a system error.
        evidence: dict[str, Any] = {
            "termination_reason": "no_match_after_max_turns",
            "max_turns": turns,
            "attempted_actions": attempted_actions,
            "max_hit_count": max_hit_count,
        }
        if last_summary:
            evidence["last_summary"] = last_summary

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
            confidence=0.1,
        )

    def _get_retrieved_provider(self) -> RetrievedContextProvider:
        """Returns the dynamic retrieved-context provider from planner state."""
        # The agents were built with a context provider instance; pull it from planner.
        # This relies on the planner using SystemPromptGenerator(context_providers={"retrieved": provider}).
        return self._planner.get_context_provider("retrieved")

    @staticmethod
    def _planner_prompt(
        *,
        query: str,
        image_query: str | None,
        turn: int,
        last_step: NextStep | None,
        last_summary: str | None,
        warnings: list[str],
    ) -> str:
        """Builds the planner input prompt for the current turn.

        Args:
            query: User text query.
            image_query: Optional image query.
            turn: Current loop turn number.
            last_step: Previous planner step, if any.
            last_summary: Previous retrieval summary, if any.
            warnings: Accumulated warning messages.

        Returns:
            Prompt text for the next planner invocation.
        """
        parts = [f"User query: {query}", f"Turn: {turn}"]
        if image_query:
            # Avoid dumping base64 into the prompt.
            if image_query.startswith("http://") or image_query.startswith("https://"):
                parts.append(f"User provided an image URL: {image_query}")
            else:
                parts.append("User provided an image (base64/data-url omitted).")
        if warnings:
            parts.append("Warnings so far:\n- " + "\n- ".join(warnings[-5:]))
        if last_step is not None:
            parts.append(f"Last action: {last_step.action}")
        if last_summary:
            parts.append("Last retrieval summary:\n" + last_summary)
        parts.append("Decide the next action now.")
        return "\n\n".join(parts)
