# Agentic RAG System Technical Specification

**Date:** January 24, 2026
**Framework:** Atomic Agents + Qdrant
**Pattern:** Self-Reflective Feedback Loop (Agentic RAG)

---

## 1. Architectural Overview

This system implements a **Self-Correcting RAG Loop** with explicit, code-owned control flow. It functions as a state machine that transitions based on the *sufficiency* of retrieved data, rather than a linear pipeline.

### Service Boundaries (Current Plan)

- **Backend/BFF (GraphQL):** The true backend that serves the frontend. It:
  - exposes GraphQL endpoints to the frontend
  - routes direct-search intents to a fast Postgres lane (`search`) for typeahead/entity lookup
  - routes natural-language intents to the agent lane (`search_ai`) and calls internal gRPC `SearchAI`
  - hydrates final results from PostgreSQL using batched loaders (avoid N+1)
- **Agent Service (internal):** The orchestrator for natural-language queries. It:
  - runs a **Planner Agent** to decide the next step (Qdrant / Postgres graph / finalize)
  - runs a **Sufficiency Agent** after each retrieval step to avoid premature finalization
  - executes retrieval against Qdrant/PostgreSQL in application code (async, bounded, audited)
  - returns mixed entity IDs + evidence; backend hydrates full objects from PostgreSQL
- **PostgreSQL:** Source of truth for entities + relationships (normalized + JSONB for long-tail fields).
- **Qdrant:** Semantic index for similarity search and candidate ID retrieval.

### The Feedback Loop (Planner/Executor + Sufficiency)
This design intentionally does **not** depend on a framework-specific “tool_calls” runtime. Instead, Atomic Agents produces typed decisions; our code executes retrieval.

1. **Plan:** `PlannerAgent` returns a `NextStep` (bounded action + structured intent).
2. **Execute:** Orchestrator executes one step:
   - `qdrant_search` via Qdrant client (semantic/entity resolution)
   - `pg_graph` via PostgreSQL graph primitives (recursive CTE / fixed DSL)
3. **Observe:** Orchestrator stores `RetrievalResult` / `GraphResult` cards into `RetrievedContextProvider`.
4. **Reflect:** `SufficiencyAgent` returns `SufficiencyReport`.
   - If insufficient: loop back to **Plan**
   - If sufficient: ask `PlannerAgent` to finalize (`NextStep.action="final"`) and return `AgentResponse`

Assumption (current): the sufficiency agent runs **after every retrieval step** for correctness. We can optimize later.

---

## 1.1 Routing Lanes: `search` vs `search_ai`

We support two retrieval lanes:

### A) `search` lane (Fast, Direct PostgreSQL)

Purpose: deterministic entity lookup (typeahead) across entity types.

- Input: simple string query + optional type filter.
- Implementation: Postgres-only, indexed, no agent.
- Ranking: popularity-first (where available), then name.

This endpoint should be fast and predictable and must not attempt relationship reasoning.

### B) `search_ai` lane (Natural Language, Agent Orchestrator)

Purpose: any natural-language question, including relationships, comparisons, recommendations, and explanations.

- Input: free-form query (+ optional `image_query`).
- Output: answer + mixed entity IDs + evidence (paths / comparisons / ranked IDs).
- Backend hydrates final objects from Postgres via GraphQL batched loaders (DataLoader pattern).

Important rule: the agent must not generate raw SQL. It can only call fixed query primitives or a validated query DSL compiled to SQL with bound parameters.

## 2. Schema Definitions (The Protocol)

These Pydantic models define the interface between the agents, the orchestrator, and the retrieval executors.

### A. Search Intent (Planner Agent $\rightarrow$ Orchestrator)
Defines *what* the agent wants to find and *why*.

```python
from pydantic import Field
from atomic_agents import BaseIOSchema
from typing import Literal, Dict, Any, Optional

class SearchIntent(BaseIOSchema):
    """
    Structured instruction for the retrieval executor.
    The agent tweaks this in every loop iteration based on previous findings.
    """
    rationale: str = Field(
        ...,
        description=(
            "Short operational reason for this step (for logs/telemetry). "
            "Do NOT include chain-of-thought."
        ),
    )
    entity_type: Literal["anime", "character", "episode", "manga"] = Field(
        ..., 
        description="The specific collection to search in (matches EntityType Enum)."
    )
    query: Optional[str] = Field(
        None, 
        description="Semantic search text. Required for content searching. Can be None if strictly filtering by ID."
    )
    image_query: Optional[str] = Field(
        None,
        description="URL or base64 of an image. If provided, triggers visual similarity search (Hybrid if 'query' is also present, or Image-Only)."
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Qdrant payload filters. "
            "Examples: "
            "{ 'id': ['uuid-1'] } -> Fetch specific item. "
            "{ 'anime_id': 'uuid-2' } -> Filter items by parent. "
            "{ 'year': 2024 } -> Metadata filter."
        )
    )
```

### A2. Graph Intent (Planner Agent $\rightarrow$ Orchestrator)

Defines relationship/path/comparison operations against PostgreSQL as a bounded set of supported primitives.

```python
from pydantic import Field
from atomic_agents import BaseIOSchema
from typing import Any, Dict, Literal, Optional

class EntityRef(BaseIOSchema):
    entity_type: Literal["anime", "character", "episode", "manga"] = Field(...)
    id: str = Field(..., description="Canonical UUID string")


class GraphIntent(BaseIOSchema):
    rationale: str = Field(
        ...,
        description="Short operational reason for this traversal/comparison (for logs).",
    )
    query_type: Literal["neighbors", "k_hop", "path", "compare"] = Field(
        ..., description="Which graph primitive to execute"
    )

    start: EntityRef = Field(..., description="Start node for traversal")
    end: Optional[EntityRef] = Field(
        None, description="Optional end node (required for query_type='path')"
    )

    max_hops: int = Field(
        default=3,
        description="Max traversal depth. Must be hard-capped to prevent graph explosion.",
    )
    edge_types: list[str] = Field(
        default_factory=list,
        description="Allowlist of edge types/relations to traverse (bounded vocabulary).",
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional constraints (year ranges, relation_type allowlist, etc.)",
    )
    limits: Dict[str, int] = Field(
        default_factory=lambda: {"max_results": 50, "max_fanout_per_hop": 50},
        description="Safety limits to prevent explosion.",
    )
```

### A3. Next Step (Planner Agent $\rightarrow$ Orchestrator)

The planner does not execute retrieval. It outputs a single bounded action per turn.

```python
from pydantic import Field
from atomic_agents import BaseIOSchema
from typing import Literal, Optional

class NextStep(BaseIOSchema):
    action: Literal["qdrant_search", "pg_graph", "final"] = Field(
        ..., description="Single next action the orchestrator must execute."
    )
    rationale: str = Field(
        ..., description="Short operational reason for this step (for logs)."
    )

    search_intent: Optional[SearchIntent] = Field(
        None, description="Required when action='qdrant_search'."
    )
    graph_intent: Optional[GraphIntent] = Field(
        None, description="Required when action='pg_graph'."
    )
    final: Optional["AgentResponse"] = Field(
        None, description="Required when action='final'."
    )
```

### B. Retrieval Result (Executor $\rightarrow$ Context)
Standardizes the output for the Agent's consumption.

```python
class RetrievalResult(BaseIOSchema):
    """
    The raw knowledge retrieved from the database.
    """
    summary: str = Field(
        ..., 
        description="Human-readable summary of results (Title, Type, Key Description) for the LLM to read."
    )
    raw_data: list[Dict] = Field(
        ..., 
        description="Full structured data. Essential for extracting UUIDs for follow-up searches."
    )
    count: int = Field(..., description="Number of items found.")
```

### B2. Graph Result (Executor $\rightarrow$ Context)

```python
from pydantic import Field
from atomic_agents import BaseIOSchema
from typing import Any, Dict, List, Optional

class PathEdge(BaseIOSchema):
    rel: str = Field(..., description="Relationship label/type")


class PathNode(BaseIOSchema):
    entity: EntityRef


class GraphPath(BaseIOSchema):
    nodes: List[EntityRef] = Field(..., description="Ordered node IDs in the path")
    rels: List[str] = Field(..., description="Ordered relationship types between nodes")


class GraphResult(BaseIOSchema):
    summary: str = Field(..., description="Human-readable summary of traversal output")
    paths: List[GraphPath] = Field(default_factory=list)
    node_ids: List[EntityRef] = Field(
        default_factory=list, description="All unique nodes referenced by result"
    )
    count: int = Field(..., description="Number of paths or nodes returned")
    meta: Dict[str, Any] = Field(default_factory=dict)
```

### B3. Sufficiency Report (Sufficiency Agent $\rightarrow$ Orchestrator)

The sufficiency agent decides whether we have enough context to produce a final answer. It may suggest the next step,
but the planner remains the source of truth for the actual `NextStep` emitted.

```python
from pydantic import Field
from atomic_agents import BaseIOSchema
from typing import Literal, Optional

class SufficiencyReport(BaseIOSchema):
    sufficient: bool = Field(..., description="True if current context is enough to answer.")
    rationale: str = Field(..., description="Short operational reason (no chain-of-thought).")
    missing: list[str] = Field(
        default_factory=list,
        description="What is missing (human readable; for logs/debug/UI).",
    )
    suggested_next_action: Optional[Literal["qdrant_search", "pg_graph"]] = Field(
        None,
        description="Optional hint; orchestrator may ignore. Planner decides the real next step.",
    )
```

### C. Final Response (Agent $\rightarrow$ User)
The final answer once the loop is resolved.

```python
class AgentResponse(BaseIOSchema):
    """
    The final answer presented to the user.
    """
    answer: str = Field(..., description="Natural language answer.")
    source_entities: list[EntityRef] = Field(
        default_factory=list,
        description="Entity references used to generate the answer. The backend uses these to fetch full records from PostgreSQL."
    )
    result_entities: list[EntityRef] = Field(
        default_factory=list,
        description="Primary result entities to display (may be a subset of source_entities).",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured evidence for the backend/UI (paths, diffs, ranked IDs).",
    )
    citations: list[str] = Field(default_factory=list, description="List of source titles/items used.")
    confidence: float = Field(..., description="0.0 to 1.0 confidence score.")
```

---

## 3. Retrieval Executors (Orchestrator-Owned)

Atomic Agents provides the schema-driven decision layer. The orchestrator owns execution against Qdrant/Postgres.
This is how we enforce:

- no raw SQL authored by the LLM
- strict limits/timeouts
- audited/observable retrieval behavior

### 3.1 Qdrant Retrieval (Semantic/Entity Resolution)

Role: semantic similarity + fuzzy entity resolution; return candidate canonical UUIDs.

Execution rules:

- If `filters.id` is present: do an ID retrieve (no embedding required).
- Otherwise: embed `query` (and/or `image_query`) and perform a vector search.

Orchestrator function signature (conceptual):

```python
async def execute_qdrant_search(intent: SearchIntent) -> RetrievalResult:
    ...
```

### 3.2 Postgres Graph Retrieval (Authoritative Relationships)

Role: relationship traversal, explanation paths, and deterministic comparisons using PostgreSQL.

Supported operations are bounded primitives (no raw SQL authored by the agent):

- `neighbors`: one-hop adjacency expansion
- `k_hop`: k-hop expansion with strict limits (depth + fanout)
- `path`: find an explanation path between two nodes (recursive CTE)
- `compare`: compute structured diffs between two entities (SQL, not embeddings)

Implementation notes:

- Use fixed, parameterized SQL templates and/or a validated query DSL compiled to SQLAlchemy Core.
- Enforce hard caps: `max_hops`, `max_results`, `max_fanout_per_hop`, and statement timeouts.

### 3.3 PostgreSQL Plug-In Contract (Implement Now, Wire Values Later)

The goal is to keep orchestrator/agent flow stable while Postgres is brought online. `PostgresGraphExecutor.execute(intent)` remains the only integration seam; all future DB wiring stays inside that executor.

Required wiring inputs (service config):

- `postgres_dsn` (or host/port/db/user/password)
- `pool_min_size`, `pool_max_size`
- `statement_timeout_ms`
- `query_timeout_ms` (client-side)

Required schema/value mappings:

- Canonical entity tables + IDs:
  - `anime(id uuid pk, ...)`
  - `character(id uuid pk, ...)`
  - `episode(id uuid pk, anime_id uuid fk, ...)`
  - `manga(id uuid pk, ...)`
- Relationship tables used by graph primitives:
  - `anime_related_anime(anime_id, related_anime_id, relation_type, ...)`
  - `anime_related_manga(anime_id, manga_id, relation_type, ...)`
  - `anime_character(anime_id, character_id, role, ...)`
  - `character_relationship(character_id, related_character_id, relation_type, ...)` (optional but recommended)
- Allowlisted edge vocabulary (must be explicit, no free-form SQL labels):
  - `shares_franchise`, `prequel`, `sequel`, `spinoff`, `same_studio`, `shared_staff`, `co_appears`, `adapted_from`, etc.

Primitive contract (no orchestrator changes required later):

- `neighbors(intent)`:
  - input: `start`, `edge_types`, `limits.max_results`
  - output: one-hop nodes in `GraphResult.node_ids`
- `k_hop(intent)`:
  - input: `start`, `max_hops`, `edge_types`, `limits.max_fanout_per_hop`
  - output: bounded traversal frontier + optional sampled paths
- `path(intent)`:
  - input: `start`, `end`, `max_hops`, `edge_types`
  - output: one or more explanation chains in `GraphResult.paths`
- `compare(intent)`:
  - input: `start`, `end`, optional filters
  - output: structured diffs in `GraphResult.meta` and summary text

Safety invariants (must remain true):

- LLM never emits raw SQL.
- SQL is parameterized/bound only.
- Hard caps always enforced server-side even if planner suggests larger values.
- Timeouts and cancellation are applied per query.
- No-match from graph traversal is a valid business outcome, not an internal error.

## 4. Context Management: `RetrievedContextProvider` (Atomic Agents v2+)

**Role:** Maintains the "Working Memory" of the current investigation. This prevents the agent from forgetting what it found in Loop 1 when it moves to Loop 2.

### Behavior
*   **State:** Maintains an ordered list of context "cards" (summaries + key IDs).
*   **Update:** Appends new findings as new cards.
*   **Injection:** Returns a formatted string (e.g., `--- RETRIEVED CONTEXT --- ...`) to the `SystemPromptGenerator`.

---

## 5. Agent Strategy & Prompting

We use two Atomic Agents:

- **PlannerAgent**: outputs `NextStep` only (bounded action + intent). No direct DB access.
- **SufficiencyAgent**: outputs `SufficiencyReport` only (is context sufficient?).

Both are configured using `SystemPromptGenerator` + `RetrievedContextProvider` to ensure they see the current context.

### A. Background
*   You are an expert anime research agent specializing in deep metadata retrieval.
*   You do NOT have direct access to databases; you only output structured decisions.
*   Your goal is to provide technically accurate answers based *only* on retrieved data.

### B. Steps (The Cognitive Loop)
1.  **Analyze Context:** Read the `Active Investigation Context` provided by your context providers.
2.  **Plan Next Step:** Output exactly one `NextStep` action.
3.  **Prefer Small Steps:** Resolve entity IDs first (Qdrant), then traverse relationships (Postgres), then finalize.
4.  **Sufficiency-Aware Finalization:** If the latest sufficiency signal indicates the context is sufficient, you MUST output `NextStep(action="final")`.
5.  **Zero Results Protocol:**
    - If a step returns **0 results**, broaden or change approach on next step.
    - After N failed steps (e.g., 3), finalize with an explicit “not found” response.
6.  **Finalize:** Only finalize when `SufficiencyAgent` deems context sufficient, or when loop limits are reached.

### C. Output Instructions
*   **Accuracy:** Only answer based on the provided context. If data is missing after 3 attempts, state what is missing.
*   **Citations:** Mention the titles/names of the entities you are referencing.
*   **UUID Extraction:** Always include canonical UUIDs as `EntityRef` entries in `source_entities` and `result_entities`.

---

## 6. Implementation Roadmap

| Step | Component | File Path | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Schema** | `libs/agent_core/src/agent_core/schemas.py` | Define `SearchIntent`, `GraphIntent`, `NextStep`, `SufficiencyReport`, `AgentResponse`. |
| **2** | **Context** | `libs/agent_core/src/agent_core/context/retrieved_context.py` | Implement `RetrievedContextProvider` (`BaseDynamicContextProvider`). |
| **3** | **Retrieval** | `libs/agent_core/src/agent_core/retrieval/qdrant_executor.py` | Implement `execute_qdrant_search(intent) -> RetrievalResult` (async). |
| **4** | **Retrieval** | `libs/agent_core/src/agent_core/retrieval/postgres_graph_executor.py` | Implement bounded graph primitives against Postgres (async), preserving the existing `execute(intent)` contract so orchestrator does not change. |
| **5** | **Agent** | `libs/agent_core/src/agent_core/agents/planner.py` | Planner agent: Input = query + context, Output = `NextStep`. |
| **6** | **Agent** | `libs/agent_core/src/agent_core/agents/sufficiency.py` | Sufficiency agent: Input = query + context, Output = `SufficiencyReport`. |
| **7** | **Orchestrator** | `libs/agent_core/src/agent_core/orchestrator.py` | Glue code: loop, execute steps, update context, enforce caps/timeouts. |
| **8** | **Service** | `apps/agent_service/src/agent_service/server.py` | Expose internal gRPC `SearchAI` RPC to backend/BFF. |

---

## 6.1 Observability, Retries, and Safety (Production Requirements)

We use Atomic Agents (Instructor) hooks and application-level metrics to make the loop debuggable and safe.

- **Correlation IDs:** assign a `request_id` per `SearchAI` RPC; include it in all logs.
- **Step Logging:** log every `NextStep` and `SufficiencyReport` (structured JSON). Do not log private chain-of-thought.
- **Latency Metrics:** track per-step timing (`planner_llm_ms`, `qdrant_ms`, `pg_graph_ms`, `sufficiency_llm_ms`) and total time.
- **Hard Caps:** enforce `max_turns`, `max_hops`, `max_results`, `max_fanout_per_hop`, and statement timeouts for Postgres.
- **Retry Policy:**
  - LLM schema validation errors (invalid `NextStep` / `SufficiencyReport`): retry up to N times with an explicit correction prompt.
  - Qdrant/Postgres transient failures: limited retries with backoff; surface partial results if budget is exhausted.
- **Deterministic Safety:** the orchestrator must validate `NextStep` consistency:
  - `action="qdrant_search"` requires `search_intent`
  - `action="pg_graph"` requires `graph_intent`
  - `action="final"` requires `final`

## 7. Reference Implementations

These code blocks provide a concrete starting point for development.

### A. RetrievedContextProvider (v2+)
```python
from atomic_agents.context import BaseDynamicContextProvider

class RetrievedContextProvider(BaseDynamicContextProvider):
    def __init__(self, title: str = "Active Investigation Context"):
        super().__init__(title=title)
        self.cards: list[str] = []

    def add_card(self, card: str) -> None:
        self.cards.append(card)

    def get_info(self) -> str:
        if not self.cards:
            return "No information has been retrieved from the database yet."
        return "\n\n---\n\n".join(self.cards)
```

### B. Orchestrator Loop (Planner/Executor + Sufficiency)
```python
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

class PlannerInput(BaseIOSchema):
    user_query: str

class SufficiencyInput(BaseIOSchema):
    user_query: str

async def run_search_ai(user_query: str, max_turns: int = 6) -> AgentResponse:
    context = RetrievedContextProvider()
    prompt = SystemPromptGenerator(context_providers=[context])

    planner = AtomicAgent[PlannerInput, NextStep](
        config=AgentConfig(system_prompt_generator=prompt)
    )
    sufficiency = AtomicAgent[SufficiencyInput, SufficiencyReport](
        config=AgentConfig(system_prompt_generator=prompt)
    )

    for _ in range(max_turns):
        step = await planner.run_async(PlannerInput(user_query=user_query))

        if step.action == "final":
            # If the planner tries to finalize early, we still run sufficiency as a guard.
            report = await sufficiency.run_async(SufficiencyInput(user_query=user_query))
            if report.sufficient and step.final is not None:
                return step.final
            # Not sufficient -> continue loop (planner will be called again with same context)
            context.add_card(f"SUFFICIENCY: insufficient ({report.missing})")
            continue

        if step.action == "qdrant_search":
            result = await execute_qdrant_search(step.search_intent)  # type: ignore[arg-type]
            context.add_card(f"QDRANT RESULT:\n{result.summary}")

        elif step.action == "pg_graph":
            graph = await execute_postgres_graph(step.graph_intent)  # type: ignore[arg-type]
            context.add_card(f"POSTGRES GRAPH:\n{graph.summary}")

        # After every retrieval, check sufficiency.
        report = await sufficiency.run_async(SufficiencyInput(user_query=user_query))
        context.add_card(f"SUFFICIENCY: {report.sufficient} missing={report.missing}")

        if report.sufficient:
            final_step = await planner.run_async(
                PlannerInput(user_query=user_query)
            )
            if final_step.action == "final" and final_step.final is not None:
                return final_step.final

    # Safety fallback: ask the planner to finalize with what it has.
    final_step = await planner.run_async(PlannerInput(user_query=user_query))
    if final_step.action == "final" and final_step.final is not None:
        return final_step.final
    # No-match is a business outcome, not a system error.
    return AgentResponse(
        answer="I couldn’t find a confident match for this query yet.",
        source_entities=[],
        result_entities=[],
        evidence={"termination_reason": "no_match_after_max_turns"},
        citations=[],
        confidence=0.1,
        warnings=["NO_MATCH_AFTER_MAX_TURNS"],
    )
```

---

## 8. Example Trace: "Which anime is Luffy in?"

1. **Turn 1 (Plan):** Planner emits `NextStep(action="qdrant_search", entity_type="character", query="Luffy")`.
2. **Turn 1 (Execute):** Orchestrator queries Qdrant; stores result card with canonical character UUID.
3. **Turn 1 (Reflect):** Sufficiency is false: missing anime names.
4. **Turn 2 (Plan):** Planner emits `NextStep(action="qdrant_search", entity_type="anime", filters={"id":[...]})`.
5. **Turn 2 (Execute):** Orchestrator retrieves anime rows from Qdrant by UUID; stores result card with titles.
6. **Turn 2 (Reflect):** Sufficiency becomes true.
7. **Finalize:** Planner emits `NextStep(action="final", final=AgentResponse(...))`.

```

## 9. Relationship Query Trace (Postgres Graph)

Example: "How is Luffy related to Nami?"

1. **Resolve entities:** Planner requests two Qdrant lookups (Luffy, Nami) to obtain canonical UUIDs.
2. **Traverse:** Planner requests `pg_graph` with `query_type="path"`:
   - `start={entity_type:"character", id:"<luffy_uuid>"}`
   - `end={entity_type:"character", id:"<nami_uuid>"}`
   - `max_hops=3`
   - `edge_types=[...]` (bounded allowlist)
3. **Final Response:** Planner returns:
   - `answer`: step-by-step chain explanation
   - `source_entities`: all node IDs in the returned path
   - `result_entities`: the two characters
   - `evidence.paths`: the structured node/edge chain

The backend/BFF hydrates the final objects from PostgreSQL using batched GraphQL loaders.

---

## 10. Design Considerations & Optimization Opportunities

### 10.1 Latency vs. Correctness
The current loop executes the `SufficiencyAgent` after every retrieval step. While this ensures high quality, it doubles the LLM overhead.
- **Optimization:** Consider a "Fast-Path" where the `PlannerAgent` can self-signal completion, using the `SufficiencyAgent` only for complex queries or after a minimum turn threshold.

### 10.2 Structured Metadata Filtering
The `SearchIntent.filters` is currently a generic dictionary.
- **Optimization:** If the schema remains stable, transition to a structured `QdrantFilters` Pydantic model. This provides the LLM with better "hints" about available metadata fields (e.g., `year`, `genre`, `score_threshold`), improving filter accuracy.

### 10.3 Explicit Failure Rationale
If a search step returns zero results, the planner may get stuck or repeat the same query.
- **Optimization:** Update `PlannerInput` to include a `last_step_status` field. If the previous turn failed, provide the agent with a `RetryRationale` (e.g., "The previous search for 'Luffy' in 'manga' returned 0 results. Suggest broadening search to 'anime' or checking ID.")

### 10.4 Response Handshake
Currently, if `SufficiencyAgent` deems context sufficient, the `PlannerAgent` is called again to format the `AgentResponse`.
- **Optimization:** For simple queries, allow the `SufficiencyAgent` to trigger a lightweight `ResponseAgent` that specializes in synthesis, bypassing the complex Planner logic for the final turn.

### 10.5 Resource Safety
Graph traversals in PostgreSQL can be expensive.
- **Enforcement:** Always accompany `max_hops` with a strict database-level `statement_timeout` (e.g., 2 seconds) and a `max_fanout` limit to prevent the orchestrator from stalling on deep relationship clusters.
