"""PostgreSQL graph executor contract and current stub implementation."""

from __future__ import annotations

from dataclasses import dataclass

from agent_core.schemas import GraphIntent, GraphResult


class GraphNotAvailableError(RuntimeError):
    """Raised when graph retrieval is requested before Postgres is configured."""

    pass


@dataclass
class PostgresGraphExecutor:
    """
    Stub executor until PostgreSQL is available.

    The source selector may request pg_graph steps; the orchestrator surfaces
    a warning and continues with available retrieval lanes.

    # TODO(postgres): Implement real graph primitives (neighbors/k_hop/path/compare)
    using bounded, parameterized SQL (recursive CTE) with hard caps on depth,
    fanout, and total results. Do not accept raw SQL from the LLM.

    # TODO(postgres): Add DB connectivity (async pool) via app settings, and
    unit/integration tests for traversal correctness + safety limits.
    """

    async def execute(self, intent: GraphIntent) -> GraphResult:
        """Executes a graph primitive for the given intent.

        Args:
            intent: Graph traversal/comparison instruction emitted by source selection.

        Raises:
            GraphNotAvailableError: Always, until the real Postgres implementation is added.
        """
        raise GraphNotAvailableError("PostgreSQL graph executor not configured yet.")
