"""Retrieved-context provider used as dynamic working memory for agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atomic_agents.context import BaseDynamicContextProvider


@dataclass(frozen=True)
class _Card:
    """Single retrieved context entry injected into prompts."""

    title: str
    body: str
    data: dict[str, Any]


class RetrievedContextProvider(BaseDynamicContextProvider):
    """
    Dynamic context provider that exposes retrieval "cards" to staged agents.

    Keep this compact: it is injected into the system prompt every turn.
    """

    def __init__(self, title: str = "Retrieved Context", max_cards: int = 12) -> None:
        """Initializes the context provider with a bounded card buffer.

        Args:
            title: Context title shown in generated system prompts.
            max_cards: Maximum number of recent cards retained.
        """
        super().__init__(title=title)
        self._max_cards = max_cards
        self._cards: list[_Card] = []

    def add_card(self, title: str, body: str, data: dict[str, Any] | None = None) -> None:
        """Appends a retrieval card and trims history to the configured max.

        Args:
            title: Short label for the card.
            body: Human-readable summary text.
            data: Optional structured metadata for debugging/inspection.
        """
        if data is None:
            data = {}
        self._cards.append(_Card(title=title, body=body, data=data))
        if len(self._cards) > self._max_cards:
            self._cards = self._cards[-self._max_cards :]

    def get_info(self) -> str:
        """Formats recent retrieval cards for prompt injection.

        Returns:
            Concatenated card text ordered by insertion time.
        """
        if not self._cards:
            return "No retrievals yet."

        parts: list[str] = []
        for card_index, card in enumerate(self._cards[-self._max_cards :], start=1):
            parts.append(f"[{card_index}] {card.title}\n{card.body}".strip())
        return "\n\n".join(parts)
