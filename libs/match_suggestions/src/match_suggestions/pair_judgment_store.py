"""Keep Jev's answers about page pairs, so a pair is asked once and read many times.

The store is reached through ``PairJudgmentStore``, so a database or blob store
can replace the JSON Lines file without changing its callers. Each item gives
the rule, then the reason for it.

- Answers are only added, never rewritten. A pair answered again gets a new
  line, and the latest line for a pair is the one read.
- A last line without its line end is left out. A batch stopped while writing
  leaves at most that line unfinished, and the pair is asked again.
- The file is read again when it changed since the last read, so the agent
  service sees answers a batch added without restarting.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from match_suggestions.model_pair_judgment import PairJudgment

# The answer file's name in seed cleaning's data folder, next to the review file.
PAIR_JUDGMENTS_FILE_NAME = "page_pair_judgments.jsonl"


class PairJudgmentStore(ABC):
    """Find and add Jev's answers about page pairs."""

    @abstractmethod
    def find_judgments(
        self, page_key_pairs: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], PairJudgment]:
        """Return the latest answers held for these pairs.

        Args:
            page_key_pairs: Pairs named by ``build_page_key_pair``.

        Returns:
            The latest answer of each pair the store holds, by pair.
        """

    @abstractmethod
    def add_judgments(self, pair_judgments: Iterable[PairJudgment]) -> None:
        """Keep these answers, each replacing any earlier answer of its pair.

        Args:
            pair_judgments: The answers to keep.
        """


class JsonlPairJudgmentStore(PairJudgmentStore):
    """Keep Jev's answers in a JSON Lines file, one answer per line."""

    def __init__(self, path: str | Path) -> None:
        """Open the store at a file, which need not exist yet.

        Args:
            path: The JSON Lines file holding the answers.
        """
        self._path = Path(path)
        self._judgments_by_pair: dict[tuple[str, str], PairJudgment] = {}
        self._read_file_version: tuple[int, int] | None = None

    def find_judgments(
        self, page_key_pairs: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], PairJudgment]:
        """Return the latest answers held for these pairs.

        Args:
            page_key_pairs: Pairs named by ``build_page_key_pair``.

        Returns:
            The latest answer of each pair the file holds, by pair.
        """
        self._read_changed_file()
        return {
            page_key_pair: self._judgments_by_pair[page_key_pair]
            for page_key_pair in page_key_pairs
            if page_key_pair in self._judgments_by_pair
        }

    def add_judgments(self, pair_judgments: Iterable[PairJudgment]) -> None:
        """Append these answers to the file.

        Args:
            pair_judgments: The answers to keep.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as store_file:
            store_file.writelines(
                f"{pair_judgment.model_dump_json()}\n"
                for pair_judgment in pair_judgments
            )

    def _read_changed_file(self) -> None:
        """Read the file again when its size or change time differs from the last read."""
        if not self._path.exists():
            return
        file_status = self._path.stat()
        file_version = (file_status.st_size, file_status.st_mtime_ns)
        if file_version == self._read_file_version:
            return
        store_text = self._path.read_text(encoding="utf-8")
        complete_lines = store_text.splitlines()[: store_text.count("\n")]
        for line in complete_lines:
            pair_judgment = PairJudgment.model_validate_json(line)
            self._judgments_by_pair[pair_judgment.page_key_pair] = pair_judgment
        self._read_file_version = file_version
