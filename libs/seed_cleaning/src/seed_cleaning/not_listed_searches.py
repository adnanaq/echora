"""Keep the AnimeSchedule searches that found nothing in one seed release.

AnimeSchedule is searched by an entry's MAL page (else AniList, else AniDB). A
search that finds nothing stores no page, so without a record every run would
search again: over a whole release that is about 13,000 searches and 11 hours.

Each item gives the rule, then the reason for it.

- A search that found nothing is not repeated within the same seed release.
  AnimeSchedule rarely adds an anime between two runs of one release.
- A new seed release starts with no record, so every search is made again
  once per release. The file keeps only the current release's searches.
- Each search is written as soon as it returns, so a stopped run resumes
  without repeating any.
- Discovery decides its own searches; this record belongs to seed cleaning.
"""

import json
from pathlib import Path

from work_matching import replace_file_lines


class NotListedSearchFile:
    """The searched-by page keys whose AnimeSchedule search found nothing in one release.

    Each line of the file reads ``{"release_name": ..., "searched_by": ...}``.
    """

    def __init__(self, path: str | Path, release_name: str) -> None:
        """Read the file's searches of this release, dropping other releases' lines.

        A partial last line, left by a write that was interrupted, is skipped
        and cut off when the file is rewritten.

        Args:
            path: The file; created on the first search added.
            release_name: The seed release being fetched, such as ``2026-43``.
        """
        self._path = Path(path)
        self._release_name = release_name
        self._searched_by_keys: set[str] = set()
        if not self._path.exists():
            return
        lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        complete_searches = [json.loads(line) for line in lines if line.endswith("\n")]
        self._searched_by_keys = {
            search["searched_by"]
            for search in complete_searches
            if search["release_name"] == release_name
        }
        if len(self._searched_by_keys) != len(lines):
            replace_file_lines(
                self._path,
                (self._format_line(key) for key in sorted(self._searched_by_keys)),
            )

    def __contains__(self, searched_by_key: object) -> bool:
        """Report whether this page's search found nothing in this release."""
        return searched_by_key in self._searched_by_keys

    def add(self, searched_by_key: str) -> None:
        """Record that the search by this page found nothing, and write it at once.

        Args:
            searched_by_key: The page key AnimeSchedule was searched by.
        """
        if searched_by_key in self._searched_by_keys:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as search_file:
            search_file.write(self._format_line(searched_by_key))
        self._searched_by_keys.add(searched_by_key)

    def _format_line(self, searched_by_key: str) -> str:
        """Return the file line recording one search of this release."""
        return (
            json.dumps(
                {"release_name": self._release_name, "searched_by": searched_by_key}
            )
            + "\n"
        )
