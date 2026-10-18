"""Replace a text file's lines in one step, so a reader never sees half a file.

Files that a run rewrites whole, such as the review file and seed cleaning's
corrections file, are written this way:

- The lines go to a temporary file next to the target, which is written to
  disk and then renamed over the target. An interrupted run leaves the old file.
- An existing file keeps its permissions, since a person may have set them.
- The folder is written to disk after the rename, because the folder holds the
  file's new name; without it a power cut can undo the rename.
"""

import os
import shutil
from collections.abc import Iterable
from pathlib import Path


def replace_file_lines(path: str | Path, lines: Iterable[str]) -> None:
    """Write the lines in place of the file, in one step.

    Args:
        path: The file to replace; it and its folder are created if missing.
        lines: The lines to write, each ending with its line break.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as temporary_file:
            temporary_file.writelines(lines)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if path.exists():
            shutil.copymode(path, temporary_path)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    _sync_folder(path.parent)


def _sync_folder(folder: Path) -> None:
    """Write a folder's entries to disk, so a rename in it survives a power cut.

    Systems that cannot open a folder, such as Windows, are skipped.
    """
    if not hasattr(os, "O_DIRECTORY"):
        return
    folder_descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(folder_descriptor)
    finally:
        os.close(folder_descriptor)
