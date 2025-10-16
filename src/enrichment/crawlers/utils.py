"""
Common utility functions for crawler modules.

Provides shared functionality for path sanitization, validation, and other
common operations used across multiple crawler implementations.
"""

from pathlib import Path


def sanitize_output_path(output_path: str) -> str:
    """
    Sanitize output path to prevent path traversal attacks.

    Args:
        output_path: User-provided output file path

    Returns:
        Sanitized absolute path

    Raises:
        ValueError: If relative path escapes working directory
    """
    p = Path(output_path)
    abs_path = p.resolve()

    # Absolute paths are allowed as-is
    if p.is_absolute():
        return str(abs_path)

    # Relative paths must remain within CWD after resolution
    try:
        abs_path.relative_to(Path.cwd())
    except ValueError as err:
        raise ValueError(
            f"Output path escapes working directory: {output_path}"
        ) from err

    return str(abs_path)
