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
        ValueError: If path contains path traversal attempts
    """
    # Resolve to absolute path and normalize
    abs_path = Path(output_path).resolve()

    # Check for path traversal attempts
    try:
        abs_path.relative_to(Path.cwd())
    except ValueError:
        # Path is outside current directory - allow if it's a valid absolute path
        # but ensure it doesn't contain suspicious patterns
        if ".." in output_path or output_path.startswith("/"):
            raise ValueError(
                f"Potentially unsafe output path: {output_path}. "
                "Use relative paths within current directory or explicit absolute paths."
            )

    return str(abs_path)
