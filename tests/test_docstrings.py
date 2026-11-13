"""Check docstring sections, subject naming, and file and function length across all source files."""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SOURCE_ROOTS = (REPO_ROOT / "apps", REPO_ROOT / "libs")
BARE_PRONOUN_START = re.compile(r"(?:^|[.!?]\s+)(It|They|Its|Their)\s")
SECTION_HEADINGS = ("Args:", "Returns:", "Raises:", "Yields:")
MAX_SOURCE_FILE_LINES = 500
MAX_FUNCTION_LINES = 50


def _list_docstrings() -> list[tuple[str, ast.AST, str]]:
    docstrings = []
    for source_root in SOURCE_ROOTS:
        for path in sorted(source_root.glob("*/src/**/*.py")):
            if "_pb2" in path.name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            relative_path = path.relative_to(REPO_ROOT)
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
                ) and (docstring := ast.get_docstring(node)):
                    name = getattr(node, "name", "<module>")
                    docstrings.append((f"{relative_path}:{name}", node, docstring))
    return docstrings


DOCSTRINGS = _list_docstrings()
FUNCTION_DOCSTRINGS = [
    (label, node, docstring)
    for label, node, docstring in DOCSTRINGS
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    and not any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in node.decorator_list
    )
]


def _has_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        argument.arg not in ("self", "cls")
        for argument in node.args.args + node.args.kwonlyargs
    )


def _returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return node.returns is not None and not (
        isinstance(node.returns, ast.Constant) and node.returns.value is None
    )


@pytest.mark.parametrize(
    ("label", "node", "docstring"),
    FUNCTION_DOCSTRINGS,
    ids=[label for label, _, _ in FUNCTION_DOCSTRINGS],
)
def test_function_docstring_has_args_and_returns(
    label: str, node: ast.FunctionDef, docstring: str
) -> None:
    missing_sections = []
    if _has_parameters(node) and "Args:" not in docstring:
        missing_sections.append("Args:")
    if (
        _returns_value(node)
        and "Returns:" not in docstring
        and "Yields:" not in docstring
    ):
        missing_sections.append("Returns:")
    assert not missing_sections, label


@pytest.mark.parametrize(
    ("label", "docstring"),
    [(label, docstring) for label, _, docstring in DOCSTRINGS],
    ids=[label for label, _, _ in DOCSTRINGS],
)
def test_docstring_sentences_name_their_subject(label: str, docstring: str) -> None:
    prose = " ".join(
        line.strip()
        for line in docstring.splitlines()
        if not line.strip().startswith(SECTION_HEADINGS)
    )
    assert not BARE_PRONOUN_START.findall(prose), label


def _list_source_files() -> list[Path]:
    source_files = [
        *(
            path
            for source_root in SOURCE_ROOTS
            for path in source_root.glob("*/src/**/*.py")
        ),
        *(REPO_ROOT / "scripts").glob("**/*.py"),
    ]
    return sorted(path for path in source_files if "_pb2" not in path.name)


def _is_model_file(path: Path) -> bool:
    return (
        path.name == "models.py"
        or path.name.startswith(("model_", "models_"))
        or path.stem.endswith("_models")
        or path.parent.name == "models"
    )


def _find_docstring_node(body: list[ast.stmt]) -> ast.Expr | None:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0]
    return None


def _collect_docstring_lines(tree: ast.AST) -> set[int]:
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) and (docstring_node := _find_docstring_node(node.body)):
            end_lineno = docstring_node.end_lineno or docstring_node.lineno
            docstring_lines.update(range(docstring_node.lineno, end_lineno + 1))
    return docstring_lines


def _count_non_empty_code_lines(path: Path) -> int:
    source_text = path.read_text(encoding="utf-8")
    docstring_lines = _collect_docstring_lines(ast.parse(source_text))
    return sum(
        1
        for lineno, line in enumerate(source_text.splitlines(), start=1)
        if lineno not in docstring_lines and line.strip()
    )


def _list_function_line_counts() -> list:
    function_line_counts = []
    for path in _list_source_files():
        relative_path = path.relative_to(REPO_ROOT)
        source_text = path.read_text(encoding="utf-8")
        lines = source_text.splitlines()
        tree = ast.parse(source_text)
        docstring_lines = _collect_docstring_lines(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            end_lineno = node.end_lineno or node.lineno
            line_count = sum(
                1
                for lineno in range(node.lineno, end_lineno + 1)
                if lineno not in docstring_lines and lines[lineno - 1].strip()
            )
            function_line_counts.append(
                pytest.param(
                    line_count, id=f"{relative_path}:{node.lineno}:{node.name}"
                )
            )
    return function_line_counts


@pytest.mark.parametrize(
    "path",
    [path for path in _list_source_files() if not _is_model_file(path)],
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_source_file_within_line_limit(path: Path) -> None:
    assert _count_non_empty_code_lines(path) <= MAX_SOURCE_FILE_LINES


@pytest.mark.parametrize("line_count", _list_function_line_counts())
def test_function_within_line_limit(line_count: int) -> None:
    assert line_count <= MAX_FUNCTION_LINES
