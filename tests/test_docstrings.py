"""Check docstring sections and subject naming across all apps and libraries."""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SOURCE_ROOTS = (REPO_ROOT / "apps", REPO_ROOT / "libs")
BARE_PRONOUN_START = re.compile(r"(?:^|[.!?]\s+)(It|They|Its|Their)\s")
SECTION_HEADINGS = ("Args:", "Returns:", "Raises:", "Yields:")


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
