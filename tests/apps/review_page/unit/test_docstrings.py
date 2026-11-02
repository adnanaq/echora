import ast
import re
from pathlib import Path

import pytest

PAGE_SOURCES = (
    Path(__file__).parents[4] / "apps" / "review_page" / "src" / "review_page"
)
BARE_PRONOUN_START = re.compile(r"(?:^|[.!?]\s+)(It|They|Its|Their)\s")
SECTION_HEADINGS = ("Args:", "Returns:", "Raises:", "Yields:")


def _list_docstrings() -> list[tuple[str, ast.AST, str]]:
    docstrings = []
    for path in sorted(PAGE_SOURCES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ) and (docstring := ast.get_docstring(node)):
                name = getattr(node, "name", "<module>")
                docstrings.append((f"{path.name}:{name}", node, docstring))
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
    missing_sections = [
        section
        for section, is_needed in (
            ("Args:", _has_parameters(node)),
            ("Returns:", _returns_value(node)),
        )
        if is_needed and section not in docstring
    ]
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
