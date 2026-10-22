"""The package imports on every Python of ``requires-python`` (>= 3.9): no ``X | Y`` type union is built at
run time (outside annotations, which ``from __future__ import annotations`` keeps as strings), and every
source file parses with the 3.9 grammar."""

from __future__ import annotations

import ast
import typing
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "corenet"
SOURCES = sorted(SRC.rglob("*.py"))


def _statements(body):
    """Statements evaluated at import time: module and class bodies, not function bodies; the value of an
    annotated assignment, not its annotation."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(stmt, ast.ClassDef):
            yield from _statements(stmt.body)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                yield stmt.value
        else:
            yield stmt


def _type_like(node) -> bool:
    """A name, attribute, subscript (``list[int]``), ``None`` or a ``|`` of those: a type expression."""
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and _type_like(node.left)
        and _type_like(node.right)
    )


def runtime_unions(path: Path) -> list[str]:
    """``<file>:<line>`` of every ``A | B`` between type expressions evaluated at import time."""
    found = []
    for stmt in _statements(ast.parse(path.read_text()).body):
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.BitOr)
                and _type_like(node)
            ):
                found.append(f"{_label(path)}:{node.lineno}")
    return sorted(set(found))


def _label(path: Path) -> str:
    return str(path.relative_to(SRC)) if SRC in path.parents else path.name


def test_detector_sees_a_runtime_union(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from __future__ import annotations\nfrom collections.abc import Iterable\n"
        "SeedLike = int | str | Iterable[int | str]\nx: int | None = None\n"
        "def f(a: int | None) -> str | None:\n    y: int | None = None\n    return None\n"
    )
    assert runtime_unions(bad) == [
        "bad.py:3"
    ]  # the annotations on lines 4-6 are not evaluated


@pytest.mark.parametrize(
    "path", SOURCES, ids=[str(p.relative_to(SRC)) for p in SOURCES]
)
def test_no_runtime_type_union(path):
    assert runtime_unions(path) == [], (
        "type union evaluated at import time (fails on Python 3.9); use typing.Union"
    )


@pytest.mark.parametrize(
    "path", SOURCES, ids=[str(p.relative_to(SRC)) for p in SOURCES]
)
def test_parses_with_python39_grammar(path):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 9))


def test_seed_alias_is_a_typing_union():
    from corenet.data.folds import SeedLike, Squares

    assert typing.get_origin(SeedLike) is typing.Union
    assert set(typing.get_args(SeedLike)[:2]) == {int, str}
    assert (
        Squares("delphes_h5", "fold", 0)._seed
        == Squares(("delphes_h5", "fold", 0))._seed
    )
