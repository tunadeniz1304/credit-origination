"""Safe expression evaluator for the policy DSL (no ``eval``).

Expressions are parsed with :mod:`ast` and walked with a strict whitelist:
numeric/boolean constants, variable names (looked up in the context),
arithmetic ``+ - * / %``, comparisons (including chained ones), ``and`` /
``or`` / ``not``, unary minus and calls to ``min``, ``max`` and ``abs``.
Attribute access, subscripts, lambdas, comprehensions and every other node
raise :class:`UnsafeExpressionError`. A ``None`` operand makes a comparison
false (missing data never fires a rule by accident).
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any


class UnsafeExpressionError(ValueError):
    """The expression uses a construct outside the whitelist."""


class UnknownVariableError(KeyError):
    """The expression references a name missing from the context."""


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_CMPOPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_FUNCS: dict[str, Callable[..., Any]] = {"min": min, "max": max, "abs": abs}


@lru_cache(maxsize=512)
def compile_expression(source: str) -> ast.Expression:
    tree = ast.parse(source, mode="eval")
    _validate(tree.body)
    return tree


def _validate(node: ast.AST) -> None:
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate(value)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise UnsafeExpressionError(f"operator {type(node.op).__name__} not allowed")
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub | ast.Not):
            raise UnsafeExpressionError("unary operator not allowed")
        _validate(node.operand)
    elif isinstance(node, ast.Compare):
        if any(type(op) not in _CMPOPS for op in node.ops):
            raise UnsafeExpressionError("comparison not allowed")
        _validate(node.left)
        for comparator in node.comparators:
            _validate(comparator)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise UnsafeExpressionError("only min/max/abs calls are allowed")
        for arg in node.args:
            _validate(arg)
    elif isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise UnsafeExpressionError("private names are not allowed")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, int | float | bool) or isinstance(node.value, complex):
            raise UnsafeExpressionError("only numeric constants are allowed")
    else:
        raise UnsafeExpressionError(f"{type(node).__name__} not allowed")


def _eval(node: ast.AST, ctx: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(bool(_eval(v, ctx)) for v in node.values)
        return any(bool(_eval(v, ctx)) for v in node.values)
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, ctx), _eval(node.right, ctx)
        if left is None or right is None:
            return None
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, ctx)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        return None if operand is None else -operand
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, ctx)
            if left is None or right is None:
                return False
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        args = [_eval(a, ctx) for a in node.args]
        if any(a is None for a in args):
            return None
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        if node.id in ("True", "False"):
            return node.id == "True"
        if node.id not in ctx:
            raise UnknownVariableError(node.id)
        return ctx[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise UnsafeExpressionError(type(node).__name__)  # pragma: no cover - validated earlier


def evaluate(source: str, context: Mapping[str, Any]) -> Any:
    return _eval(compile_expression(source).body, context)
