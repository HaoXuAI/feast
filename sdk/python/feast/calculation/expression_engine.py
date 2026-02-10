"""
Expression engine for evaluating SQL-like calculation expressions.

Uses sqlglot as the SQL parser and execution engine. Expressions are
evaluated by wrapping them in ``SELECT <expr> AS <alias> FROM __data__``
and executing against an in-memory table via ``sqlglot.executor``.

Supports all standard SQL expression syntax including:
- Arithmetic: +, -, *, /
- Comparisons: >, <, >=, <=, =, !=, <>
- Logical operators: AND, OR, NOT
- CASE WHEN ... THEN ... ELSE ... END
- COALESCE(expr, expr, ...)
- DATEDIFF(start, end)
- IS NULL / IS NOT NULL
- Column references
- Numeric and string literals
"""

from typing import Any, Dict, List

import sqlglot
import sqlglot.executor


class ExpressionError(Exception):
    """Raised when an expression cannot be parsed or evaluated."""

    pass


def evaluate_expression(expr: str, row: Dict[str, Any]) -> Any:
    """
    Evaluate a SQL-like expression against a single row of data.

    Args:
        expr: The SQL-like expression string (e.g., ``"amount + tax"``).
        row: A dictionary mapping column names to values.

    Returns:
        The computed result.

    Raises:
        ExpressionError: If the expression is invalid or evaluation fails.
    """
    try:
        result = evaluate_batch(expr, "result", [row])
        return result[0]
    except ExpressionError:
        raise
    except Exception as e:
        raise ExpressionError(f"Failed to evaluate expression '{expr}': {e}") from e


def evaluate_batch(
    expr: str, alias: str, rows: List[Dict[str, Any]]
) -> List[Any]:
    """
    Evaluate a SQL-like expression against multiple rows of data at once.

    Args:
        expr: The SQL-like expression string.
        alias: The output column alias name.
        rows: A list of dictionaries, each mapping column names to values.

    Returns:
        A list of computed results, one per input row.

    Raises:
        ExpressionError: If the expression is invalid or evaluation fails.
    """
    if not rows:
        return []

    try:
        # Quote the alias to avoid SQL keyword conflicts
        quoted_alias = f'"{alias}"'
        sql = f"SELECT {expr} AS {quoted_alias} FROM __data__"
        result = sqlglot.executor.execute(sql, tables={"__data__": rows})
        return [row[0] for row in result.rows]
    except Exception as e:
        raise ExpressionError(f"Failed to evaluate expression '{expr}': {e}") from e
