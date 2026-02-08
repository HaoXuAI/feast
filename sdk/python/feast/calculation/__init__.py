"""
Calculation module for Feast.

Provides the Calculation class for defining SQL-like expression-based features
in OnDemandFeatureViews.
"""

from typing import Any, Dict, List, Optional, Union

from typeguard import typechecked

from feast.protos.feast.core.Calculation_pb2 import Calculation as CalculationProto


@typechecked
class Calculation:
    """
    A Calculation describes a computed feature that is applied to an
    OnDemandFeatureView via the `calculations` parameter.

    The expression supports SQL-like syntax including:
    - Arithmetic: +, -, *, / (e.g., ``"source.amount + 1"``)
    - Comparisons: >, <, >=, <=, ==, != (e.g., ``"source.amount > 100"``)
    - Logical operators: AND, OR, NOT
    - CASE statements: ``"CASE WHEN amount > 100 THEN 'high' ELSE 'low' END"``
    - COALESCE: ``"COALESCE(source.amount, 0)"``
    - DATEDIFF: ``"DATEDIFF('minutes', source.start_time, source.end_time)"``

    Attributes:
        name: The name of this calculated feature.
        expr: The calculation expression string.
        tags: User-defined metadata tags as key-value pairs.
    """

    name: str
    expr: str
    tags: Dict[str, str]

    def __init__(
        self,
        *,
        name: str,
        expr: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Creates a Calculation object.

        Args:
            name: The name of the calculated feature.
            expr: The SQL-like expression string defining the calculation.
            tags: Optional user-defined metadata tags.

        Raises:
            ValueError: If name or expr is empty.
        """
        if not name:
            raise ValueError("Calculation 'name' must not be empty.")
        if not expr:
            raise ValueError("Calculation 'expr' must not be empty.")
        self.name = name
        self.expr = expr
        self.tags = tags or {}

    def to_proto(self) -> CalculationProto:
        """Converts a Calculation object to its protobuf representation."""
        return CalculationProto(
            name=self.name,
            expr=self.expr,
            tags=self.tags,
        )

    @classmethod
    def from_proto(cls, calc_proto: CalculationProto) -> "Calculation":
        """Creates a Calculation object from a protobuf representation."""
        return cls(
            name=calc_proto.name,
            expr=calc_proto.expr,
            tags=dict(calc_proto.tags),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Calculation):
            raise TypeError("Comparisons should only involve Calculation objects.")
        return (
            self.name == other.name
            and self.expr == other.expr
            and self.tags == other.tags
        )

    def __repr__(self) -> str:
        return (
            f"Calculation(name={self.name!r}, expr={self.expr!r}, tags={self.tags!r})"
        )

    def __str__(self) -> str:
        return f"Calculation(name={self.name}, expr={self.expr})"


def apply_calculations(
    data: Union[Dict[str, List[Any]], "pyarrow.Table"],
    calculations: List[Calculation],
    mode: str,
) -> Union[Dict[str, List[Any]], "pyarrow.Table"]:
    """
    Apply a list of Calculation objects to input data.

    Args:
        data: Input data as a dict (python mode) or pyarrow Table (pandas mode).
        calculations: List of Calculation objects to evaluate.
        mode: Execution mode - 'python' or 'pandas'.

    Returns:
        The data with calculated columns appended.
    """
    if not calculations:
        return data

    if mode == "python":
        return _apply_calculations_dict(data, calculations)
    elif mode == "pandas":
        return _apply_calculations_arrow(data, calculations)
    else:
        raise ValueError(f"Unsupported mode for calculations: {mode}")


def _apply_calculations_dict(
    data: Dict[str, List[Any]],
    calculations: List[Calculation],
) -> Dict[str, List[Any]]:
    """Apply calculations to dict-based data."""
    from feast.calculation.expression_engine import evaluate_expression

    if not data or not any(len(v) > 0 for v in data.values() if isinstance(v, list)):
        return data

    result = dict(data)
    # Determine the number of rows
    num_rows = 0
    for v in data.values():
        if isinstance(v, list):
            num_rows = len(v)
            break

    for calc in calculations:
        computed_values = []
        for i in range(num_rows):
            row = {k: (v[i] if isinstance(v, list) else v) for k, v in data.items()}
            computed_values.append(evaluate_expression(calc.expr, row))
        result[calc.name] = computed_values

    return result


def _apply_calculations_arrow(
    data: "pyarrow.Table",
    calculations: List[Calculation],
) -> "pyarrow.Table":
    """Apply calculations to a pyarrow Table."""
    import pyarrow as pa

    from feast.calculation.expression_engine import evaluate_expression

    if data.num_rows == 0:
        return data

    result = data
    for calc in calculations:
        computed_values = []
        col_dict = {name: data.column(name).to_pylist() for name in data.column_names}
        for i in range(data.num_rows):
            row = {k: v[i] for k, v in col_dict.items()}
            computed_values.append(evaluate_expression(calc.expr, row))
        result = result.append_column(calc.name, pa.array(computed_values))

    return result


__all__ = ["Calculation", "apply_calculations"]
