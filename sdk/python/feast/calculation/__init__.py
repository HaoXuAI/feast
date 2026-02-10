"""
Calculation module for Feast.

Provides the Calculation class for defining SQL-like expression-based features
in OnDemandFeatureViews.
"""

from typing import Dict, Optional

from typeguard import typechecked

from feast.protos.feast.core.Calculation_pb2 import Calculation as CalculationProto


@typechecked
class Calculation:
    """
    A Calculation describes a computed feature that is applied to an
    OnDemandFeatureView via the `calculations` parameter.

    Expressions use standard SQL syntax, powered by sqlglot. Supported operations:
    - Arithmetic: +, -, *, / (e.g., ``"amount + 1"``)
    - Comparisons: >, <, >=, <=, =, !=, <> (e.g., ``"amount > 100"``)
    - Logical operators: AND, OR, NOT
    - CASE statements: ``"CASE WHEN amount > 100 THEN 'high' ELSE 'low' END"``
    - COALESCE: ``"COALESCE(amount, 0)"``
    - DATEDIFF: ``"DATEDIFF(end_time, start_time)"``
    - IS NULL / IS NOT NULL
    - String functions: CONCAT, UPPER, LOWER, etc.
    - Math functions: ABS, ROUND, CEIL, FLOOR, etc.
    - Conditional: IF(condition, true_val, false_val)
    - BETWEEN, IN

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


__all__ = ["Calculation"]
