# Copyright 2026 The Feast Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Calculation class and expression engine (powered by sqlglot)."""

from datetime import datetime, timezone

import pytest

from feast.calculation import Calculation
from feast.calculation.expression_engine import (
    ExpressionError,
    evaluate_batch,
    evaluate_expression,
)


class TestCalculationClass:
    """Tests for the Calculation class."""

    def test_basic_creation(self):
        calc = Calculation(name="total", expr="amount + tax")
        assert calc.name == "total"
        assert calc.expr == "amount + tax"
        assert calc.tags == {}

    def test_creation_with_tags(self):
        calc = Calculation(
            name="total", expr="amount + tax", tags={"team": "ml", "version": "1"}
        )
        assert calc.tags == {"team": "ml", "version": "1"}

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Calculation(name="", expr="amount + 1")

    def test_empty_expr_raises(self):
        with pytest.raises(ValueError, match="expr"):
            Calculation(name="total", expr="")

    def test_equality(self):
        a = Calculation(name="total", expr="amount + tax")
        b = Calculation(name="total", expr="amount + tax")
        assert a == b

    def test_inequality_different_name(self):
        a = Calculation(name="total", expr="amount + tax")
        b = Calculation(name="subtotal", expr="amount + tax")
        assert a != b

    def test_inequality_different_expr(self):
        a = Calculation(name="total", expr="amount + tax")
        b = Calculation(name="total", expr="amount - tax")
        assert a != b

    def test_inequality_different_tags(self):
        a = Calculation(name="total", expr="amount + tax", tags={"v": "1"})
        b = Calculation(name="total", expr="amount + tax", tags={"v": "2"})
        assert a != b

    def test_repr(self):
        calc = Calculation(name="total", expr="amount + 1")
        r = repr(calc)
        assert "total" in r
        assert "amount + 1" in r

    def test_str(self):
        calc = Calculation(name="total", expr="amount + 1")
        s = str(calc)
        assert "total" in s
        assert "amount + 1" in s


class TestCalculationProto:
    """Tests for proto serialization/deserialization."""

    def test_to_proto(self):
        calc = Calculation(name="total", expr="amount + tax", tags={"v": "1"})
        proto = calc.to_proto()
        assert proto.name == "total"
        assert proto.expr == "amount + tax"
        assert dict(proto.tags) == {"v": "1"}

    def test_from_proto(self):
        calc = Calculation(name="total", expr="amount + tax", tags={"v": "1"})
        proto = calc.to_proto()
        restored = Calculation.from_proto(proto)
        assert restored == calc

    def test_roundtrip(self):
        calcs = [
            Calculation(name="a", expr="x + 1"),
            Calculation(name="b", expr="COALESCE(x, 0)", tags={"t": "test"}),
        ]
        for calc in calcs:
            assert Calculation.from_proto(calc.to_proto()) == calc


class TestExpressionEngine:
    """Tests for the sqlglot-powered expression engine."""

    # -- Arithmetic --

    def test_addition(self):
        assert evaluate_expression("amount + 1", {"amount": 10}) == 11

    def test_subtraction(self):
        assert evaluate_expression("amount - 5", {"amount": 10}) == 5

    def test_multiplication(self):
        assert evaluate_expression("amount * 2", {"amount": 10}) == 20

    def test_division(self):
        assert evaluate_expression("amount / 2", {"amount": 10}) == 5.0

    def test_division_by_zero(self):
        with pytest.raises(ExpressionError):
            evaluate_expression("amount / 0", {"amount": 10})

    def test_float_arithmetic(self):
        result = evaluate_expression("price * quantity", {"price": 9.99, "quantity": 3})
        assert abs(result - 29.97) < 0.001

    def test_complex_arithmetic(self):
        result = evaluate_expression(
            "amount + tax - discount", {"amount": 100, "tax": 10, "discount": 5}
        )
        assert result == 105

    def test_operator_precedence(self):
        result = evaluate_expression("a + b * c", {"a": 2, "b": 3, "c": 4})
        assert result == 14  # 2 + (3*4) = 14, not (2+3)*4 = 20

    def test_parenthesized_expression(self):
        result = evaluate_expression("(a + b) * c", {"a": 2, "b": 3, "c": 4})
        assert result == 20

    def test_unary_minus(self):
        assert evaluate_expression("-amount", {"amount": 10}) == -10

    # -- Comparisons (SQL standard: = for equality, <> or != for not equal) --

    def test_greater_than(self):
        assert evaluate_expression("amount > 100", {"amount": 150}) is True
        assert evaluate_expression("amount > 100", {"amount": 50}) is False

    def test_less_than(self):
        assert evaluate_expression("amount < 100", {"amount": 50}) is True
        assert evaluate_expression("amount < 100", {"amount": 150}) is False

    def test_greater_equal(self):
        assert evaluate_expression("amount >= 100", {"amount": 100}) is True

    def test_less_equal(self):
        assert evaluate_expression("amount <= 100", {"amount": 100}) is True

    def test_equal(self):
        assert evaluate_expression("amount = 100", {"amount": 100}) is True
        assert evaluate_expression("amount = 100", {"amount": 99}) is False

    def test_not_equal(self):
        assert evaluate_expression("amount != 100", {"amount": 99}) is True

    def test_not_equal_sql_standard(self):
        assert evaluate_expression("amount <> 100", {"amount": 99}) is True

    # -- Logical operators --

    def test_and(self):
        assert (
            evaluate_expression(
                "amount > 100 AND amount < 200", {"amount": 150}
            )
            is True
        )
        assert (
            evaluate_expression(
                "amount > 100 AND amount < 200", {"amount": 50}
            )
            is False
        )

    def test_or(self):
        assert (
            evaluate_expression(
                "amount < 10 OR amount > 100", {"amount": 150}
            )
            is True
        )

    def test_not(self):
        assert evaluate_expression("NOT amount > 100", {"amount": 50}) is True

    # -- Literals --

    def test_string_literal(self):
        result = evaluate_expression("'hello'", {"_placeholder": 0})
        assert result == "hello"

    def test_null(self):
        assert evaluate_expression("NULL", {"_placeholder": 0}) is None

    def test_is_null(self):
        assert evaluate_expression("amount IS NULL", {"amount": None}) is True
        assert evaluate_expression("amount IS NULL", {"amount": 5}) is False

    def test_is_not_null(self):
        assert evaluate_expression("amount IS NOT NULL", {"amount": 5}) is True
        assert evaluate_expression("amount IS NOT NULL", {"amount": None}) is False

    # -- CASE expressions --

    def test_case_when_true(self):
        result = evaluate_expression(
            "CASE WHEN amount > 100 THEN 'high' ELSE 'low' END",
            {"amount": 150},
        )
        assert result == "high"

    def test_case_when_false(self):
        result = evaluate_expression(
            "CASE WHEN amount > 100 THEN 'high' ELSE 'low' END",
            {"amount": 50},
        )
        assert result == "low"

    def test_case_multiple_when(self):
        result = evaluate_expression(
            "CASE WHEN amount > 1000 THEN 'very high' WHEN amount > 100 THEN 'high' ELSE 'low' END",
            {"amount": 500},
        )
        assert result == "high"

    def test_case_no_else(self):
        result = evaluate_expression(
            "CASE WHEN amount > 100 THEN 'high' END",
            {"amount": 50},
        )
        assert result is None

    # -- COALESCE --

    def test_coalesce_first_non_null(self):
        result = evaluate_expression(
            "COALESCE(amount, 0)", {"amount": 42}
        )
        assert result == 42

    def test_coalesce_fallback(self):
        result = evaluate_expression(
            "COALESCE(amount, 0)", {"amount": None}
        )
        assert result == 0

    def test_coalesce_multiple(self):
        result = evaluate_expression(
            "COALESCE(a, b, c, 99)", {"a": None, "b": None, "c": 7}
        )
        assert result == 7

    def test_coalesce_all_null(self):
        result = evaluate_expression(
            "COALESCE(a, b)", {"a": None, "b": None}
        )
        assert result is None

    # -- DATEDIFF (sqlglot 2-arg: DATEDIFF(end, start) returns days) --

    def test_datediff_days(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 8, tzinfo=timezone.utc)
        result = evaluate_expression(
            "DATEDIFF(end_time, start_time)",
            {"start_time": start, "end_time": end},
        )
        assert result == 7

    # -- Column references --

    def test_simple_column_ref(self):
        result = evaluate_expression("amount + 1", {"amount": 10})
        assert result == 11

    def test_underscore_column_ref(self):
        result = evaluate_expression(
            "source__amount + 1", {"source__amount": 10}
        )
        assert result == 11

    def test_missing_column_raises(self):
        with pytest.raises(ExpressionError):
            evaluate_expression("missing_col + 1", {"amount": 10})

    # -- Batch evaluation --

    def test_evaluate_batch(self):
        rows = [{"amount": 10}, {"amount": 20}, {"amount": 30}]
        result = evaluate_batch("amount + 1", "total", rows)
        assert result == [11, 21, 31]

    def test_evaluate_batch_empty(self):
        result = evaluate_batch("amount + 1", "total", [])
        assert result == []

    # -- Error handling --

    def test_invalid_expression(self):
        with pytest.raises(ExpressionError):
            evaluate_expression("@#$%", {"amount": 10})

    # -- Additional SQL functions via sqlglot --

    def test_concat(self):
        result = evaluate_expression(
            "CONCAT(first_name, ' ', last_name)",
            {"first_name": "John", "last_name": "Doe"},
        )
        assert result == "John Doe"

    def test_abs(self):
        assert evaluate_expression("ABS(amount)", {"amount": -42}) == 42

    def test_if_function(self):
        result = evaluate_expression(
            "IF(amount > 100, 'high', 'low')", {"amount": 150}
        )
        assert result == "high"

    def test_between(self):
        assert (
            evaluate_expression(
                "amount BETWEEN 10 AND 100", {"amount": 50}
            )
            is True
        )
        assert (
            evaluate_expression(
                "amount BETWEEN 10 AND 100", {"amount": 150}
            )
            is False
        )

    def test_in_list(self):
        assert (
            evaluate_expression(
                "status IN ('active', 'pending')", {"status": "active"}
            )
            is True
        )
        assert (
            evaluate_expression(
                "status IN ('active', 'pending')", {"status": "closed"}
            )
            is False
        )
