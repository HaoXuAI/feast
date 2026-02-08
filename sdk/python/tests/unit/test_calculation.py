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

"""Tests for the Calculation class and expression engine."""

from datetime import datetime, timezone

import pyarrow as pa
import pytest

from feast.calculation import Calculation, apply_calculations
from feast.calculation.expression_engine import (
    ExpressionError,
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
    """Tests for the SQL-like expression engine."""

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
        with pytest.raises(ExpressionError, match="Division by zero"):
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

    # -- Comparisons --

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
        assert evaluate_expression("amount == 100", {"amount": 100}) is True
        assert evaluate_expression("amount == 100", {"amount": 99}) is False

    def test_not_equal(self):
        assert evaluate_expression("amount != 100", {"amount": 99}) is True

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
        result = evaluate_expression("'hello'", {})
        assert result == "hello"

    def test_boolean_true(self):
        assert evaluate_expression("TRUE", {}) is True

    def test_boolean_false(self):
        assert evaluate_expression("FALSE", {}) is False

    def test_null(self):
        assert evaluate_expression("NULL", {}) is None

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

    # -- DATEDIFF --

    def test_datediff_minutes(self):
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
        result = evaluate_expression(
            "DATEDIFF('minutes', start_time, end_time)",
            {"start_time": start, "end_time": end},
        )
        assert result == 30

    def test_datediff_hours(self):
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        result = evaluate_expression(
            "DATEDIFF('hours', start_time, end_time)",
            {"start_time": start, "end_time": end},
        )
        assert result == 3

    def test_datediff_days(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 8, tzinfo=timezone.utc)
        result = evaluate_expression(
            "DATEDIFF('days', start_time, end_time)",
            {"start_time": start, "end_time": end},
        )
        assert result == 7

    def test_datediff_seconds(self):
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, 45, tzinfo=timezone.utc)
        result = evaluate_expression(
            "DATEDIFF('seconds', start_time, end_time)",
            {"start_time": start, "end_time": end},
        )
        assert result == 45

    # -- Column references --

    def test_dotted_column_ref(self):
        # "source.amount" should resolve to key "source__amount" or "source.amount" or just "amount"
        result = evaluate_expression(
            "source.amount + 1", {"source__amount": 10}
        )
        assert result == 11

    def test_simple_column_ref(self):
        result = evaluate_expression("amount + 1", {"amount": 10})
        assert result == 11

    def test_dotted_column_ref_with_dot_key(self):
        result = evaluate_expression(
            "source.amount + 1", {"source.amount": 10}
        )
        assert result == 11

    def test_missing_column_raises(self):
        with pytest.raises(ExpressionError, match="not found"):
            evaluate_expression("missing_col + 1", {"amount": 10})

    # -- Error handling --

    def test_invalid_expression(self):
        with pytest.raises(ExpressionError):
            evaluate_expression("@#$%", {"amount": 10})

    def test_datediff_non_datetime(self):
        with pytest.raises(ExpressionError, match="datetime"):
            evaluate_expression(
                "DATEDIFF('minutes', start, end_val)",
                {"start": "not a date", "end_val": "also not"},
            )


class TestApplyCalculations:
    """Tests for apply_calculations function."""

    def test_apply_to_dict_python_mode(self):
        data = {
            "amount": [100, 200, 300],
            "tax": [10, 20, 30],
        }
        calcs = [Calculation(name="total", expr="amount + tax")]
        result = apply_calculations(data, calcs, "python")
        assert result["total"] == [110, 220, 330]

    def test_apply_multiple_calculations_dict(self):
        data = {
            "amount": [100, 200],
            "tax": [10, 20],
        }
        calcs = [
            Calculation(name="total", expr="amount + tax"),
            Calculation(name="is_high", expr="amount > 150"),
        ]
        result = apply_calculations(data, calcs, "python")
        assert result["total"] == [110, 220]
        assert result["is_high"] == [False, True]

    def test_apply_to_arrow_pandas_mode(self):
        table = pa.table({"amount": [100, 200, 300], "tax": [10, 20, 30]})
        calcs = [Calculation(name="total", expr="amount + tax")]
        result = apply_calculations(table, calcs, "pandas")
        assert isinstance(result, pa.Table)
        assert result.column("total").to_pylist() == [110, 220, 330]

    def test_no_calculations_returns_original(self):
        data = {"amount": [1, 2]}
        result = apply_calculations(data, [], "python")
        assert result is data

    def test_empty_data_returns_original(self):
        data = {"amount": [], "tax": []}
        calcs = [Calculation(name="total", expr="amount + tax")]
        result = apply_calculations(data, calcs, "python")
        assert result == data

    def test_empty_arrow_table_returns_original(self):
        table = pa.table({"amount": pa.array([], type=pa.int64())})
        calcs = [Calculation(name="total", expr="amount + 1")]
        result = apply_calculations(table, calcs, "pandas")
        assert result.num_rows == 0

    def test_case_expression_in_apply(self):
        data = {
            "amount": [50, 150, 250],
        }
        calcs = [
            Calculation(
                name="tier",
                expr="CASE WHEN amount > 200 THEN 'premium' WHEN amount > 100 THEN 'standard' ELSE 'basic' END",
            )
        ]
        result = apply_calculations(data, calcs, "python")
        assert result["tier"] == ["basic", "standard", "premium"]

    def test_coalesce_in_apply(self):
        data = {
            "value": [None, 42, None],
        }
        calcs = [Calculation(name="safe_value", expr="COALESCE(value, 0)")]
        result = apply_calculations(data, calcs, "python")
        assert result["safe_value"] == [0, 42, 0]

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            apply_calculations({"a": [1]}, [Calculation(name="b", expr="a + 1")], "unknown")


class TestCalculationInOnDemandFeatureView:
    """Tests for Calculation integration with OnDemandFeatureView."""

    def test_odfv_with_calculations_proto_roundtrip(self):
        """Test that ODFV with calculations survives proto serialization."""
        from unittest.mock import MagicMock

        from feast.data_source import RequestSource
        from feast.field import Field
        from feast.on_demand_feature_view import OnDemandFeatureView
        from feast.transformation.pandas_transformation import PandasTransformation
        from feast.types import Float64, Int64

        request_source = RequestSource(
            name="input",
            schema=[
                Field(name="amount", dtype=Int64),
                Field(name="tax", dtype=Int64),
            ],
        )

        def dummy_udf(features_df):
            return features_df

        transformation = PandasTransformation(
            udf=dummy_udf,
            udf_string="def dummy_udf(features_df): return features_df",
        )

        calcs = [
            Calculation(name="total", expr="amount + tax"),
            Calculation(name="is_high", expr="amount > 100", tags={"v": "1"}),
        ]

        odfv = OnDemandFeatureView(
            name="test_odfv",
            schema=[Field(name="total", dtype=Float64)],
            sources=[request_source],
            feature_transformation=transformation,
            mode="pandas",
            calculations=calcs,
        )

        proto = odfv.to_proto()

        # Verify calculations in proto
        assert len(proto.spec.calculations) == 2
        assert proto.spec.calculations[0].name == "total"
        assert proto.spec.calculations[0].expr == "amount + tax"
        assert proto.spec.calculations[1].name == "is_high"
        assert proto.spec.calculations[1].expr == "amount > 100"
        assert dict(proto.spec.calculations[1].tags) == {"v": "1"}

        # Roundtrip
        restored = OnDemandFeatureView.from_proto(proto)
        assert len(restored.calculations) == 2
        assert restored.calculations[0] == calcs[0]
        assert restored.calculations[1] == calcs[1]
