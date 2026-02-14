from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pytest

from feast.data_source import RequestSource
from feast.field import Field
from feast.infra.compute_engines.realtime.compute import (
    DataInputNode,
    RealTimeComputeEngine,
    RealTimeComputeEngineConfig,
    RealTimeFeatureBuilder,
    TransformationNode,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Int64


def _create_request_source():
    return RequestSource(
        name="request_source",
        schema=[
            Field(name="input_a", dtype=Int64),
            Field(name="input_b", dtype=Float32),
        ],
    )


def _create_feature_view():
    """Create a FeatureView with a transformation for testing."""
    request_source = _create_request_source()

    @on_demand_feature_view(
        schema=[
            Field(name="output_sum", dtype=Float32),
        ],
        sources=[request_source],
        mode="pandas",
    )
    def test_fv(inputs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"output_sum": inputs["input_a"] + inputs["input_b"]}
        )

    return test_fv


class TestRealTimeComputeEngineConfig:
    def test_config_defaults(self):
        config = RealTimeComputeEngineConfig()
        assert config.type == "local"
        assert config.backend == "python"

    def test_config_with_backend(self):
        config = RealTimeComputeEngineConfig(backend="pandas")
        assert config.backend == "pandas"


class TestDataInputNode:
    def test_execute(self):
        table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        node = DataInputNode(name="test_input", data=table)

        context = MagicMock()
        context.node_outputs = {}

        result = node.execute(context)
        assert result.data.equals(table)


class TestTransformationNode:
    def test_execute(self):
        fv = _create_feature_view()

        input_table = pa.table({"input_a": [1, 2, 3], "input_b": [4.0, 5.0, 6.0]})
        input_node = DataInputNode(name="input", data=input_table)

        transform_node = TransformationNode(
            name="transform",
            feature_view=fv,
            inputs=[input_node],
        )

        context = MagicMock()
        context.node_outputs = {
            "input": input_node.execute(context),
        }

        result = transform_node.execute(context)
        result_df = result.data.to_pandas()
        assert "output_sum" in result_df.columns
        assert list(result_df["output_sum"]) == [5.0, 7.0, 9.0]


class TestRealTimeFeatureBuilder:
    def test_build_returns_execution_plan(self):
        fv = _create_feature_view()
        input_table = pa.table({"input_a": [1], "input_b": [2.0]})

        builder = RealTimeFeatureBuilder(fv, input_table)
        plan = builder.build()

        assert len(plan.nodes) == 2
        assert isinstance(plan.nodes[0], DataInputNode)
        assert isinstance(plan.nodes[1], TransformationNode)

    def test_build_and_execute(self):
        fv = _create_feature_view()
        input_table = pa.table({"input_a": [1, 2], "input_b": [3.0, 4.0]})

        builder = RealTimeFeatureBuilder(fv, input_table)
        plan = builder.build()

        context = MagicMock()
        context.repo_config = MagicMock()
        context.online_store = MagicMock()
        context.node_outputs = {}

        result = plan.execute(context)
        result_df = result.data.to_pandas()
        assert list(result_df["output_sum"]) == [4.0, 6.0]


class TestRealTimeComputeEngine:
    def test_default_backend_is_python(self):
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )
        assert engine._backend_name == "python"

    def test_transform(self):
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        fv = _create_feature_view()
        input_df = pd.DataFrame({"input_a": [1, 2, 3], "input_b": [4.0, 5.0, 6.0]})

        result_df = engine.transform(fv, input_df)
        assert "output_sum" in result_df.columns
        assert list(result_df["output_sum"]) == [5.0, 7.0, 9.0]

    def test_transform_dict(self):
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        fv = _create_feature_view()
        input_dict = {"input_a": [1, 2, 3], "input_b": [4.0, 5.0, 6.0]}

        result = engine.transform_dict(fv, input_dict)
        assert "output_sum" in result
        assert list(result["output_sum"]) == [5.0, 7.0, 9.0]

    def test_transform_single_row(self):
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        fv = _create_feature_view()
        input_df = pd.DataFrame({"input_a": [10], "input_b": [20.0]})

        result_df = engine.transform(fv, input_df)
        assert list(result_df["output_sum"]) == [30.0]

    def test_materialize_not_implemented(self):
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        fv = _create_feature_view()
        input_table = pa.table({"input_a": [1], "input_b": [2.0]})

        with pytest.raises(NotImplementedError):
            engine.materialize(fv, input_table, "test_project")
