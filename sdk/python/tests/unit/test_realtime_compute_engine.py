from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pytest

from feast.data_source import RequestSource
from feast.entity import Entity
from feast.feature_view import FeatureView
from feast.field import Field
from feast.infra.compute_engines.realtime.compute import (
    DataInputNode,
    ODFVTransformationNode,
    RealTimeComputeEngine,
    RealTimeComputeEngineConfig,
)
from feast.on_demand_feature_view import OnDemandFeatureView, on_demand_feature_view
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


def _create_request_source():
    return RequestSource(
        name="request_source",
        schema=[
            Field(name="input_a", dtype=Int64),
            Field(name="input_b", dtype=Float32),
        ],
    )


def _create_odfv(online=False, offline=False):
    request_source = _create_request_source()

    @on_demand_feature_view(
        schema=[
            Field(name="output_sum", dtype=Float32),
        ],
        sources=[request_source],
        mode="pandas",
        online=online,
        offline=offline,
    )
    def test_odfv(inputs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"output_sum": inputs["input_a"] + inputs["input_b"]}
        )

    return test_odfv


class TestOnDemandFeatureViewFlags:
    def test_odfv_default_flags(self):
        """Test that ODFV defaults to online=False, offline=False."""
        odfv = _create_odfv()
        assert odfv.online is False
        assert odfv.offline is False
        assert odfv.ttl == timedelta(0)

    def test_odfv_online_flag(self):
        """Test ODFV with online=True."""
        odfv = _create_odfv(online=True)
        assert odfv.online is True
        assert odfv.offline is False
        # write_to_online_store should be set for backward compat
        assert odfv.write_to_online_store is True

    def test_odfv_offline_flag(self):
        """Test ODFV with offline=True."""
        odfv = _create_odfv(offline=True)
        assert odfv.online is False
        assert odfv.offline is True

    def test_odfv_both_flags(self):
        """Test ODFV with both online=True and offline=True."""
        odfv = _create_odfv(online=True, offline=True)
        assert odfv.online is True
        assert odfv.offline is True

    def test_odfv_backward_compat_write_to_online_store(self):
        """Test backward compatibility: write_to_online_store=True maps to online=True."""
        request_source = _create_request_source()
        odfv = OnDemandFeatureView(
            name="test_compat_odfv",
            schema=[Field(name="output_sum", dtype=Float32)],
            sources=[request_source],
            mode="pandas",
            write_to_online_store=True,
            feature_transformation=None,
            udf=lambda x: x,
        )
        assert odfv.online is True
        assert odfv.write_to_online_store is True

    def test_odfv_ttl(self):
        """Test ODFV with custom TTL."""
        request_source = _create_request_source()

        @on_demand_feature_view(
            schema=[Field(name="output_sum", dtype=Float32)],
            sources=[request_source],
            mode="pandas",
            ttl=timedelta(hours=1),
        )
        def ttl_odfv(inputs: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"output_sum": inputs["input_a"] + inputs["input_b"]})

        assert ttl_odfv.ttl == timedelta(hours=1)

    def test_odfv_equality_with_new_flags(self):
        """Test that ODFVs with different flags are not equal."""
        odfv1 = _create_odfv(online=True, offline=False)
        odfv2 = _create_odfv(online=False, offline=True)
        assert odfv1 != odfv2

    def test_odfv_copy_preserves_flags(self):
        """Test that __copy__ preserves new flags."""
        import copy

        odfv = _create_odfv(online=True, offline=True)
        odfv_copy = copy.copy(odfv)
        assert odfv_copy.online is True
        assert odfv_copy.offline is True
        assert odfv_copy.ttl == timedelta(0)


class TestRealTimeComputeEngineConfig:
    def test_config_defaults(self):
        """Test default configuration."""
        config = RealTimeComputeEngineConfig()
        assert config.type == "local"
        assert config.backend == "pandas"

    def test_config_with_backend(self):
        """Test configuration with backend."""
        config = RealTimeComputeEngineConfig(backend="pandas")
        assert config.backend == "pandas"


class TestDataInputNode:
    def test_execute(self):
        """Test DataInputNode provides pre-loaded data."""
        table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        node = DataInputNode(name="test_input", data=table)

        context = MagicMock()
        context.node_outputs = {}

        result = node.execute(context)
        assert result.data.equals(table)


class TestODFVTransformationNode:
    def test_execute(self):
        """Test ODFVTransformationNode applies transformation."""
        odfv = _create_odfv()

        from feast.infra.compute_engines.backends.pandas_backend import PandasBackend

        backend = PandasBackend()
        input_table = pa.table({"input_a": [1, 2, 3], "input_b": [4.0, 5.0, 6.0]})
        input_node = DataInputNode(name="input", data=input_table)

        transform_node = ODFVTransformationNode(
            name="transform",
            odfv=odfv,
            backend=backend,
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


class TestRealTimeComputeEngine:
    def test_transform(self):
        """Test RealTimeComputeEngine.transform() applies ODFV transformation."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
            backend="pandas",
        )

        odfv = _create_odfv()
        input_df = pd.DataFrame({"input_a": [1, 2, 3], "input_b": [4.0, 5.0, 6.0]})

        result_df = engine.transform(odfv, input_df)
        assert "output_sum" in result_df.columns
        assert list(result_df["output_sum"]) == [5.0, 7.0, 9.0]

    def test_transform_dict(self):
        """Test RealTimeComputeEngine.transform_dict() applies ODFV transformation."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
            backend="pandas",
        )

        odfv = _create_odfv()
        input_dict = {"input_a": [1, 2, 3], "input_b": [4.0, 5.0, 6.0]}

        result = engine.transform_dict(odfv, input_dict)
        # Result may be a dict or DataFrame depending on the ODFV mode
        if isinstance(result, pd.DataFrame):
            assert "output_sum" in result.columns
            assert list(result["output_sum"]) == [5.0, 7.0, 9.0]
        else:
            assert "output_sum" in result
            assert list(result["output_sum"]) == [5.0, 7.0, 9.0]

    def test_should_write_online(self):
        """Test _should_write_online with new online flag."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        odfv_online = _create_odfv(online=True)
        odfv_offline = _create_odfv(online=False)

        assert engine._should_write_online(odfv_online) is True
        assert engine._should_write_online(odfv_offline) is False

    def test_should_write_offline(self):
        """Test _should_write_offline with new offline flag."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        odfv_offline = _create_odfv(offline=True)
        odfv_online_only = _create_odfv(offline=False)

        assert engine._should_write_offline(odfv_offline) is True
        assert engine._should_write_offline(odfv_online_only) is False

    def test_has_udf(self):
        """Test _has_udf correctly detects UDF presence."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        odfv = _create_odfv()
        assert engine._has_udf(odfv) is True

    def test_find_dependent_odfvs_no_deps(self):
        """Test _find_dependent_odfvs with no dependencies."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        registry = MagicMock()
        registry.list_on_demand_feature_views.return_value = []
        registry.list_feature_views.return_value = []
        registry.list_stream_feature_views.return_value = []

        result = engine._find_dependent_odfvs("test_push", registry, "test_project")
        assert result == []

    def test_compute_from_push_no_deps(self):
        """Test compute_from_push with no dependent ODFVs."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
        )

        registry = MagicMock()
        registry.list_on_demand_feature_views.return_value = []
        registry.list_feature_views.return_value = []
        registry.list_stream_feature_views.return_value = []

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = engine.compute_from_push("test_push", df, registry, "test_project")
        assert result == {}

    def test_build_push_execution_plan_online(self):
        """Test execution plan includes online write node when online=True."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
            backend="pandas",
        )

        odfv = _create_odfv(online=True)
        input_table = pa.table({"input_a": [1], "input_b": [2.0]})

        plan = engine._build_push_execution_plan(odfv, input_table)
        node_names = [n.name for n in plan.nodes]

        assert f"{odfv.name}_input" in node_names
        assert f"{odfv.name}_transform" in node_names
        assert f"{odfv.name}_online_write" in node_names
        assert f"{odfv.name}_offline_write" not in node_names

    def test_build_push_execution_plan_offline(self):
        """Test execution plan includes offline write node when offline=True."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
            backend="pandas",
        )

        odfv = _create_odfv(offline=True)
        input_table = pa.table({"input_a": [1], "input_b": [2.0]})

        plan = engine._build_push_execution_plan(odfv, input_table)
        node_names = [n.name for n in plan.nodes]

        assert f"{odfv.name}_input" in node_names
        assert f"{odfv.name}_transform" in node_names
        assert f"{odfv.name}_offline_write" in node_names
        assert f"{odfv.name}_online_write" not in node_names

    def test_build_push_execution_plan_both(self):
        """Test execution plan includes both write nodes when both flags set."""
        engine = RealTimeComputeEngine(
            repo_config=MagicMock(),
            online_store=MagicMock(),
            backend="pandas",
        )

        odfv = _create_odfv(online=True, offline=True)
        input_table = pa.table({"input_a": [1], "input_b": [2.0]})

        plan = engine._build_push_execution_plan(odfv, input_table)
        node_names = [n.name for n in plan.nodes]

        assert f"{odfv.name}_online_write" in node_names
        assert f"{odfv.name}_offline_write" in node_names

