import logging
from typing import Dict, List, Literal, Optional

import pandas as pd
import pyarrow as pa

from feast.feature_view import FeatureView
from feast.infra.compute_engines.dag.context import ExecutionContext
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.plan import ExecutionPlan
from feast.infra.compute_engines.local.arrow_table_value import ArrowTableValue
from feast.infra.compute_engines.local.local_node import LocalNode
from feast.infra.online_stores.online_store import OnlineStore
from feast.repo_config import FeastConfigBaseModel, RepoConfig

logger = logging.getLogger(__name__)


class RealTimeComputeEngineConfig(FeastConfigBaseModel):
    """Configuration for RealTime Compute Engine."""

    type: Literal["local"] = "local"
    """RealTime Compute Engine type selector"""

    backend: str = "python"
    """Backend to use for DataFrame operations (e.g., 'python', 'pandas', 'polars')"""


# ---------------------------------------------------------------------------
# DAG Node building blocks
# ---------------------------------------------------------------------------


class DataInputNode(LocalNode):
    """A DAG node that provides pre-loaded data as input.

    This is the real-time equivalent of LocalSourceReadNode: instead of
    reading from an offline store, it wraps data that was pushed or
    fetched at request time.
    """

    def __init__(self, name: str, data: pa.Table):
        super().__init__(name)
        self._data = data

    def execute(self, context: ExecutionContext) -> ArrowTableValue:
        return ArrowTableValue(data=self._data)


class TransformationNode(LocalNode):
    """A DAG node that applies a FeatureView's transformation.

    Calls feature_view.transform_arrow() which dispatches to the
    appropriate mode (pandas, python, substrait, etc.).

    Once OnDemandFeatureView inherits from FeatureView, this can be
    replaced by the generic LocalTransformationNode.
    """

    def __init__(
        self,
        name: str,
        feature_view: FeatureView,
        inputs: Optional[List[DAGNode]] = None,
    ):
        super().__init__(name, inputs=inputs or [])
        self.feature_view = feature_view

    def execute(self, context: ExecutionContext) -> ArrowTableValue:
        input_table = self.get_single_table(context).data
        result_table = self.feature_view.transform_arrow(input_table)
        return ArrowTableValue(data=result_table)


# ---------------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------------


class RealTimeFeatureBuilder:
    """
    Translates a FeatureView + input data into an ExecutionPlan.

    This mirrors LocalFeatureBuilder's role:
    - LocalFeatureBuilder: builds DAG from offline store reads
    - RealTimeFeatureBuilder: builds DAG from pre-loaded input data

    Once OnDemandFeatureView inherits from FeatureView, this will
    extend FeatureBuilder and use FeatureResolver for dependency
    walking, gaining filter, join, aggregation, dedup, and validation
    steps automatically.
    """

    def __init__(
        self,
        feature_view: FeatureView,
        input_data: pa.Table,
    ):
        self.feature_view = feature_view
        self.input_data = input_data

    def build_source_node(self) -> DataInputNode:
        return DataInputNode(
            name=f"{self.feature_view.name}_source",
            data=self.input_data,
        )

    def build_transformation_node(
        self, input_node: DAGNode
    ) -> TransformationNode:
        return TransformationNode(
            name=f"{self.feature_view.name}_transform",
            feature_view=self.feature_view,
            inputs=[input_node],
        )

    def build(self) -> ExecutionPlan:
        """
        Build an ExecutionPlan for the FeatureView.

        Current pipeline: source -> transform
        Once this extends FeatureBuilder: source -> [join] -> transform
            -> [filter] -> [agg/dedup] -> [validate] -> [output]
        """
        source_node = self.build_source_node()
        transform_node = self.build_transformation_node(source_node)
        return ExecutionPlan(nodes=[source_node, transform_node])


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RealTimeComputeEngine:
    """
    RealTimeComputeEngine handles FeatureView computation in real-time,
    both on the read path (transform at request time) and the write
    path (materialize from push).

    This mirrors LocalComputeEngine's architecture:
    - LocalComputeEngine  -> LocalFeatureBuilder -> ExecutionPlan
    - RealTimeComputeEngine -> RealTimeFeatureBuilder -> ExecutionPlan
    """

    def __init__(
        self,
        *,
        repo_config: RepoConfig,
        online_store: OnlineStore,
        backend: str = "python",
    ):
        self.repo_config = repo_config
        self.online_store = online_store
        self._backend_name = backend

    def _get_execution_context(self, project: str = "") -> ExecutionContext:
        return ExecutionContext(
            project=project,
            repo_config=self.repo_config,
            offline_store=None,
            online_store=self.online_store,
            entity_defs=[],
        )

    def transform(
        self,
        feature_view: FeatureView,
        input_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute FeatureView transformation at serving time.

        Uses RealTimeFeatureBuilder to construct and execute a DAG,
        following the same pattern as LocalComputeEngine.

        Args:
            feature_view: The FeatureView to transform.
            input_df: DataFrame containing source features and request data.

        Returns:
            Transformed DataFrame with output features.
        """
        input_table = pa.Table.from_pandas(input_df)

        builder = RealTimeFeatureBuilder(feature_view, input_table)
        plan = builder.build()
        context = self._get_execution_context()
        result = plan.execute(context)
        return result.data.to_pandas()

    def transform_dict(
        self,
        feature_view: FeatureView,
        feature_dict: Dict[str, list],
    ) -> Dict[str, list]:
        """
        Execute FeatureView transformation on a dictionary of features.

        Args:
            feature_view: The FeatureView to transform.
            feature_dict: Dictionary of feature name -> values.

        Returns:
            Transformed dictionary with output features.
        """
        input_df = pd.DataFrame(feature_dict)
        result_df = self.transform(feature_view, input_df)
        return result_df.to_dict(orient="list")

    def materialize(
        self,
        feature_view: FeatureView,
        input_data: pa.Table,
        project: str,
    ) -> pa.Table:
        """
        Materialize features from push data to online/offline stores.

        This is the write-path counterpart to transform(). Once
        RealTimeFeatureBuilder extends FeatureBuilder, the build()
        output will include output nodes (reusing LocalOutputNode)
        that write to stores based on feature_view.online/offline.

        Args:
            feature_view: The FeatureView to materialize.
            input_data: Arrow table containing pushed data.
            project: The Feast project name.

        Returns:
            The transformed Arrow table.
        """
        raise NotImplementedError(
            "materialize() will be implemented when RealTimeFeatureBuilder "
            "extends FeatureBuilder with output node support."
        )
