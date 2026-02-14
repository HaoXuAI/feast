import logging
from typing import Dict, List, Literal, Optional, Set, Union

import pandas as pd
import pyarrow as pa

from feast.infra.compute_engines.backends.base import DataFrameBackend
from feast.infra.compute_engines.backends.factory import BackendFactory
from feast.infra.compute_engines.dag.context import ExecutionContext
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.plan import ExecutionPlan
from feast.infra.compute_engines.local.arrow_table_value import ArrowTableValue
from feast.infra.compute_engines.local.local_node import LocalNode
from feast.infra.online_stores.online_store import OnlineStore
from feast.infra.registry.base_registry import BaseRegistry
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.utils import _convert_arrow_to_proto

logger = logging.getLogger(__name__)


class RealTimeComputeEngineConfig(FeastConfigBaseModel):
    """Configuration for RealTime Compute Engine."""

    type: Literal["local"] = "local"
    """RealTime Compute Engine type selector"""

    backend: str = "pandas"
    """Backend to use for DataFrame operations (e.g., 'pandas', 'polars')"""


class DataInputNode(LocalNode):
    """A DAG node that provides pre-loaded data as input."""

    def __init__(self, name: str, data: pa.Table):
        super().__init__(name)
        self._data = data

    def execute(self, context: ExecutionContext) -> ArrowTableValue:
        return ArrowTableValue(data=self._data)


class ODFVTransformationNode(LocalNode):
    """A DAG node that executes an ODFV UDF-based transformation."""

    def __init__(
        self,
        name: str,
        odfv: "OnDemandFeatureView",  # noqa: F821
        backend: DataFrameBackend,
        inputs: Optional[List[DAGNode]] = None,
    ):
        super().__init__(name, inputs=inputs or [])
        self.odfv = odfv
        self.backend = backend

    def execute(self, context: ExecutionContext) -> ArrowTableValue:
        input_table = self.get_single_table(context).data
        transformed_table = self.odfv.transform_arrow(input_table)
        return ArrowTableValue(data=transformed_table)


class OnlineStoreWriteNode(LocalNode):
    """A DAG node that writes results to the online store."""

    def __init__(
        self,
        name: str,
        feature_view: "OnDemandFeatureView",  # noqa: F821
        inputs: Optional[List[DAGNode]] = None,
    ):
        super().__init__(name, inputs=inputs or [])
        self.feature_view = feature_view

    def execute(self, context: ExecutionContext) -> ArrowTableValue:
        input_table = self.get_single_table(context).data

        if input_table.num_rows == 0:
            return ArrowTableValue(data=input_table)

        online_store = context.online_store
        join_key_to_value_type = {
            entity.name: entity.dtype.to_value_type()
            for entity in self.feature_view.entity_columns
        }

        rows_to_write = _convert_arrow_to_proto(
            input_table, self.feature_view, join_key_to_value_type
        )

        online_store.online_write_batch(
            config=context.repo_config,
            table=self.feature_view,
            data=rows_to_write,
            progress=lambda x: None,
        )

        return ArrowTableValue(data=input_table)


class OfflineStoreWriteNode(LocalNode):
    """A DAG node that writes results to the offline store."""

    def __init__(
        self,
        name: str,
        feature_view: "OnDemandFeatureView",  # noqa: F821
        inputs: Optional[List[DAGNode]] = None,
    ):
        super().__init__(name, inputs=inputs or [])
        self.feature_view = feature_view

    def execute(self, context: ExecutionContext) -> ArrowTableValue:
        input_table = self.get_single_table(context).data

        if input_table.num_rows == 0:
            return ArrowTableValue(data=input_table)

        offline_store = context.offline_store
        offline_store.offline_write_batch(
            config=context.repo_config,
            feature_view=self.feature_view,
            table=input_table,
            progress=lambda x: None,
        )

        return ArrowTableValue(data=input_table)


class RealTimeComputeEngine:
    """
    RealTimeComputeEngine orchestrates ODFV (OnDemandFeatureView) transformations
    in real-time, both on the write path (push) and on the read path (get_online_features
    with transform=True).

    This engine reuses existing DAG infrastructure (ExecutionPlan, DAGNode, DataFrameBackend)
    to compose lightweight execution plans for ODFV computation.

    Key responsibilities:
    - compute_from_push(): When data is pushed, find dependent ODFVs, execute their
      transformations, and write results to online/offline stores based on ODFV config.
    - transform(): Execute ODFV transformations at serving time when transform=True.

    The DAG plan is: DataInput -> [UDFTransform] -> [Write]
    """

    def __init__(
        self,
        *,
        repo_config: RepoConfig,
        online_store: OnlineStore,
        backend: str = "pandas",
    ):
        self.repo_config = repo_config
        self.online_store = online_store
        self._backend = BackendFactory.from_name(backend)

    def _get_backend(self) -> DataFrameBackend:
        return self._backend

    def _find_dependent_odfvs(
        self,
        push_source_name: str,
        registry: BaseRegistry,
        project: str,
    ) -> List["OnDemandFeatureView"]:  # noqa: F821
        """
        Find all OnDemandFeatureViews that depend on feature views with the given
        push source.

        Args:
            push_source_name: Name of the push source that received data.
            registry: The feature registry.
            project: The Feast project name.

        Returns:
            List of dependent OnDemandFeatureViews.
        """
        from feast.on_demand_feature_view import OnDemandFeatureView

        all_odfvs = registry.list_on_demand_feature_views(project, allow_cache=True)
        all_fvs = registry.list_feature_views(project, allow_cache=True)
        all_sfvs = registry.list_stream_feature_views(project, allow_cache=True)

        # Find feature views that have this push source
        fv_names_with_push_source: Set[str] = set()
        from feast.data_source import PushSource

        for fv in list(all_fvs) + list(all_sfvs):
            if (
                fv.stream_source is not None
                and isinstance(fv.stream_source, PushSource)
                and fv.stream_source.name == push_source_name
            ):
                fv_names_with_push_source.add(fv.name)

        # Find ODFVs that reference any of these feature views
        dependent_odfvs: List[OnDemandFeatureView] = []
        for odfv in all_odfvs:
            for source_name in odfv.source_feature_view_projections:
                if source_name in fv_names_with_push_source:
                    dependent_odfvs.append(odfv)
                    break

        return dependent_odfvs

    def _has_udf(self, odfv: "OnDemandFeatureView") -> bool:  # noqa: F821
        """Check if ODFV has a UDF transformation defined."""
        return odfv.feature_transformation is not None

    def _build_push_execution_plan(
        self,
        odfv: "OnDemandFeatureView",  # noqa: F821
        input_data: pa.Table,
    ) -> ExecutionPlan:
        """
        Build an execution plan for computing an ODFV from pushed data and writing
        results to the appropriate stores.

        The plan is: DataInput -> [UDFTransform] -> [OnlineWrite, OfflineWrite]
        """
        backend = self._get_backend()

        # Node 1: Input data
        input_node = DataInputNode(
            name=f"{odfv.name}_input",
            data=input_data,
        )
        nodes: List[DAGNode] = [input_node]
        last_node: DAGNode = input_node

        # Node 2 (optional): UDF transformation
        if self._has_udf(odfv):
            transform_node = ODFVTransformationNode(
                name=f"{odfv.name}_transform",
                odfv=odfv,
                backend=backend,
                inputs=[last_node],
            )
            nodes.append(transform_node)
            last_node = transform_node

        # Node 4: Write to online store if online=True
        if self._should_write_online(odfv):
            online_write_node = OnlineStoreWriteNode(
                name=f"{odfv.name}_online_write",
                feature_view=odfv,
                inputs=[last_node],
            )
            nodes.append(online_write_node)

        # Node 5: Write to offline store if offline=True
        if self._should_write_offline(odfv):
            offline_write_node = OfflineStoreWriteNode(
                name=f"{odfv.name}_offline_write",
                feature_view=odfv,
                inputs=[last_node],
            )
            nodes.append(offline_write_node)

        return ExecutionPlan(nodes=nodes)

    def _should_write_online(self, odfv: "OnDemandFeatureView") -> bool:  # noqa: F821
        """Check if ODFV should write to online store."""
        # Support both new `online` flag and legacy `write_to_online_store`
        if hasattr(odfv, "online") and odfv.online is not None:
            return odfv.online
        return getattr(odfv, "write_to_online_store", False)

    def _should_write_offline(self, odfv: "OnDemandFeatureView") -> bool:  # noqa: F821
        """Check if ODFV should write to offline store."""
        return getattr(odfv, "offline", False)

    def compute_from_push(
        self,
        push_source_name: str,
        df: pd.DataFrame,
        registry: BaseRegistry,
        project: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process push-triggered computation: find dependent ODFVs, execute their
        transformations, and write results to the appropriate stores.

        Args:
            push_source_name: Name of the push source that received data.
            df: The pushed data as a DataFrame.
            registry: The feature registry.
            project: The Feast project name.

        Returns:
            Dictionary mapping ODFV names to their transformed DataFrames.
        """
        dependent_odfvs = self._find_dependent_odfvs(
            push_source_name, registry, project
        )

        if not dependent_odfvs:
            return {}

        input_table = pa.Table.from_pandas(df)
        results: Dict[str, pd.DataFrame] = {}

        for odfv in dependent_odfvs:
            if not self._should_write_online(odfv) and not self._should_write_offline(
                odfv
            ):
                continue

            try:
                plan = self._build_push_execution_plan(odfv, input_table)

                # Create a minimal execution context
                context = ExecutionContext(
                    project=project,
                    repo_config=self.repo_config,
                    offline_store=None,  # Will be set if needed
                    online_store=self.online_store,
                    entity_defs=[],
                )

                plan.execute(context)

                # Extract the transformed result from the transform node
                for node_name in [
                    f"{odfv.name}_transform",
                ]:
                    if node_name in context.node_outputs:
                        result_table = context.node_outputs[node_name]
                        if isinstance(result_table, ArrowTableValue):
                            results[odfv.name] = result_table.data.to_pandas()
                        elif hasattr(result_table, "data"):
                            results[odfv.name] = result_table.data.to_pandas()
                        break

                logger.info(
                    f"Successfully computed ODFV '{odfv.name}' from push to '{push_source_name}'"
                )

            except Exception as e:
                logger.error(
                    f"Failed to compute ODFV '{odfv.name}' from push to '{push_source_name}': {e}"
                )
                raise

        return results

    def transform(
        self,
        odfv: "OnDemandFeatureView",  # noqa: F821
        input_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute ODFV transformation at serving time (when transform=True in
        get_online_features). This applies UDF transformations without writing
        results to any store.

        Args:
            odfv: The OnDemandFeatureView to transform.
            input_df: DataFrame containing source features and request data.

        Returns:
            Transformed DataFrame with ODFV output features.
        """
        if self._has_udf(odfv):
            input_table = pa.Table.from_pandas(input_df)
            transformed_table = odfv.transform_arrow(input_table)
            return transformed_table.to_pandas()

        return input_df

    def transform_dict(
        self,
        odfv: "OnDemandFeatureView",  # noqa: F821
        feature_dict: Dict[str, list],
    ) -> Dict[str, list]:
        """
        Execute ODFV transformation on a dictionary of features.

        Args:
            odfv: The OnDemandFeatureView to transform.
            feature_dict: Dictionary of feature name -> values.

        Returns:
            Transformed dictionary with ODFV output features.
        """
        if self._has_udf(odfv):
            input_df = pd.DataFrame(feature_dict)
            result_df = self.transform(odfv, input_df)
            return result_df.to_dict(orient="list")

        return feature_dict
