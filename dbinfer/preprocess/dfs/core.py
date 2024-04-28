# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import abc
from typing import Tuple, Dict, Optional, List, Union
import pandas as pd
import logging
import numpy as np
import tqdm
import pydantic
import featuretools as ft
import pprint
from featuretools.primitives import AggregationPrimitive
from enum import Enum
from collections import defaultdict
import copy
from dbinfer_bench import (
    DBBRDBDataset,
    DBBTableSchema,
    DBBColumnSchema,
    DBBColumnDType,
)

from . import primitives as prim
from .primitives import (
    Concat,
    Join,
    ArrayMax,
    ArrayMin,
    ArrayMean,
)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

_DFS_ENGINE_REGISTRY = {}


def base_feature_is_key(feature, keys):
    if isinstance(feature, (ft.AggregationFeature, ft.DirectFeature)):
        return base_feature_is_key(feature.base_features[0], keys)
    elif isinstance(feature, ft.IdentityFeature):
        return (feature.dataframe_name, feature.get_name()) in keys
    else:
        raise NotImplementedError(f'Unsupported subfeature {feature}')

def base_feature_is_target(feature, target):
    if isinstance(feature, (ft.AggregationFeature, ft.DirectFeature)):
        return base_feature_is_target(feature.base_features[0], target)
    elif isinstance(feature, ft.IdentityFeature):
        return (feature.dataframe_name, feature.get_name()) == target
    else:
        raise NotImplementedError(f'Unsupported subfeature {feature}')

def dfs_engine(preprocess_class):
    global _DFS_ENGINE_REGISTRY
    _DFS_ENGINE_REGISTRY[preprocess_class.name] = preprocess_class
    return preprocess_class


def get_dfs_engine(name: str):
    global _DFS_ENGINE_REGISTRY
    preprocess_class = _DFS_ENGINE_REGISTRY.get(name, None)
    if preprocess_class is None:
        raise ValueError(f"Cannot find the dfs engine of name {name}.")
    return preprocess_class


def get_dfs_engine_choice():
    global _DFS_ENGINE_REGISTRY
    names = _DFS_ENGINE_REGISTRY.keys()
    return Enum("DFSEngineChoice", {name.upper(): name for name in names})


class DFSConfig(pydantic.BaseModel):
    agg_primitives: List[str] = [
        "max",
        "min",
        "mean",
        "count",
        "mode",
        "concat",
        "join",
        "arraymax",
        "arraymin",
        "arraymean",
    ]
    max_depth: int = 1
    use_cutoff_time: bool = False
    engine: str = "featuretools"

class EntitySetBuilder:

    def __init__(self, entity_set : ft.EntitySet):
        self.entity_set = entity_set
        self.cutoff_time = None
        self.relationships = []

    def add_dataframe(
        self,
        dataframe_name : str,
        dataframe : pd.DataFrame,
        index : str,
        time_index : Optional[str] = None,
        logical_types : Optional[Dict[str, str]] = None,
        semantic_tags : Optional[Dict[str, str]] = None,
    ):
        self.entity_set.add_dataframe(
            dataframe_name=dataframe_name,
            dataframe=dataframe,
            index=index,
            time_index=time_index,
            logical_types=logical_types,
            semantic_tags=semantic_tags,
        )

    def add_relationship(
        self,
        pk_table : str,
        pk_column : str,
        fk_table : str,
        fk_column : str,
    ):
        self.relationships.append((pk_table, pk_column, fk_table, fk_column))
        self.entity_set.add_relationship(pk_table, pk_column, fk_table, fk_column)

    def set_cutoff_time(
        self,
        cutoff_time : Optional[pd.DataFrame]
    ):
        self.cutoff_time = cutoff_time

    def set_task_index(
        self,
        name: str,
        index : pd.Series
    ):
        pass


class DFSEngine:

    config_class = DFSConfig

    def __init__(self, config: DFSConfig, dataset: DBBRDBDataset, task_id: int):
        self.config = config
        self.dataset = dataset
        self.task_id = task_id
        self.agg_primitives = []
        for prim in config.agg_primitives:
            if prim == "concat":
                self.agg_primitives.append(Concat)
            elif prim == "join":
                self.agg_primitives.append(Join)
            elif prim == "arraymax":
                self.agg_primitives.append(ArrayMax)
            elif prim == "arraymin":
                self.agg_primitives.append(ArrayMin)
            elif prim == "arraymean":
                self.agg_primitives.append(ArrayMean)
            else:
                self.agg_primitives.append(prim)

    def run(self) -> Tuple[pd.DataFrame, Dict[str, ft.FeatureBase]]:
        features = self.prepare()
        if len(features) == 0:
            raise RuntimeError("No feature to compute, try to increase the depth.")
        else:
            logger.debug(f"Features to compute: {pprint.pformat(features)}")
            feature_df = self.compute(features)
            logger.debug(f"Generated dataframe: {feature_df.columns}")
        feature_map = {feat.get_name(): feat for feat in features}
        assert set(feature_map) >= set(feature_df.columns.tolist())
        return feature_df, feature_map

    def prepare(self):
        entity_set = ft.EntitySet(
            self.dataset.metadata.dataset_name
            + "-"
            + self.dataset.tasks[self.task_id].metadata.name
        )

        builder = EntitySetBuilder(entity_set)
        self.build_dataframes(builder, full_data=False)
        entity_set = builder.entity_set
        cutoff_time = builder.cutoff_time
        logger.debug(entity_set)

        features = ft.dfs(
            entityset=entity_set,
            target_dataframe_name="__task__",
            cutoff_time=cutoff_time,
            include_cutoff_time=False,
            max_depth=self.config.max_depth,
            agg_primitives=self.agg_primitives,
            trans_primitives=[],
            return_types="all",
            features_only=True,
        )

        features = self.filter_features(features, builder)

        return features

    @abc.abstractmethod
    def compute(
        self,
        features: List[ft.FeatureBase],
    ) -> pd.DataFrame:
        raise NotImplementedError

    def filter_features(
        self,
        features: List[ft.FeatureBase],
        builder : EntitySetBuilder,
    ) -> List[ft.FeatureBase]:
        if len(features) == 0:
            return features

        # Get the set of foreign/primary keys.
        keys = set()
        for pk_tbl, pk_col, fk_tbl, fk_col in builder.relationships:
            keys.add((pk_tbl, pk_col))
            keys.add((fk_tbl, fk_col))
        # Get the target label.
        task = self.dataset.tasks[self.task_id]
        use_cutoff_time = (
            self.config.use_cutoff_time
            and task.metadata.time_column is not None
        )

        new_features = []
        for feat in features:
            feat_str = str(feat)
            if "__task__" in feat_str:
                # Remove features involving the task table.
                continue
            if base_feature_is_key(feat, keys):
                continue
            if not use_cutoff_time:
                # Filter out features based on prediction target when cutoff time
                # is disabled.
                target = (task.metadata.target_table, task.metadata.target_column)
                if base_feature_is_target(feat, target):
                    continue
            new_features.append(feat)
        return new_features

    def build_dataframes(
        self,
        entity_set_builder : EntitySetBuilder,
        full_data : bool = True
    ):
        task = self.dataset.tasks[self.task_id]
        target_table = task.metadata.target_table
        for col_schema in task.metadata.columns:
            col_name = col_schema.name
            col_data = task.train_set[col_name]
            if col_schema.dtype == DBBColumnDType.primary_key and len(col_data) != len(
                np.unique(col_data)
            ):
                # TODO(minjie): This is a hack. If a column in task table
                # is marked as primary key but has duplicate values, it
                # is actually a foreign key and that task table conceptually
                # refers to a non-existing target table. This requires more
                # comprehensive fix in the future. The current hack is to set
                # the target_table to be None and only handle it in DFS.
                logger.warning(
                    "Setting target table to None because the primary key in the"
                    " task table contains duplicate values."
                )
                target_table = None
                break

        task_df_name = "__task__"

        fk_to_pk = {}
        selfloop_fk_to_pk = {}
        for rel in self.dataset.metadata.relationships:
            if rel.fk.table == rel.pk.table:
                # Self-loop relationship.
                selfloop_fk_to_pk[(rel.fk.table, rel.fk.column)] = (rel.pk.table, rel.pk.column)
            else:
                fk_to_pk[(rel.fk.table, rel.fk.column)] = (rel.pk.table, rel.pk.column)

        # Remove self-loop relationships.
        table_schemas, tables, new_fk_to_pk = self.remove_selfloop(selfloop_fk_to_pk)
        fk_to_pk.update(new_fk_to_pk)

        # Add dataframes
        for table_schema in table_schemas:
            table_name = table_schema.name
            logger.debug(f' Add table "{table_name}".')
            df = pd.DataFrame()
            logical_types = {}
            semantic_tags = {}
            index = None
            for col_schema in table_schema.columns:
                col_name = col_schema.name
                col_data = tables[table_name][col_name]
                logger.debug(f'   Parse column "{col_name}".')
                if not full_data:
                    col_data = col_data[:10]
                series, log_ty, tag = parse_one_column(col_schema, col_data)
                df[col_name] = series
                logical_types[col_name] = log_ty
                if tag == "index":
                    index = col_name
                else:
                    # NOTE: Featuretools does not allow setting "index"
                    # semantic tags.
                    semantic_tags[col_name] = tag
                if (
                    col_schema.dtype == DBBColumnDType.foreign_key
                    and fk_to_pk[(table_name, col_name)][0] == target_table
                ):
                    # The table has foreign key referring to the primary key of
                    # the *target* table. As the task table is conceptually part
                    # of the target table, we add additional foreign key to
                    # refer to the primary key of the *task* table.
                    target_column = fk_to_pk[(table_name, col_name)][1]
                    extra_col_name = f"__extra_fk_{col_name}"
                    logger.debug(f'   Add extra fk column "{extra_col_name}".')
                    fk_to_pk[(table_name, extra_col_name)] = (
                        task_df_name,
                        target_column,
                    )
                    df[extra_col_name] = series
                    logical_types[extra_col_name] = log_ty
                    semantic_tags[extra_col_name] = tag
            if index is None:
                # Featuretools requires a default index.
                df["__index__"] = np.arange(len(df))
                index = "__index__"
            entity_set_builder.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                logical_types=logical_types,
                semantic_tags=semantic_tags,
                index=index,
                time_index=table_schema.time_column,
            )

        # Add task dataframe
        logger.debug(f' Add task table.')
        task_df = pd.DataFrame()
        logical_types = {}
        semantic_tags = {}
        index = None
        for col_schema in task.metadata.columns:
            col_name = col_schema.name
            col_dtype = col_schema.dtype
            if col_dtype in [DBBColumnDType.primary_key, DBBColumnDType.foreign_key]:
                # NOTE: Only add key columns. Other columns will not be used in DFS.
                col_data = np.concatenate(
                    [
                        task.train_set[col_name],
                        task.validation_set[col_name],
                        task.test_set[col_name],
                    ]
                )
                if not full_data:
                    col_data = col_data[:10]
                series, log_ty, tag = parse_one_column(col_schema, col_data)
                if col_dtype == DBBColumnDType.primary_key and target_table is None:
                    # TODO(minjie): This is a hack. If a column in task table
                    # is marked as primary key but has duplicate values, it
                    # is actually a foreign key and that task table conceptually
                    # refers to a non-existing target table. This requires more
                    # comprehensive fix in the future. Here, we force it to be foreign key,
                    # and **GUESS** its foreign key to be the primary key of the
                    # current target table.
                    log_ty = "Categorical"
                    tag = "foreign_key"
                    pk = (task.metadata.target_table, col_name)
                    fk_to_pk[(task_df_name, col_name)] = pk

                elif col_dtype == DBBColumnDType.foreign_key:
                    pk = fk_to_pk[(target_table, col_name)]
                    fk_to_pk[(task_df_name, col_name)] = pk

                task_df[col_name] = series
                logical_types[col_name] = log_ty
                if tag == "index":
                    index = col_name
                else:
                    # NOTE: Featuretools does not allow setting "index"
                    # semantic tags.
                    semantic_tags[col_name] = tag

        if index is None:
            # Featuretools requires a default index.
            task_df["__index__"] = np.arange(len(task_df))
            index = "__index__"

        entity_set_builder.add_dataframe(
            dataframe_name=task_df_name,
            dataframe=task_df,
            logical_types=logical_types,
            semantic_tags=semantic_tags,
            index=index,
        )

        entity_set_builder.set_task_index(index, task_df[index])

        # Add relationships
        for (fk_tbl, fk_col), (pk_tbl, pk_col) in fk_to_pk.items():
            entity_set_builder.add_relationship(pk_tbl, pk_col, fk_tbl, fk_col)

        # Calculate cutoff-time dataframe
        cutoff_time = None
        if self.config.use_cutoff_time and task.metadata.time_column is not None:
            col_name = task.metadata.time_column
            col_data = np.concatenate(
                [
                    task.train_set[col_name],
                    task.validation_set[col_name],
                    task.test_set[col_name],
                ]
            )
            if not full_data:
                col_data = col_data[:10]
            time = pd.to_datetime(pd.Series(col_data, copy=False))
            cutoff_time = pd.DataFrame(
                {
                    # NOTE: featuretools will modify task_df's index **IN-PLACE** in
                    # add_dataframe().  This will misalign the values in `time` and
                    # `task_df` if we continue to use pandas Series for dataframe
                    # creation, as pandas will do an outer join on the indices
                    # by default.  To workaround this, we use the numpy array form
                    # directly.
                    index : task_df[index].values,
                    "time": time.values,
                }
            )
        entity_set_builder.set_cutoff_time(cutoff_time)

    def remove_selfloop(
        self,
        selfloop_fk_to_pk: Dict[Tuple[str, str], Tuple[str, str]],
    ):
        # Find tables with self-loop relationships.
        selfloop_tables = defaultdict(list)
        for (fk_tbl, fk_col) in selfloop_fk_to_pk.keys():
            selfloop_tables[fk_tbl].append(fk_col)
        
        new_fk_to_pk = {}
        table_schemas = []
        table_data = {}
        for table_schema in self.dataset.metadata.tables:
            table_name = table_schema.name
            if table_name not in selfloop_tables:
                table_schemas.append(table_schema)
                table_data[table_name] = self.dataset.tables[table_name]
                continue

            # Create extra table and schema.
            selfloop_table_name = f"{table_name}_selfloop"
            selfloop_table_schema = DBBTableSchema(
                name=selfloop_table_name,
                source="", format="numpy",
                columns=[], time_column=None)
            logger.debug(f"Create a new table `{selfloop_table_name}` to remove self-loop "
                         f"relationship in table `{table_name}`.")
            pk_col = selfloop_fk_to_pk[(table_name, selfloop_tables[table_name][0])][1]
            new_col_names = selfloop_tables[table_name] + [pk_col]
            for col_schema in table_schema.columns:
                if col_schema.name not in new_col_names:
                    continue
                col_schema = copy.deepcopy(col_schema)
                col_schema.dtype = DBBColumnDType.foreign_key
                selfloop_table_schema.columns.append(col_schema)
            selfloop_table_data = {
                col: self.dataset.tables[table_name][col] for col in new_col_names
            }

            # Modify the original table.
            # Remove keys that form selfloops.
            orig_table_data = dict(self.dataset.tables[table_name])
            orig_table_schema = copy.deepcopy(table_schema)
            orig_table_schema.columns = [
                col_schema
                for col_schema in orig_table_schema.columns
                if col_schema.name not in selfloop_tables[table_name]
            ]
            for fk_col in selfloop_tables[table_name]:
                orig_table_data.pop(fk_col)

            # Save.
            table_schemas += [selfloop_table_schema, orig_table_schema]
            table_data[selfloop_table_name] = selfloop_table_data
            table_data[table_name] = orig_table_data
            # Add new relationships.
            for col in new_col_names:
                new_fk_to_pk[(selfloop_table_name, col)] = (table_name, pk_col)
        
        return table_schemas, table_data, new_fk_to_pk


def parse_one_column(
    col_schema: DBBColumnSchema, col_data: np.ndarray
) -> Tuple[pd.Series, str, str]:
    if col_schema.dtype == DBBColumnDType.category_t:
        series = pd.Series(col_data, copy=False)
        log_ty = "Categorical"
        tag = "category"
    elif col_schema.dtype == DBBColumnDType.float_t:
        if col_data.ndim > 1:
            series = pd.Series(list(col_data))
            log_ty = "Array"
            tag = "array"
        else:
            series = pd.Series(col_data, copy=False)
            log_ty = "Double"
            tag = "numeric"
    elif col_schema.dtype == DBBColumnDType.datetime_t:
        series = pd.Series(col_data, copy=False)
        log_ty = "Datetime"
        tag = "string"
    elif col_schema.dtype == DBBColumnDType.text_t:
        series = pd.Series(col_data, copy=False)
        log_ty = "Text"
        tag = "text"
    elif col_schema.dtype == DBBColumnDType.primary_key:
        series = pd.Series(col_data, copy=False)
        log_ty = "Categorical"
        tag = "index"
    elif col_schema.dtype == DBBColumnDType.foreign_key:
        series = pd.Series(col_data, copy=False)
        log_ty = "Categorical"
        tag = "foreign_key"
    else:
        raise ValueError(f"Unsupported dtype {col_schema.dtype}.")
    return series, log_ty, tag
