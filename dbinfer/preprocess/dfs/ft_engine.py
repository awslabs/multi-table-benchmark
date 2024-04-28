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


from typing import Optional, List
import pandas as pd
import featuretools as ft
import tqdm
import logging

from .core import DFSEngine, DFSConfig, dfs_engine, EntitySetBuilder

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

@dfs_engine
class FeaturetoolsEngine(DFSEngine):

    name = "featuretools"
    config = DFSConfig

    def compute(
        self, features: List[ft.FeatureBase],
    ) -> pd.DataFrame:
        entity_set = ft.EntitySet(
            self.dataset.metadata.dataset_name
            + "-"
            + self.dataset.tasks[self.task_id].metadata.name
        )

        builder = EntitySetBuilder(entity_set)
        self.build_dataframes(builder)
        entity_set = builder.entity_set
        cutoff_time = builder.cutoff_time
        logger.debug(entity_set)

        with tqdm.tqdm(total=100) as pbar:

            def _cb(update, progress_percent, time_elapsed):
                pbar.update(int(update))

            feature_df, feature_defs = ft.dfs(
                entityset=entity_set,
                target_dataframe_name="__task__",
                seed_features=features,
                cutoff_time=cutoff_time,
                include_cutoff_time=False,
                max_depth=self.config.max_depth,
                agg_primitives=self.agg_primitives,
                trans_primitives=[],
                return_types="all",
                n_jobs=1,
                progress_callback=_cb,
            )

        # Remove unexpected columns added to the task table, which is likely
        # due to bugs in featuretools.
        feat_names = set([feat.get_name() for feat in features])
        col_names = set(feature_df.columns)
        col_to_drop = col_names - feat_names
        feature_df = feature_df.drop(columns=col_to_drop, errors="ignore")

        return feature_df
