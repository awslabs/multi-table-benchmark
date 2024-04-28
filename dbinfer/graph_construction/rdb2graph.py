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


import pydantic

from .er_graph_construction import (
    ERGraphConstruction,
    ERGraphConstructionConfig
)

class RDB2GraphConfig(pydantic.BaseModel):
    # Whether to construct a relation table as edges.
    # If not, all tables will be constructed as nodes.
    relation_table_as_edge : bool = False


class RDB2Graph(ERGraphConstruction):

    config_class = RDB2GraphConfig
    name = "r2n"
