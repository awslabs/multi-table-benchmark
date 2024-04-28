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


from enum import Enum

class GraphConstructionChoice(str, Enum):
    r2ne = "r2ne"
    r2n = "r2n"

def get_graph_construction_class(graph_construction_name):
    if graph_construction_name == "r2ne":
        from .er_graph_construction import ERGraphConstruction
        graph_construction_class = ERGraphConstruction
    elif graph_construction_name == "r2n":
        from .rdb2graph import RDB2Graph
        graph_construction_class = RDB2Graph
    else:
        raise ValueError("Unknown graph construction name:", graph_construction_name)
    return graph_construction_class
