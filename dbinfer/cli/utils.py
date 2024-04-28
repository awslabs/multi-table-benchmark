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
from pathlib import Path
import pandas as pd
import numpy as np
from uuid import UUID, uuid5

def degree_info(graph):
    deg = [
        graph.in_degrees(etype=etype)
        for etype in graph.canonical_etypes
    ]
    deg_quantile = [
        np.quantile(d, [0.2, 0.4, 0.6, 0.8, 1.0])
        for d in deg
    ]
    df = pd.DataFrame(
        {
            et : qt
            for (_, et, _), qt in zip(graph.canonical_etypes, deg_quantile)
        },
        index=["20%", "40%", "60%", "80%", "100%"]
    )
    return df.transpose()

_DBINFER_NAMESPACE_UUID = UUID('42138ced-2d20-4b0c-a4ab-b7b1646a7ba2')

def generate_uuid(*args, **kwargs):
    """
    Generate a UUID for the list of arguments and keyword arguments.
    """
    data = str([str(arg) for arg in args])
    data += str({str(key): str(value) for key, value in kwargs.items()})

    return uuid5(_DBINFER_NAMESPACE_UUID, data)
