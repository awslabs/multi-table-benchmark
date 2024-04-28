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


from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np

from .dataset_meta import (
    DBBTableDataFormat
)

def get_table_data_loader(format : DBBTableDataFormat):
    if format not in LOADER_MAP:
        raise ValueError(f"Unsupported table format: {format}")
    return LOADER_MAP[format]

def parquet_loader(path : str) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(str(path))
    return { col : df[col].to_numpy() for col in df }

def numpy_loader(path : str) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return { name : npz[name] for name in npz.files }

LOADER_MAP = {
    DBBTableDataFormat.PARQUET : parquet_loader,
    DBBTableDataFormat.NUMPY : numpy_loader,
}
