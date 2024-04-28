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


from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import os
import numpy as np
import abc
import pandas as pd

from .dataset_meta import DBBTableDataFormat

class TableWriter:

    @abc.abstractmethod
    def write(self, path: Path, table_name: str, table_data: Dict[str, np.ndarray]):
        """Write the table data."""
        raise NotImplementedError()

    @abc.abstractmethod
    def filename(self, path: Path, table_name: str) -> Path:
        """Return the on-disk filename that stores the table data."""
        raise NotImplementedError()

class ParquetTableWriter(TableWriter):

    def write(self, path: Path, table_name: str, table_data: Dict[str, np.ndarray]):
        filename = self.filename(path, table_name)
        df = pd.DataFrame(table_data)
        df.to_parquet(filename)

    def filename(self, path: Path, table_name: str) -> Path:
        return path / f"{table_name}.pqt"

class NumpyTableWriter(TableWriter):

    def write(self, path: Path, table_name: str, table_data: Dict[str, np.ndarray]):
        filename = self.filename(path, table_name)
        np.savez(filename, **table_data)

    def filename(self, path: Path, table_name: str) -> Path:
        return path / f"{table_name}.npz"

def get_table_data_writer(format : DBBTableDataFormat) -> TableWriter:
    if format not in WRITER_MAP:
        raise ValueError(f"Unsupported table format: {format}")
    return WRITER_MAP[format]()

WRITER_MAP = {
    DBBTableDataFormat.PARQUET : ParquetTableWriter,
    DBBTableDataFormat.NUMPY : NumpyTableWriter,
}
