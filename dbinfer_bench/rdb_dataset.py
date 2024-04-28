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


from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
import os
import numpy as np
import sqlalchemy
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    ForeignKey,
    Uuid,
    Float,
    ARRAY,
    VARCHAR,
    DateTime
)

from . import yaml_utils
from .download import download_or_get_path
from .dataset_meta import (
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBTaskMeta,
    DBBColumnDType,
    DBBColumnSchema,
    DBBTableSchema,
    DBBTableDataFormat,
    DBBColumnID,
    DBBRelationship,
    DBBRDBDatasetMeta,
)
from .table_loader import get_table_data_loader
from .table_writer import get_table_data_writer

__all__ = ['DBBRDBTask', 'DBBRDBDataset', 'DBBRDBTaskCreator',
           'DBBRDBDatasetCreator', 'load_rdb_data']

@dataclass
class DBBRDBTask:
    metadata : DBBTaskMeta
    train_set : Dict[str, np.ndarray]
    validation_set : Dict[str, np.ndarray]
    test_set : Dict[str, np.ndarray]

class DBBRDBDataset:

    def __init__(
        self,
        path : Path
    ):
        self.path = Path(path)
        self._metadata = self._load_metadata()
        self._load_data()

    def _load_metadata(self):
        return yaml_utils.load_pyd(DBBRDBDatasetMeta, self.path / 'metadata.yaml')

    def _load_data(self):
        # Load tables.
        self._tables = {}
        for table_schema in self.metadata.tables:
            table_path = self.path / table_schema.source
            loader = get_table_data_loader(table_schema.format)
            self._tables[table_schema.name] = loader(table_path)

        # Load tasks.
        self._tasks = []
        for task_meta in self.metadata.tasks:
            loader = get_table_data_loader(task_meta.format)
            def _load_split(split):
                table_path = self.path / task_meta.source.format(split=split)
                return loader(table_path)
            train_set = _load_split('train')
            validation_set = _load_split('validation')
            test_set = _load_split('test')
            self._tasks.append(DBBRDBTask(
                task_meta, train_set, validation_set, test_set))

    @property
    def dataset_name(self) -> str:
        return self.metadata.dataset_name

    @property
    def metadata(self) -> DBBRDBDatasetMeta:
        return self._metadata

    @property
    def tasks(self) -> List[DBBRDBTask]:
        return self._tasks

    @property
    def tables(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._tables

    def get_task(self, name : str) -> DBBRDBTask:
        for task in self.tasks:
            if task.metadata.name == name:
                return task
        raise ValueError(f"Unknown task {name}.")

    @property
    def sqlalchemy_metadata(self) -> sqlalchemy.MetaData:
        """Get metadata in sqlalchemy structure."""
        metadata = MetaData()
        pks, referred_pks = {}, {}
        for tbl_meta in self.metadata.tables:
            tbl_name = tbl_meta.name
            cols = []
            for col_meta in tbl_meta.columns:
                col_name = col_meta.name
                col_data = self.tables[tbl_name][col_name]
                if col_meta.dtype == DBBColumnDType.float_t:
                    if (col_data.shape) == 1:
                        col = Column(col_name, Float)
                    else:
                        col = Column(col_name, ARRAY(Float))
                elif col_meta.dtype == DBBColumnDType.category_t:
                    col = Column(col_name, VARCHAR)
                elif col_meta.dtype == DBBColumnDType.datetime_t:
                    col = Column(col_name, DateTime)
                elif col_meta.dtype == DBBColumnDType.text_t:
                    col = Column(col_name, String)
                elif col_meta.dtype == DBBColumnDType.foreign_key:
                    col = Column(col_name, None, ForeignKey(col_meta.link_to))
                    link_tbl, link_col = col_meta.link_to.split('.')
                    referred_pks[link_tbl] = link_col
                elif col_meta.dtype == DBBColumnDType.primary_key:
                    col = Column(col_name, Uuid, primary_key=True)
                    pks[tbl_name] = col_name
                else:
                    col = Column(col_name, VARCHAR)
                cols.append(col)
            alchemy_tbl = Table(tbl_name, metadata, *cols)
        # Create missing tables.
        for tbl, col in referred_pks.items():
            if tbl not in pks:
                alchemy_tbl = Table(tbl, metadata, Column(col, Uuid, primary_key=True))
            elif col != pks[tbl]:
                raise ValueError(f"Detect two primary keys ({col} and {pks[tbl]}) for table '{tbl}'!")

        return metadata

    def save(self, path : Path):
        ds_ctor = DBBRDBDatasetCreator(self.metadata.dataset_name)
        ds_ctor.replace_tables_from(self)
        for task in self.tasks:
            task_ctor = DBBRDBTaskCreator(task.metadata.name)
            task_ctor.copy_fields_from(task.metadata)
            task.metadata.column_dict
            for col_name in task.train_set:
                col_meta = dict(task.metadata.column_dict[col_name])
                col_meta.pop('name')
                task_ctor.add_task_data(
                    col_name,
                    task.train_set[col_name],
                    task.validation_set[col_name],
                    task.test_set[col_name],
                    **col_meta)
            if task.metadata.task_type == DBBTaskType.retrieval:
                for col_name in [
                    task.metadata.key_prediction_label_column,
                    task.metadata.key_prediction_query_idx_column,
                ]:
                    task_ctor.add_task_data(
                        col_name,
                        None,
                        task.validation_set[col_name],
                        task.test_set[col_name],
                        dtype=None)
            ds_ctor.add_task(task_ctor)
        ds_ctor.done(path)

def load_rdb_data(name_or_path : str) -> DBBRDBDataset:
    path = download_or_get_path(name_or_path)
    return DBBRDBDataset(path)

class DBBRDBTaskCreator:

    def __init__(self, name : str):
        self.task_fields = {
            'name' : name,
            'columns' : {},
        }

    def set_task_type(self, task_type : DBBTaskType):
        return self.add_task_field("task_type", task_type)

    def set_evaluation_metric(self, metric : DBBTaskEvalMetric):
        return self.add_task_field("evaluation_metric", metric)

    def set_target_table(self, tbl : str):
        return self.add_task_field("target_table", tbl)

    def set_target_column(self, col : str):
        return self.add_task_field("target_column", col)

    def set_target_time_column(self, col : str):
        return self.add_task_field("time_column", col)

    def set_key_prediction_label_column(self, col : str):
        return self.add_task_field("key_prediction_label_column", col)

    def set_key_prediction_query_idx_column(self, col : str):
        return self.add_task_field("key_prediction_query_idx_column", col)

    def add_task_field(self, key : str, val : Any):
        self.task_fields[key] = val
        return self

    def add_task_data(
        self,
        name : str,
        train_data : Optional[np.ndarray],
        validation_data : Optional[np.ndarray],
        test_data : Optional[np.ndarray],
        dtype : DBBColumnDType,
        **extra_meta
    ):
        assert train_data is not None or validation_data is not None or test_data is not None
        self.task_fields['columns'][name] = {
            'name' : name,
            'data' : (train_data, validation_data, test_data),
            'dtype' : dtype,
        }
        self.task_fields['columns'][name].update(extra_meta)
        return self

    def copy_fields_from(self, task_meta : DBBTaskMeta):
        task_meta_dict = task_meta.dict()
        self.task_fields = task_meta_dict
        self.task_fields['columns'] = {}
        return self

    def done(
        self,
        path: Path,
        table_format: DBBTableDataFormat = DBBTableDataFormat.NUMPY
    ) -> DBBTaskMeta:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"Provided path {path} must be a directory.")
        task_path = path / self.task_fields['name']
        task_path.mkdir(parents=True, exist_ok=True)
        # Save task table.
        train_table = {}
        val_table = {}
        test_table = {}
        col_schemas = []
        for key, col_schema in self.task_fields['columns'].items():
            train_data, val_data, test_data = col_schema.pop('data')
            if train_data is not None:
                train_table[key] = train_data
            if val_data is not None:
                val_table[key] = val_data
            if test_data is not None:
                test_table[key] = test_data

            if train_data is not None:
                # NOTE: Only write schema of training columns. Skip val/test -only
                #   columns such as `key_prediction_label_column` and
                #   `key_prediction_query_idx_column` used by retrieval tasks.
                col_schemas.append(col_schema)
        table_writer = get_table_data_writer(table_format)
        table_writer.write(task_path, "train", train_table)
        table_writer.write(task_path, "validation", val_table)
        table_writer.write(task_path, "test", test_table)
        task_meta = dict(self.task_fields)
        task_meta['format'] = table_format
        task_meta['source'] = str(table_writer.filename(task_path, "{split}").relative_to(path))
        task_meta['columns'] = col_schemas
        return DBBTaskMeta(**task_meta)

class DBBRDBDatasetCreator:

    def __init__(self, name : str):
        self.name = name
        self.tasks = []
        self.tables = {}

    def add_table(self, table_name : str):
        if table_name in self.tables:
            raise ValueError(f"Table {table_name} has already been added.")
        self.tables[table_name] = {"columns" : {}}
        return self

    def add_column(
        self,
        table_name : str,
        column_name : str,
        data : np.ndarray,
        dtype : DBBColumnDType,
        **extra_meta
    ):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist. Please add_table first.")

        if column_name in self.tables[table_name]:
            raise ValueError(f"Column {column_name} already exists.")

        self.tables[table_name]["columns"][column_name] = {
            'name' : column_name,
            'data' : data,
            'dtype' : dtype,
        }
        self.tables[table_name]["columns"][column_name].update(extra_meta)

        return self

    def set_time_column(self, table: str, time_col: str):
        other_time_col = self.tables[table].get("time_column")
        if other_time_col is not None and time_col != other_time_col:
            raise ValueError(f"A table can only have one time column but got {time_col} and {other_time_col}.")
        self.tables[table]["time_column"] = time_col

    def add_task(
        self,
        task_creator : DBBRDBTaskCreator
    ):
        self.tasks.append(task_creator)
        return self

    def add_column_group(
        self,
        col_group : List[Tuple[str, str]]
    ):
        if self.column_groups is None:
            self.column_groups = []
        col_group = [DBBColumnID(table=tbl, column=col) for tbl, col in col_group]
        self.column_groups.append(col_group)
        return self

    def replace_tables_from(
        self,
        other : DBBRDBDataset
    ):
        self.tables = {}
        for table_schema in other.metadata.tables:
            table_name = table_schema.name
            self.add_table(table_name)
            self.set_time_column(table_name, table_schema.time_column)
            for col_schema in table_schema.columns:
                col_name = col_schema.name
                col_schema = col_schema.dict()
                self.add_column(
                    table_name,
                    col_name,
                    other.tables[table_name][col_name],
                    **col_schema
                )
        return self

    def _validate(self):
        for table_name, table_info in self.tables.items():
            table_size = None
            for col_name, col_info in table_info["columns"].items():
                if table_size is None:
                    table_size = len(col_info['data'])
                elif len(col_info['data']) != table_size:
                    raise ValueError(
                        f"Expect all columns to have the same length."
                        f" But got {col_info['data']} and {table_size}."
                    )

    def done(
        self,
        path: Path,
        table_format: DBBTableDataFormat = DBBTableDataFormat.NUMPY
    ):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"Provided path {path} must be a directory.")

        self._validate()

        # Write task data.
        tasks = [task_ctor.done(path, table_format) for task_ctor in self.tasks]

        # Write table data.
        table_writer = get_table_data_writer(table_format)
        schemas = []
        for table_name, table_info in self.tables.items():
            data_dir = path / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            table_data = {}
            col_schemas = []
            for col_name, col_info in table_info["columns"].items():
                table_data[col_name] = col_info.pop("data")
                col_schemas.append(col_info)
            table_writer.write(data_dir, table_name, table_data)
            source = str(table_writer.filename(data_dir, table_name).relative_to(path))
            schema = DBBTableSchema.parse_obj({
                'name' : table_name,
                'source' : source,
                'format' : table_format,
                'columns' : col_schemas,
                'time_column' : table_info.get('time_column')
            })
            schemas.append(schema)

        metadata = DBBRDBDatasetMeta(
            dataset_name=self.name,
            tables=schemas,
            tasks=tasks,
        )

        yaml_utils.save_pyd(metadata, path / 'metadata.yaml')
