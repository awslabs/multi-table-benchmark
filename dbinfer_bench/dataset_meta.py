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
from enum import Enum
import pydantic

__all__ = [
    "TIMESTAMP_FEATURE_NAME",
    "DBBColumnDType",
    "DTYPE_EXTRA_FIELDS",
    "DBBColumnSchema",
    "DBBTableDataFormat",
    "DBBTableSchema",
    "DBBTaskType",
    "TASK_EXTRA_FIELDS",
    "DBBTaskEvalMetric",
    "DBBTaskMeta",
    "DBBColumnID",
    "DBBRelationship",
    "DBBRDBDatasetMeta",
]

TIMESTAMP_FEATURE_NAME = '__timestamp__'

class DBBColumnDType(str, Enum):
    """Column data type model."""
    float_t = 'float'            # np.float32
    category_t = 'category'      # object
    datetime_t = 'datetime'      # np.datetime64
    text_t = 'text'              # str
    timestamp_t = 'timestamp'    # np.int64
    foreign_key = 'foreign_key'  # object
    primary_key = 'primary_key'  # object

DTYPE_EXTRA_FIELDS = {
    # in_size : An integer tells the size of the feature dimension.
    DBBColumnDType.float_t : ["in_size"],
    # num_categories : An integer tells the total number of categories.
    DBBColumnDType.category_t : ["num_categories"],
    # link_to : A string in the format of <TABLE>.<COLUMN>
    # capacity : The number of unique keys.
    DBBColumnDType.foreign_key : ["link_to", "capacity"],
    # capacity : The number of unique keys.
    DBBColumnDType.primary_key : ["capacity"],
}

class DBBColumnSchema(pydantic.BaseModel):
    """Column schema model.

    Column schema allows extra fields other than the explicitly defined members.
    See `DTYPE_EXTRA_FIELDS` dictionary for more details.
    """
    class Config:
        extra = pydantic.Extra.allow
        use_enum_values = True

    # Column name.
    name : str
    # Column data type.
    dtype : DBBColumnDType

class DBBTableDataFormat(str, Enum):
    PARQUET = 'parquet'
    NUMPY = 'numpy'

class DBBTableSchema(pydantic.BaseModel):
    """Table schema model."""

    # Name of the table.
    name : str
    # On-disk data path (relative to the root data folder) to load this table.
    source: str
    # On-disk format for storing this table.
    format: DBBTableDataFormat
    # Column schemas.
    columns: List[DBBColumnSchema]
    # Time column name.
    time_column: Optional[str] = None

    @property
    def column_dict(self) -> Dict[str, DBBColumnSchema]:
        """Get column schemas in a dictionary where the keys are column names."""
        return {col_schema.name : col_schema for col_schema in self.columns}

class DBBTaskType(str, Enum):
    classification = 'classification'
    regression = 'regression'
    retrieval = 'retrieval'

TASK_EXTRA_FIELDS = {
    DBBTaskType.classification : ['num_classes'],
    DBBTaskType.retrieval : [
        'key_prediction_label_column',
        'key_prediction_query_idx_column',
    ],
    DBBTaskType.regression : [],
}

class DBBTaskEvalMetric(str, Enum):
    auroc = 'auroc'
    ap = 'ap'
    accuracy = 'accuracy'
    f1 = 'f1'
    hinge = 'hinge'
    recall = 'recall'
    mae = 'mae'
    mse = 'mse'
    msle = 'msle'
    pearson = 'pearson'
    rmse = 'rmse'
    r2 = 'r2'
    mrr = 'mrr'
    hr = 'hr'
    ndcg = 'ndcg'

class DBBTaskMeta(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.allow
        use_enum_values = True

    name : str
    source : str
    format : DBBTableDataFormat
    columns : List[DBBColumnSchema]
    time_column : Optional[str] = None

    evaluation_metric : DBBTaskEvalMetric
    target_column : str
    target_table : str
    task_type : Optional[DBBTaskType] = None
    key_prediction_label_column: Optional[str] = "label"
    key_prediction_query_idx_column: Optional[str] = "query_idx"

    @property
    def column_dict(self) -> Dict[str, DBBColumnSchema]:
        return {col_schema.name : col_schema for col_schema in self.columns}

class DBBColumnID(pydantic.BaseModel):
    table : str
    column : str

class DBBRelationship(pydantic.BaseModel):
    fk : DBBColumnID
    pk : DBBColumnID

class DBBRDBDatasetMeta(pydantic.BaseModel):
    """Dataset metadata model."""
    # Dataset name.
    dataset_name : str
    # Table schemas.
    tables : List[DBBTableSchema]
    # Task metadata.
    tasks : List[DBBTaskMeta]

    @property
    def relationships(self) -> List[DBBRelationship]:
        """Get all relationships in a list."""
        rels = []
        for table in self.tables:
            for col in table.columns:
                if col.dtype == DBBColumnDType.foreign_key:
                    link_tbl, link_col = col.link_to.split('.')
                    fk = {'table' : table.name, 'column' : col.name}
                    pk = {'table' : link_tbl, 'column' : link_col}
                    rels.append(DBBRelationship.parse_obj({
                        'fk' : fk, 'pk' : pk}))
        return rels

    column_groups : Optional[List[List[DBBColumnID]]] = None
