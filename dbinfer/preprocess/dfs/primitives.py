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
import numpy as np
from featuretools.primitives import AggregationPrimitive
import woodwork as ww
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType

class TokSeq(LogicalType):
    primary_dtype = "object"
    standard_tags = {"token_seq"}
ww.type_system.add_type(TokSeq)

class Array(LogicalType):
    primary_dtype = "object"
    standard_tags = {"array"}
ww.type_system.add_type(Array)

class Text(LogicalType):
    primary_dtype = "object"
    standard_tags = {"text"}
ww.type_system.add_type(Text)

class Concat(AggregationPrimitive):
    name = "concat"
    input_types = [ColumnSchema(logical_type=TokSeq, semantic_tags={"token_seq"})]
    return_type = ColumnSchema(logical_type=TokSeq, semantic_tags={"token_seq"})

    def get_function(self):
        def _agg(column):
            ret = []
            for val in column:
                if isinstance(val, list):
                    ret += val
            return ret
        return _agg

class Join(AggregationPrimitive):
    name = "join"
    input_types = [ColumnSchema(logical_type=Text, semantic_tags={"text"})]
    return_type = ColumnSchema(logical_type=Text, semantic_tags={"text"})

    def get_function(self):
        def _agg(column):
            return column.str.cat(sep='\n')
        return _agg

class ArrayMax(AggregationPrimitive):
    name = "arraymax"
    input_types = [ColumnSchema(logical_type=Array, semantic_tags={"array"})]
    return_type = ColumnSchema(logical_type=Array, semantic_tags={"array"})

    def get_function(self):
        def _agg(column):
            stack = _nanstack(column)
            return stack.max(0) if isinstance(stack, np.ndarray) else np.nan
        return _agg

class ArrayMin(AggregationPrimitive):
    name = "arraymin"
    input_types = [ColumnSchema(logical_type=Array, semantic_tags={"array"})]
    return_type = ColumnSchema(logical_type=Array, semantic_tags={"array"})

    def get_function(self):
        def _agg(column):
            stack = _nanstack(column)
            return stack.min(0) if isinstance(stack, np.ndarray) else np.nan
        return _agg

class ArrayMean(AggregationPrimitive):
    name = "arraymean"
    input_types = [ColumnSchema(logical_type=Array, semantic_tags={"array"})]
    return_type = ColumnSchema(logical_type=Array, semantic_tags={"array"})

    def get_function(self):
        def _agg(column):
            stack = _nanstack(column)
            return stack.mean(0) if isinstance(stack, np.ndarray) else np.nan
        return _agg

def _nanstack(arr_list : List[np.ndarray]) -> np.ndarray:
    """Stack a list of numpy ndarrays that may contain NaN."""
    arr_len = None
    for arr in arr_list:
        if isinstance(arr, np.ndarray):
            arr_len = len(arr)
            break
    if arr_len is None:
        return np.nan  # all values are NaN
    fill_val = np.zeros(arr_len)
    new_arr_list = [arr if isinstance(arr, np.ndarray) else fill_val for arr in arr_list]
    return np.stack(new_arr_list)
