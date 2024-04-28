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


"""Datetime utilities."""
import pandas as pd
import dask.dataframe as dd
import numpy as np

def dask_dt2ts(dt : dd.Series) -> dd.Series:
    """Convert a Dask datetime series to epoch timestamp series."""
    return dt.map_partitions(lambda x: dt2ts(x))

def dt2ts(dt : np.ndarray) -> np.ndarray:
    dt = dt.astype('datetime64[ns]')
    return (dt - np.array(0).astype('datetime64[ns]')).astype('int64')

def ts2dt(ts : np.ndarray) -> np.ndarray:
    return np.array(ts).astype('datetime64[ns]')

def dt2year(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.year.values

def dt2month(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.month.values

def dt2day(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.day.values

def dt2dayofweek(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.dayofweek.values
