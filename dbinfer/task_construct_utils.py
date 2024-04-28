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


import numpy as np
import pandas as pd

def train_val_test_split_by_ratio(df, train_ratio, val_ratio):
    """
    Split a dataframe without shuffling according to train-validation-test ratio

    .. code::

       train_ratio : val_ratio : (1 - train_ratio - val_ratio)
    """
    n = df.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    return train_df, val_df, test_df


def train_val_test_split_shuffled(df, train_ratio, val_ratio):
    """
    Split a dataframe with shuffling according to train-validation-test ratio

    .. code::

       train_ratio : val_ratio : (1 - train_ratio - val_ratio)
    """
    df = df.iloc[np.random.permutation(df.shape[0])]
    return train_val_test_split_by_ratio(df, train_ratio, val_ratio)


def train_val_test_split_by_temporal(df, time_col, train_ratio, val_ratio):
    """
    Split a dataframe temporally with timestamp indicated by ``time_col``
    according to train-validation-test ratio

    .. code::

       train_ratio : val_ratio : (1 - train_ratio - val_ratio)
    """
    df = df.sort_values(time_col)
    time = pd.to_datetime(df[time_col])
    train_time_cutoff, val_time_cutoff = np.quantile(
        np.unique(time),
        [train_ratio, train_ratio + val_ratio]
    )
    train_df = df[time <= train_time_cutoff]
    val_df = df[
        (time > train_time_cutoff) &
        (time <= val_time_cutoff)
    ]
    test_df = df[time > val_time_cutoff]
    assert train_df.shape[0] > 0
    assert val_df.shape[0] > 0
    assert test_df.shape[0] > 0
    return train_df, val_df, test_df


def train_val_test_split_by_split_column(df, split_col):
    """
    Split a dataframe according to ``split_col``: rows with value ``train`` becomes
    training set, rows with value ``valid`` becomes validation set, and rows with
    value ``test`` becomes test set.
    """
    train_df = df[df[split_col] == 'train']
    val_df = df[df[split_col] == 'valid']
    test_df = df[df[split_col] == 'test']
    return train_df, val_df, test_df


def assign_global_time_cutoff(df, time_col, prediction_time_col):
    """
    Allows temporal prediction on the training/validation/test instances use
    all other training/validation/test instances.

    Assigns the maximum of ``time_col`` into the ``prediction_time_col`` column.

    This is equivalent to converting a task with ``global`` temporality constraint
    into another one with an ``instance_wise`` temporality constraint.
    """
    df = df.assign(**{prediction_time_col: df[time_col].max()})
    return df


def assign_train_val_test_time_cutoff(train_df, val_df, test_df, time_col, prediction_time_col):
    """
    Applies the following temporality constraints:

    * Allows temporal prediction on the training instances to use all other training
      instances.

    * Allows temporal prediction on the validation instances to use all training
      instances, but none of the other validation instances.

    * Allows temporal prediction on the test instances use all training and validation
      instances, but none of the other test instances.
    """
    train_time_threshold = train_df[time_col].max()
    val_time_threshold = val_df[time_col].max()
    assert val_time_threshold >= train_time_threshold
    train_df = train_df.assign(**{prediction_time_col: train_time_threshold})
    val_df = val_df.assign(**{prediction_time_col: train_time_threshold})
    test_df = test_df.assign(**{prediction_time_col: val_time_threshold})
    return train_df, val_df, test_df




def generate_eval_negative_samples(
    df,
    all_pos_df,
    id_col,
    num_negatives,
    label_col='label',
    query_idx='query_idx'
):
    """
    Augments a dataframe by generating negative examples with column ``id_col``
    corrupted, and adds a column ``label_col`` indicating whether a row is
    corrupted (0) or not (1).

    Also generate a query index column to indicate which query each sample belongs to.
    It is done by grouping columns other than the ``id_col``, i.e., two samples with
    the same values of those columns are consider the same query.
    """
    neg_df = df.loc[np.repeat(df.index, num_negatives)].reset_index(drop=True)
    neg_df[label_col] = 0
    neg_df[id_col] = neg_df[id_col].sample(frac=1, replace=True).values

    # If any generated negative example actually exist as a positive example
    # (i.e. it is a false negative), remove it.
    # This is done by groupby(columns_except_label_col).max(), since it
    # is equivalent to removing the duplicate records with zero labels.
    all_pos_df = all_pos_df.assign(**{label_col:1})  # add an all-one label column
    mixed_df = pd.concat([all_pos_df, neg_df])
    columns_except_label_col = list(df.columns[df.columns != label_col])
    mixed_df = (
        mixed_df.groupby(columns_except_label_col)[label_col]
        .max().reset_index()
    )
    neg_df = mixed_df[mixed_df[label_col] == 0]

    df = df.assign(**{label_col:1})  # add an all-one label column
    augmented_df = pd.concat([df, neg_df])

    # Compute query index column.
    groupby_columns = [col for col in df.columns if col not in [id_col, label_col]]
    augmented_df[query_idx] = augmented_df.groupby(groupby_columns).ngroup().astype(int)

    return augmented_df


def save_partitioned_parquet(df, path, npartitions=None, partition_cols=None):
    import dask
    dask.config.set(scheduler='processes')
    import dask.dataframe as dd
    df = dd.from_pandas(df, npartitions=npartitions or 1)
    df.to_parquet(path, partition_on=partition_cols)


def tail_temporal(df, time_col, n_samples):
    """
    Take the last ``n_samples`` rows sorted by ``time_col``.
    """
    time = pd.to_datetime(df[time_col])
    trunc_timestamp = time.quantile(1 - n_samples / df.shape[0])
    return df[time > trunc_timestamp]
