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


from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np

import dbinfer_bench as dbb
from ..device import DeviceInfo
from ..solutions import (
    get_tabml_solution_class,
    parse_config_from_tabular_dataset,
    get_tabml_solution_choice,
)
from .. import yaml_utils
from .fit_utils import _fit_main

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

TabMLSolutionChoice = get_tabml_solution_choice()

def fit_tab(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : TabMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    config_path : Path = typer.Option(
        None,
        "--config_path", "-c",
        help="Solution configuration path. Use default if not specified."
    ),
    checkpoint_path : str = typer.Option(
        None, 
        "--checkpoint_path", "-p",
        help="Checkpoint path."
    ),
    enable_wandb : bool = typer.Option(
        True,
        "--enable-wandb/--disable-wandb",
        help="Enable Weight&Bias for logging."
    ),
    num_runs : int = typer.Option(
        1,
        "--num-runs", "-n",
        help="Number of runs."
    )
):
    solution_class = get_tabml_solution_class(solution_name.value)
    if config_path is None:
        logger.info("No solution configuration file provided. Use default configuration.")
        solution_config = solution_class.config_class()
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        solution_config = yaml_utils.load_pyd(solution_class.config_class, config_path)

    logger.debug(f"Solution config:\n{solution_config.json()}")

    logger.info("Loading data ...")
    dataset = dbb.load_rdb_data(dataset_path)

    data_config = parse_config_from_tabular_dataset(dataset, task_name)
    logger.debug(f"Data config:\n{data_config.json()}")

    def _invoke_fit(solution, run_ckpt_path : Path, device : DeviceInfo):
        summary = solution.fit(dataset, task_name, run_ckpt_path, device)
        return summary

    def _invoke_test(solution, run_ckpt_path : Path, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        val_metric = solution.evaluate(
            dataset.get_task(task_name).validation_set,
            device
        )
        test_metric = solution.evaluate(
            dataset.get_task(task_name).test_set,
            device
        )
        return val_metric, test_metric

    _fit_main(
        solution_class,
        dataset,
        data_config,
        solution_config,
        checkpoint_path,
        enable_wandb,
        num_runs,
        _invoke_fit,
        _invoke_test
    )
