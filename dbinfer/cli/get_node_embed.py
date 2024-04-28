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
import torch

import dbinfer_bench as dbb

from ..device import DeviceInfo, get_device_info
from ..solutions import (
    get_gml_solution_class,
    parse_config_from_graph_dataset,
    get_gml_solution_choice,
)
from .. import yaml_utils
from ..solutions.graph_dataset_config import GraphDatasetConfig

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

GMLSolutionChoice = get_gml_solution_choice()


def get_node_embed(
    solution_name : GMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    checkpoint_path : str = typer.Argument(
        ...,
        help="Path to the workspace. "
        " The workspace should contain data_config.yaml, solution_config.yaml, and model.pt."
    ),
    output_path : str = typer.Argument(
        ...,
        help="Path to save node embeddings."
    ),
):
    checkpoint_path = Path(checkpoint_path)
    solution_config_path = checkpoint_path / "solution_config.yaml"
    data_config_path = checkpoint_path / "data_config.yaml"

    solution_class = get_gml_solution_class(solution_name.value)

    logger.info(f"Load solution configuration file: {solution_config_path}.")
    solution_config = yaml_utils.load_pyd(solution_class.config_class, solution_config_path)
    logger.debug(f"Solution config:\n{solution_config.json()}")

    data_config = yaml_utils.load_pyd(GraphDatasetConfig, data_config_path)

    logger.info("Creating solution ...")
    solution = solution_class(solution_config, data_config)
    solution.load_from_checkpoint(checkpoint_path)
    embeddings = solution.model.get_node_embeddings()

    torch.save(embeddings, output_path)
