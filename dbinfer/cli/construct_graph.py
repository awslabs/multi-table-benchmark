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


import typer
import logging
import os
import numpy as np

import dbinfer_bench as dbb

from ..graph_construction.utils import (
    GraphConstructionChoice,
    get_graph_construction_class,
)

from .. import yaml_utils

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def construct_graph(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder.")
    ),
    graph_construction_name : GraphConstructionChoice = typer.Argument(
        ...,
        help="Graph construction algorithm name"
    ),
    output_path : str = typer.Argument(
        ..., 
        help="Output path for the graph dataset."
    ),
    config_path : str = typer.Option(
        None,
        "--config_path", "-c",
        help="Configuration path. Use default if not specified."
    )
):

    graph_construction_class = get_graph_construction_class(graph_construction_name)

    logger.info("Loading data.")
    dataset = dbb.load_rdb_data(dataset_path)

    if config_path is None:
        logger.info("No configuration file provided. Use default configuration.")
        config = graph_construction_class.config_class()
    else:
        logger.info(f"Load configuration file: {config_path}.")
        config = yaml_utils.load_pyd(graph_construction_class.config_class, config_path)
    logger.debug(config.json())

    logger.info("Creating graph construction class.")
    graph_construction = graph_construction_class(config)

    graph_construction.build(dataset, output_path)
