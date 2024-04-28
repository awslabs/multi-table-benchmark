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


import abc
from pathlib import Path

from dbinfer_bench import DBBRDBDataset

class GraphConstruction:

    config_class = None
    name = "base"

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def build(
        self,
        dataset : DBBRDBDataset,
        output_path : Path
    ):
        pass
