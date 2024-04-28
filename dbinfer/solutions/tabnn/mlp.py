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


from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registry

class MLPTabNNConfig(pydantic.BaseModel):
    hid_size : Optional[int] = 128
    dropout : Optional[float] = 0.5
    num_layers : Optional[int] = 3
    use_bn : Optional[bool] = False

@registry.tabnn
class MLPTabNN(nn.Module):
    name = "mlp"
    config_class = MLPTabNNConfig
    def __init__(self,
                 config : MLPTabNNConfig,
                 num_fields : int,
                 field_size : int,
                 out_size : int):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        in_size = num_fields * field_size
        self.layers += nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_size, config.hid_size, bias=not config.use_bn),
            nn.ReLU(),
        )
        if config.use_bn:
            self.layers.append(nn.BatchNorm1d(config.hid_size))
        for i in range(config.num_layers - 2):
            self.layers += nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hid_size, config.hid_size, bias=not config.use_bn),
                nn.ReLU(),
            )
            if config.use_bn:
                self.layers.append(nn.BatchNorm1d(config.hid_size))
        self.layers.append(nn.Linear(config.hid_size, out_size))

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """Forward

        Input shape:
            X : (N, F, D_f)
                N is the batch size, F is the number of fields, D_f is field size.

        Output shape:
            H : (N, D_o), D_o is the output size.
        """
        X = X.view(X.shape[0], -1)
        return self.layers(X)
