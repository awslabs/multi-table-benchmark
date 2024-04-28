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


"""Base class for feature encoders and registry."""

from typing import List, Dict, Optional, Any
import torch
import torch.nn as nn

from dbinfer_bench import DBBColumnDType

__all__ = ['feat_encoder', 'get_feat_encoder_class']
_ENCODER_REGISTRY = {}

class BaseFeatEncoder(nn.Module):

    """Base feature encoder class.

    Design principles:

      - Each feature encoder class is associated with one column dtype.
      - No need to handle missing, spurious values which belongs to featurization process.
      - All inputs are torch tensors.

    Parameters
    ----------
    config : dict[str, any]
        Configuration dict for initializing an encoder.
    out_size : int, optional
        The output feature size after encoding. If not specified,
        the encoder can have its own decision.

    Attributes
    ----------
    out_size : int
        Encoded feature size.
    """

    def __init__(
        self,
        config : Dict[str, Any],
        out_size : Optional[int] = None
    ):
        super().__init__()
        self._config = config
        self._out_size = out_size

    @property
    def config(self):
        return self._config

    @property
    def out_size(self):
        return self._out_size

    def forward(self, input_feat : torch.Tensor) -> torch.Tensor:
        """Forward function.

        All feature encoder takes one input feature tensor and return
        one encoded feature tensor.
        """
        return input_feat

def feat_encoder(dtype : DBBColumnDType):
    def _reg(encoder_class):
        global _ENCODER_REGISTRY
        _ENCODER_REGISTRY[dtype] = encoder_class
        return encoder_class
    return _reg

def get_encoder_class(dtype : DBBColumnDType, allow_missing=False):
    global _ENCODER_REGISTRY
    encoder_class = _ENCODER_REGISTRY.get(dtype, None)
    if not allow_missing and encoder_class is None:
        raise RuntimeError(f'Cannot find the encoder class for dtype {dtype}.')
    return encoder_class
