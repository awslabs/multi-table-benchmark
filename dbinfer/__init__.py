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


from . import logger
from . import time_budget

def _check_dgl_version():
    import dgl
    required_version = "2.1a240205"
    parts = dgl.__version__.split('+')
    current_version = parts[0]
    if current_version != required_version:
        raise RuntimeError(
            f"Required DGL version {required_version} but the installed version is {current_version}."
        )

_check_dgl_version()
