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


__all__ = ['tabnn', 'get_tabnn_class']

_TABNN_REGISTRY = {}

def tabnn(tabnn_class):
    global _TABNN_REGISTRY
    _TABNN_REGISTRY[tabnn_class.name] = tabnn_class
    return tabnn_class

def get_tabnn_class(name : str):
    global _TABNN_REGISTRY
    tabnn_class = _TABNN_REGISTRY.get(name, None)
    if tabnn_class is None:
        raise ValueError(f"Cannot find the TabNN class of name {name}.")
    return tabnn_class
