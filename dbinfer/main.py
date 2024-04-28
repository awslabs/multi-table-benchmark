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

from .cli import (
    fit_gml,
    fit_tab,
    evaluate_gml,
    preprocess,
    construct_graph,
    list_builtin,
    download,
    get_node_embed
)

app = typer.Typer()

app.command()(fit_gml)
app.command()(fit_tab)
app.command()(evaluate_gml)
app.command()(preprocess)
app.command()(construct_graph)
app.command()(list_builtin)
app.command()(download)
app.command()(get_node_embed)

if __name__ == '__main__':
    app()
