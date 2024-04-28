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


from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass
import numpy as np
import dgl
import torch

@dataclass
class Het2HomInfo:
    ntype_to_id : Dict[str, int]
    etype_to_id : Dict[str, int]
    ntype_offset : np.ndarray
    etype_offset : np.ndarray
    num_dst_nodes_dict : Dict[str, int]

def block_to_homogeneous(
    block,
    source_ndata : Dict[str, torch.Tensor],
    edata : Dict[str, torch.Tensor],
    srcdata_name = '__tmp_src_feat__',
    edata_name = '__tmp_edge_feat__',
):
    """Convert a DGLBlock to a homogeneous graph.

    This function performs a two-step process to convert a given block into a
    homogeneous graph. It first transforms the block into a heterogeneous graph,
    treating its source nodes as heterogeneous nodes. Then, this heterogeneous
    graph is further converted to a homogeneous graph. The function outputs
    includes the homogeneous graph, the mapping from node type to node type ID,
    node type count offsets, and the number of destination nodes for each node
    type. Notably, the `source_ndata` of the original block is retained and
    stored in the `srcdata_name` attribute of the homogeneous graph.

    Parameters
    ----------
    block : dgl.Block
        The block to convert.
    source_ndata : dict[str, Tensor]
        The source node features of the block.
    edata : dict[str, Tensor]
        The edge features of the block.
    srcdata_name : str
        The name of the source node features in the homogeneous graph.
    edata_name : str
        The name of the edge features in the homogenous graph.
    
    Returns
    -------
    hom_g : dgl.DGLBlock
        The converted homogeneous graph.
    srcdata : torch.Tensor
        The converted source feature tensor.
    edata : Optional[torch.Tensor]
        The converted edge feature tensor. None means no edge feature.
    info : Het2HomInfo
        Auxiliary info.
    """
    num_dst_nodes_dict = {}
    num_src_nodes_dict = {}
    for ntype in block.dsttypes:
        num_dst_nodes_dict[ntype] = block.number_of_dst_nodes(ntype)
    for ntype in block.srctypes:
        num_src_nodes_dict[ntype] = block.number_of_src_nodes(ntype)

    hetero_edges = {}
    for srctpye, etype, dsttype in block.canonical_etypes:
        src, dst = block.all_edges(etype=etype, order="eid")
        hetero_edges[(srctpye, etype, dsttype)] = (src, dst)
    hetero_g = dgl.heterograph(
        hetero_edges,
        num_nodes_dict=num_src_nodes_dict,
        idtype=block.idtype,
        device=block.device,
    )
    ntype_to_id = {ntype: hetero_g.get_ntype_id(ntype) for ntype in hetero_g.ntypes}
    etype_to_id = {etype: hetero_g.get_etype_id(etype) for etype in hetero_g.canonical_etypes}
    assert len(source_ndata) == len(hetero_g.ntypes)
    hetero_g.ndata[srcdata_name] = source_ndata
    has_edata = (len(edata) != 0)
    if has_edata:
        edata = {
            tuple(etype_str.split(':')) : data
            for etype_str, data in edata.items()
        }
        any_data = list(edata.values())[0]
        size = any_data.shape[1]
        device = any_data.device
        for etype in hetero_g.canonical_etypes:
            if etype not in edata:
                # Pad missing edge features with zeros.
                edata[etype] = torch.zeros(
                    (hetero_g.num_edges(etype=etype), size),
                    device=device)
        hetero_g.edata[edata_name] = edata

    homo_g, ntype_counts, etype_counts = dgl.to_homogeneous(
        hetero_g,
        ndata=[srcdata_name],
        edata=[edata_name] if has_edata else [],
        return_count=True
    )
    srcdata = homo_g.ndata.pop(srcdata_name)
    edata = homo_g.edata.pop(edata_name, None)
    ntype_offset = np.insert(np.cumsum(ntype_counts), 0, 0)
    etype_offset = np.insert(np.cumsum(etype_counts), 0, 0)
    info = Het2HomInfo(
        ntype_to_id,
        etype_to_id,
        ntype_offset,
        etype_offset,
        num_dst_nodes_dict
    )
    return (
        homo_g,
        srcdata,
        edata,
        info
    )

def convert_dstdata_to_dict(
    dstdata : torch.Tensor,
    hetero_mfg : dgl.DGLGraph,
    info : Het2HomInfo,
) -> Dict[str, torch.Tensor]:
    dstdata_dict = {}
    for ntype in hetero_mfg.dsttypes:
        ntype_id = info.ntype_to_id[ntype]
        feature = dstdata[
            info.ntype_offset[ntype_id] : info.ntype_offset[ntype_id + 1]
        ]
        dstdata_dict[ntype] = feature[: info.num_dst_nodes_dict[ntype]]
    return dstdata_dict
