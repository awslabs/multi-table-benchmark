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
import copy
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import pydantic
import tqdm
import time
import os
import wandb
import functools
from collections import defaultdict

import dgl
import dgl.graphbolt as gb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from dbinfer_bench import (
    DBBGraphDataset,
    DBBTaskType,
    TIMESTAMP_FEATURE_NAME,
)

from .base import (
    GraphMLSolution,
    GraphMLSolutionConfig,
    FitSummary,
)
from .graph_dataset_config import (
    GraphConfig,
    GraphDatasetConfig,
)
from .encoders import GraphFeatDictEncoder, IdDictEncoder
from .predictor import PredictorConfig, Predictor, SeedLookup
from .negative_sampler import DBBNegativeSampler
from ..evaluator import get_metric_fn, get_loss_fn
from ..device import DeviceInfo
from .. import yaml_utils
from ..time_budget import TimeBudgetedIterator

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

NType = str
EType = Tuple[str, str, str]

__all__ = ['BaseGMLSolution']

class BaseGNNSolutionConfig(GraphMLSolutionConfig):
    predictor : Optional[PredictorConfig] = PredictorConfig()
    use_multiprocessing : bool = True
    eval_trials : int = 10

class BaseGNN(nn.Module):

    def __init__(
        self,
        solution_config : BaseGNNSolutionConfig,
        data_config : GraphDatasetConfig
    ):
        super().__init__()
        self._solution_config = solution_config
        self._data_config = data_config

        self.feat_encoder = GraphFeatDictEncoder(
            data_config,
            solution_config.feat_encode_size)
        self.node_id_encoder = IdDictEncoder(
            data_config.graph.num_nodes,
            solution_config.embed_ntypes,
            solution_config.feat_encode_size
        )
        node_out_size_dict = dict(self.node_id_encoder.out_size_dict)
        node_out_size_dict.update(self.feat_encoder.node_out_size_dict)
        for ntype in solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                node_out_size_dict[ntype] += self.node_id_encoder.out_size_dict[ntype]

        if solution_config.predictor is None:
            assert data_config.task.num_seeds == 1, \
                "Setting predictor to be None is only allowed for node-level prediction."
            gnn_out_size = Predictor.get_out_size(data_config.task)
        else:
            gnn_out_size = None
        self.gnn = self.create_gnn(
            node_out_size_dict,
            self.feat_encoder.edge_out_size_dict,
            self.feat_encoder.seed_ctx_out_size,
            gnn_out_size)

        self.seed_lookup = SeedLookup(data_config.task.seed_type)
        if solution_config.predictor is None:
            self.predictor = lambda seed_embeds, seed_ctx_embeds : seed_embeds
        else:
            self.predictor = Predictor(
                data_config.task,
                solution_config.predictor,
                self.gnn.out_size,
                self.feat_encoder.seed_ctx_out_size,
            )

    @property
    def solution_config(self) -> BaseGNNSolutionConfig:
        return self._solution_config

    @property
    def data_config(self) -> GraphDatasetConfig:
        return self._data_config

    @abc.abstractmethod
    def create_gnn(
        self,
        graph_cfg : GraphConfig,
        node_feat_size_dict : Dict[str, int],
        edge_feat_size_dict : Dict[str, int],
        seed_feat_size : int,
        out_size : Optional[int],
    ) -> nn.Module:
        pass

    def forward(
        self,
        mfgs,
        node_feat_dict : Dict[str, Dict[str, torch.Tensor]],
        input_node_id_dict : Dict[str, torch.Tensor],
        edge_feat_dicts : List[Dict[str, Dict[str, torch.Tensor]]],
        seed_feat_dict : Dict[str, Dict[str, torch.Tensor]],
        seed_lookup_idx : torch.Tensor
    ):
        # Encode IDs.
        H_id_dict = self.node_id_encoder(input_node_id_dict)

        # Encode features.
        H_feat_dict = self.feat_encoder(node_feat_dict)

        # Mask leakage features. They are features of target type that exists
        # in RDB but not in seed contexts.
        target_type = self.data_config.task.target_type
        seed_type = self.data_config.task.seed_type
        if seed_type == target_type and target_type in H_feat_dict:
            # Mask out features of seeds.
            neigh_feat_set = set(H_feat_dict[target_type].keys())
            seed_feat_set = set(seed_feat_dict['__seed__'].keys())
            num_seeds = mfgs[-1].num_dst_nodes(ntype=target_type)
            for key_to_mask in neigh_feat_set - seed_feat_set:
                H = H_feat_dict[target_type][key_to_mask]
                H[:num_seeds] = 0.

        H_feat_dict = _cat_feat(H_feat_dict)

        # Merge two dictionaries.
        H_node_dict = dict(H_id_dict)
        H_node_dict.update(H_feat_dict)
        for ntype in self.solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                H_node_dict[ntype] = torch.cat(
                    [H_feat_dict[ntype], H_id_dict[ntype]], dim=1)

        # Encode edges.
        H_edge_dicts = [
            _cat_feat(self.feat_encoder(edge_feat_dict))
            for edge_feat_dict in edge_feat_dicts
        ]

        # Message passing.
        H_node_dict = self.gnn(mfgs, H_node_dict, H_edge_dicts)

        # Prediction head.
        seed_embeds = self.seed_lookup(H_node_dict, seed_lookup_idx)
        seed_ctx_embeds = _cat_feat(self.feat_encoder(seed_feat_dict))['__seed__']
        return self.predictor(seed_embeds, seed_ctx_embeds)

    def get_node_embeddings(self) -> Dict[str, torch.Tensor]:
        return self.node_id_encoder.get_embedding_dict()

class BaseGMLSolution(GraphMLSolution):
    """Base GML solution class."""

    def __init__(
        self,
        solution_config : BaseGNNSolutionConfig,
        data_config : GraphDatasetConfig
    ):
        self.solution_config = solution_config
        self.data_config = data_config
        self.model = self.create_model()
        self._dataloaders = {}

        logger.debug(self.model)
        total_size = 0
        for param in self.model.parameters():
            total_size += param.nelement() * param.element_size()
        logger.debug(f"Model parameter size: {total_size/1024**2:.2f} MB")

    @abc.abstractmethod
    def create_model(self) -> nn.Module:
        pass

    def create_dataloader(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        mode : str,
    ) -> DataLoader:
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        if (id(item_set_dict), id(graph), id(feat_store), model_device, mode) not in self._dataloaders:
            self._dataloaders[id(item_set_dict), id(graph), id(feat_store), model_device, mode] = self._create_dataloader(
                item_set_dict, graph, feat_store, device, mode
            )
        return self._dataloaders[id(item_set_dict), id(graph), id(feat_store), model_device, mode]

    def _create_dataloader(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        mode : str,
    ) -> DataLoader:
        shuffle = (mode == 'train')
        batch_size = self.solution_config.batch_size \
            if mode == 'train' else self.solution_config.eval_batch_size
        # Declare data loading procedure.

        # 1. Sample items to form initial minibatch.
        datapipe = gb.ItemSampler(
            item_set_dict, batch_size=batch_size, shuffle=shuffle)

        # 2. (optional) negative sampling
        if self.data_config.task.task_type == DBBTaskType.retrieval and mode == 'train':
            datapipe = datapipe.tgif_sample_negative(
                graph,
                self.solution_config.negative_sampling_ratio,
                self.data_config.task.target_seed_idx,
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            )

        if mode == 'train' or self.solution_config.eval_fanouts is None:
            fanouts = self.solution_config.fanouts
        else:
            fanouts = self.solution_config.eval_fanouts
        if (
            self.solution_config.enable_temporal_sampling
            and self.data_config.task.seed_timestamp is not None
        ):
            logger.info("Using temporal neighbor sampler.")
            has_node_timestamp = TIMESTAMP_FEATURE_NAME in graph.node_attributes
            has_edge_timestamp = TIMESTAMP_FEATURE_NAME in graph.edge_attributes
            datapipe = datapipe.temporal_sample_neighbor(
                graph, fanouts=fanouts,
                node_timestamp_attr_name=TIMESTAMP_FEATURE_NAME if has_node_timestamp else None,
                edge_timestamp_attr_name=TIMESTAMP_FEATURE_NAME if has_edge_timestamp else None,
            )
        else:
            datapipe = datapipe.sample_neighbor(
                graph, fanouts=fanouts)

        # 4. (optional) exclude seed edges from the sampled subgraph.
        if self.data_config.task.seed_type in self.data_config.graph.etypes:
            datapipe = datapipe.transform(
                functools.partial(
                    gb.exclude_seed_edges, include_reverse_edges=True,
                    reverse_etypes_mapping=self.data_config.graph.reverse_etypes_mapping))

        # 5. Fetch node/edge features of the surrounding subgraph.
        node_feature_keys = {}
        for ntype, ft_cfg_dict in self.data_config.node_features.items():
            ft_names = [
                ft_name for ft_name, ft_cfg in ft_cfg_dict.items()
                if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
            ]
            node_feature_keys[ntype] = ft_names
        edge_feature_keys = {}
        for etype, ft_cfg_dict in self.data_config.edge_features.items():
            ft_names = [
                ft_name for ft_name, ft_cfg in ft_cfg_dict.items()
                if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
            ]
            edge_feature_keys[etype] = ft_names
        datapipe = datapipe.fetch_feature(
            feat_store, node_feature_keys, edge_feature_keys)

        # 6. Copy to training device.
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        datapipe = datapipe.copy_to(model_device)

        # Create dataloader.
        if self.solution_config.use_multiprocessing:
            dataloader = gb.DataLoader(datapipe, num_workers=device.cpu_count // 2)
        else:
            dataloader = gb.DataLoader(datapipe, num_workers=0)
        return dataloader

    def create_optimizer(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=self.solution_config.lr)

    def get_input_node_ids(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo
    ) -> Dict[NType, torch.Tensor]:
        return {
            ntype : ids.to(device)
            for ntype, ids in minibatch.input_nodes.items()
        }

    def get_node_feats(
        self,
        minibatch : gb.MiniBatch,
    ) -> Dict[NType, Dict[str, torch.Tensor]]:
        node_feat_dict = defaultdict(dict)
        for (ntype, feat_name), feat in minibatch.node_features.items():
            node_feat_dict[ntype][feat_name] = feat
        return node_feat_dict

    def get_edge_feats(
        self, minibatch : gb.MiniBatch
    ) -> List[Dict[EType, Dict[str, torch.Tensor]]]:
        edge_feat_dicts = []
        for efeat_dict in minibatch.edge_features:
            new_efeat_dict = defaultdict(dict)
            for (etype, feat_name), feat in efeat_dict.items():
                new_efeat_dict[etype][feat_name] = feat
            edge_feat_dicts.append(new_efeat_dict)
        return edge_feat_dicts

    def get_seed_lookup_idx(
        self, minibatch : gb.MiniBatch
    ) -> Optional[torch.Tensor]:
        """Seed lookup index is used to lookup seed embeddings from
        the output of message passing, and arange them to align with
        the input seed tensor.

        For example, if the input seed shape is (N,K), the look up index
        is of the same shape (N,K). Suppose message passing computes
        node embedding of shape (M,D), where M is typically <= N*K because
        seeds can have duplicates. Then a lookup operation will gives
        a seed embedding tensor of shape (N,K,D).

        None return value means identity mapping.
        """
        if minibatch.seed_nodes is not None:
            return None
        else:
            assert minibatch.compacted_node_pairs is not None
            seed_type = self.data_config.task.seed_type
            if minibatch.compacted_negative_srcs is not None:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                neg_src = minibatch.compacted_negative_srcs[seed_type]
                neg_dst = minibatch.compacted_negative_dsts[seed_type]
                # TODO(minjie): Here is another logic that is coupled tightly with GB's
                # internal behavior of how negative samples are aranged.
                neg_src = neg_src.view(-1, self.solution_config.negative_sampling_ratio)
                neg_dst = neg_dst.view(-1, self.solution_config.negative_sampling_ratio)
                all_src = torch.cat([pos_src.unsqueeze(1), neg_src], dim=1)
                all_dst = torch.cat([pos_dst.unsqueeze(1), neg_dst], dim=1)
                idx = torch.stack([all_src.reshape(-1), all_dst.reshape(-1)]).T
            else:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                idx = torch.stack([pos_src, pos_dst]).T
            return idx

    def get_labels(
        self, minibatch : gb.MiniBatch
    ) -> torch.Tensor:
        seed_type = self.data_config.task.seed_type
        return minibatch.labels[seed_type]

    def get_seed_feats(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo,
        mode : str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        seed_type = self.data_config.task.seed_type
        feats = {
            ft_name : getattr(minibatch, ft_name)[seed_type]
            for ft_name, ft_cfg in self.data_config.seed_features.items()
            if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
        }
        if self.data_config.task.task_type == DBBTaskType.retrieval and mode == 'train':
            feats = {
                ft_name : feat[minibatch.query_idx[seed_type]]
                for ft_name, feat in feats.items()
            }
        feats = {ft_name : feat.to(device) for ft_name, feat in feats.items()}
        return {'__seed__' : feats}

    def get_query_idx(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo
    ) -> Optional[torch.Tensor]:
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            query_idx = minibatch.query_idx[self.data_config.task.seed_type]
            return query_idx.to(device)
        else:
            return None

    def fit(
        self,
        dataset : DBBGraphDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        ckpt_path = Path(ckpt_path)
        metric_fn = get_metric_fn(self.data_config.task)
        loss_fn = get_loss_fn(self.data_config.task)

        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)

        train_loader = self.create_dataloader(
            dataset.graph_tasks[task_name].train_set,
            dataset.graph, dataset.feature, device, 'train')
        # TODO(minjie): replace the following with len(train_loader) in the future.
        num_batches = len(dataset.graph_tasks[task_name].train_set) // self.solution_config.batch_size + 1

        optimizer = self.create_optimizer()

        best_val_metric = float('-inf')
        counter = 0
        self.checkpoint(ckpt_path)
        for epoch in TimeBudgetedIterator(
            range(self.solution_config.epochs),
            self.solution_config.time_budget
        ):
            self.model.train()
            total_loss = 0.
            total_train_metric = 0.
            with tqdm.tqdm(train_loader, total=num_batches) as tq:
                t0 = time.time()
                for step, minibatch in enumerate(tq):
                    # Prepare data.
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'train')
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch)
                    query_idx = self.get_query_idx(minibatch, model_device)
                    labels = self.get_labels(minibatch)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx
                    )
                    loss = loss_fn(logits, labels)
                    # Backward.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Logging.
                    if step % 20 == 0:
                        # Metrics that are computationally costly.
                        if query_idx is not None:
                            query_idx = query_idx.cpu()
                        train_metric = metric_fn(
                            query_idx, logits.cpu(), labels.cpu()).item()
                        grad_norm = sum(
                            p.grad.norm() ** 2
                            for p in self.model.parameters() if p.grad is not None
                        )
                    total_loss += loss.detach()
                    total_train_metric += train_metric
                    tq.set_postfix(
                        {
                            'Train loss': f'{loss.item():.4f}',
                            'Train metric': f'{train_metric:.4f}',
                            'Grad norm': f'{grad_norm:.6f}'
                        },
                        refresh=False,
                    )
                    wandb.log(
                        {'loss' : loss, 'grad_norm': grad_norm, 'train_metric': train_metric}
                    )

            total_loss /= step + 1
            total_train_metric /= step + 1

            val_metric = self.evaluate(
                dataset.graph_tasks[task_name].validation_set, dataset.graph, dataset.feature, device)
            if val_metric <= best_val_metric:
                counter += 1
                logger.info(
                    f"EarlyStopping counter: {counter} out of {self.solution_config.patience}"
                )
                if counter >= self.solution_config.patience:
                    break
            else:
                counter = 0
                logger.debug('Checkpointing ...')
                self.checkpoint(ckpt_path)
                best_val_metric = val_metric

            logger.info(
                f"Epoch {epoch:04d} | loss: {total_loss:.4f} | "
                f"train metric: {total_train_metric:.4f} | "
                f"val metric: {val_metric:.4f} | "
                f"best val metric: {best_val_metric:.4f}"
            )
            wandb.log({'val_metric' : val_metric})

        summary = FitSummary()
        summary.val_metric = float(best_val_metric)
        summary.train_metric = float(total_train_metric)

        return summary

    def evaluate(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
    ) -> float:
        metric_fn = get_metric_fn(self.data_config.task)
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)
        self.model.eval()

        eval_loader = self.create_dataloader(
            item_set_dict, graph, feat_store, device, 'eval')
        # TODO(minjie): replace the following with len(val_loader) in the future.
        num_batches = len(item_set_dict) // self.solution_config.eval_batch_size + 1
        with torch.no_grad():
            logits_per_trial = []
            labels_list = []
            query_idx_list = []
            for t in range(self.solution_config.eval_trials):
                logits_list = []
                for minibatch in tqdm.tqdm(eval_loader, total=num_batches):
                    # Prepare data.
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'eval')
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch)
                    query_idx = self.get_query_idx(minibatch, model_device)
                    labels = self.get_labels(minibatch)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx
                    )
                    logits_list.append(logits)
                    if t == 0:
                        if query_idx is None:
                            query_idx_list = None
                        else:
                            query_idx_list.append(query_idx)
                        labels_list.append(labels)
                all_logits = torch.cat(logits_list)
                if t == 0:
                    if query_idx_list is None:
                        all_query_idx = None
                    else:
                        all_query_idx = torch.cat(query_idx_list).cpu()
                    all_labels = torch.cat(labels_list).cpu()
                logits_per_trial.append(all_logits.cpu())
        all_logits = torch.stack(logits_per_trial, 0).mean(0)
        return metric_fn(all_query_idx, all_logits, all_labels).item()

    def checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path / 'model.pt')
        yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')

    def load_from_checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.data_config = yaml_utils.load_pyd(
            GraphDatasetConfig, ckpt_path / 'data_config.yaml')
        self.model = self.create_model()
        self.model.load_state_dict(torch.load(ckpt_path / 'model.pt'))

def _cat_feat(feat_dict : Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    cat_feat_dict = {}
    for ty, ty_feat_dict in feat_dict.items():
        if len(ty_feat_dict) == 0:
            cat_feat_dict[ty] = None
        else:
            cat_feat_dict[ty] = torch.cat(
                [ty_feat_dict[feat_name] for feat_name in sorted(ty_feat_dict)], dim=1)
    return cat_feat_dict
