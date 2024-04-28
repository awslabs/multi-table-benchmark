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
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from dbinfer_bench import DBBRDBDataset, DBBTaskType, DBBColumnDType

from .base import (
    TabularMLSolution,
    TabularMLSolutionConfig,
    FitSummary,
    tabml_solution,
)
from .tabular_dataset_config import TabularDatasetConfig
from .encoders import FeatDictEncoder, IdDictEncoder
from . import negative_sampler as NS
from .tabnn import get_tabnn_class
from ..evaluator import get_metric_fn, get_loss_fn
from ..device import DeviceInfo
from .. import yaml_utils
from ..time_budget import TimeBudgetedIterator

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

__all__ = ['TabNNSolution', 'TabNNSolutionConfig']

class TabularDataset(Dataset):
    def __init__(self, table : Dict[str, np.ndarray]):
        self.table = {col : torch.from_numpy(val) for col, val in table.items()}
        self.columns = list(table.keys())

    def __len__(self):
        return len(self.table[self.columns[0]])

    def __getitem__(self, idx):
        return {col : self.table[col][idx] for col in self.columns}

class TabNNSolutionConfig(TabularMLSolutionConfig):
    feat_encode_size : Optional[int] = 32
    nn_name : str
    nn_config : Optional[Dict[str, Any]] = None

class TabNN(nn.Module):
    def __init__(
        self,
        data_config : TabularDatasetConfig,
        solution_config : TabNNSolutionConfig
    ):
        super().__init__()
        self.solution_config = solution_config
        feat_cfg = {}
        key_capacity = {}
        for feat_name, cfg in data_config.features.items():
            if feat_name in solution_config.embed_keys:
                if cfg.dtype == DBBColumnDType.primary_key:
                    logger.warning(
                        f"Embedding primary key {feat_name} may lead cause over-fitting."
                        " Please use with caution."
                    )
                key_capacity[feat_name] = cfg.extra_fields['capacity']
            elif cfg.dtype in [
                DBBColumnDType.primary_key,
                DBBColumnDType.foreign_key,
            ]: 
                continue
            elif feat_name in [
                data_config.task.target_column,
                data_config.task.time_column,
                data_config.task.key_prediction_label_column,
                data_config.task.key_prediction_query_idx_column,
            ]:
                continue
            else:
                feat_cfg[feat_name] = cfg

        self.feat_encoder = FeatDictEncoder(
            feat_cfg,
            feature_groups=[],
            feat_encode_size=solution_config.feat_encode_size)
        self.id_encoder = IdDictEncoder(
            key_capacity,
            solution_config.embed_keys,
            solution_config.feat_encode_size)
        out_size_dict = copy.deepcopy(self.feat_encoder.out_size_dict)
        out_size_dict.update(self.id_encoder.out_size_dict)
        if solution_config.feat_encode_size is None:
            num_fields = 1
            field_size = sum(out_size_dict.values())
        else:
            num_fields = len(out_size_dict)
            field_size = solution_config.feat_encode_size
        if num_fields == 0:
            raise ValueError(f"No valid input feature among "
                             f"{list(data_config.features.keys())}.")

        nn_class = get_tabnn_class(solution_config.nn_name)
        if solution_config.nn_config is None:
            nn_cfg = nn_class.config_class()
        else:
            nn_cfg = nn_class.config_class.parse_obj(solution_config.nn_config)

        task_config = data_config.task
        if task_config.task_type == DBBTaskType.classification:
            out_size = task_config.num_classes
        elif task_config.task_type == DBBTaskType.regression:
            out_size = 1
        elif task_config.task_type == DBBTaskType.retrieval:
            out_size = 1
        else:
            raise ValueError(f"Unsupported task type {task_config.task_type}.")

        self.nn = nn_class(nn_cfg, num_fields, field_size, out_size)

    def forward(
        self,
        feat_dict : Dict[str, torch.Tensor],
        key_id_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        H_dict = self.feat_encoder(feat_dict)
        H_dict.update(self.id_encoder(key_id_dict))
        if self.solution_config.feat_encode_size is None:
            # Variable-length fields.
            H = torch.cat([H_dict[key] for key in sorted(H_dict.keys())], dim=1)
            H = H.unsqueeze(1)  # (N, D) -> (N, 1, D)
        else:
            H = torch.stack([H_dict[key] for key in sorted(H_dict.keys())], dim=1)
        return self.nn(H).squeeze(-1)

@tabml_solution
class TabNNSolution(TabularMLSolution):
    """NN-based tabular ML solution class."""
    config_class = TabNNSolutionConfig
    name = "tabnn"

    def __init__(
        self,
        solution_config : TabNNSolutionConfig,
        data_config : TabularDatasetConfig
    ):
        self.solution_config = solution_config
        self.data_config = data_config
        self.model = self.create_model()

        logger.debug(self.model)
        total_size = 0
        for param in self.model.parameters():
            total_size += param.nelement() * param.element_size()
        logger.debug(f"Model parameter size: {total_size/1024**2:.2f} MB")

    def create_model(self) -> nn.Module:
        return TabNN(self.data_config, self.solution_config)

    def create_dataloader(
        self,
        table : Dict[str, np.ndarray],
        device : DeviceInfo,
        mode : str
    ) -> DataLoader:
        dataset = TabularDataset(table)

        shuffle = (mode == 'train')
        batch_size = self.solution_config.batch_size \
            if mode == 'train' else self.solution_config.eval_batch_size
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=device.cpu_count // 2
        )
        return dataloader

    def create_optimizer(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=self.solution_config.lr)

    def get_feats(self, minibatch : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            feat_name : feat for feat_name, feat in minibatch.items()
            if feat_name not in [
                self.data_config.task.target_column,
                self.data_config.task.time_column,
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            ] and self.data_config.features[feat_name].dtype not in [
                DBBColumnDType.primary_key,
                DBBColumnDType.foreign_key
            ]
        }

    def get_keys(self, minibatch : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            feat_name : feat for feat_name, feat in minibatch.items()
            if feat_name not in [
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            ] and self.data_config.features[feat_name].dtype in [
                DBBColumnDType.primary_key,
                DBBColumnDType.foreign_key
            ]
        }

    def get_labels(self, minibatch : Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            return minibatch[self.data_config.task.key_prediction_label_column]
        else:
            return minibatch[self.data_config.task.target_column]
    
    def negative_sampling(
        self,
        minibatch : Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Augment the minibatch with negative samples.

        Returns
        -------
        minibatch : Dict[str, torch.Tensor]
            New minibatch with negative samples, labels and query index.
        """
        target_column_name = self.data_config.task.target_column
        target_column_capacity = (
            self.data_config.features[target_column_name].extra_fields['capacity']
        )
        minibatch = NS.negative_sampling(
            minibatch,
            self.solution_config.negative_sampling_ratio,
            target_column_name,
            target_column_capacity,
            self.data_config.task.key_prediction_label_column,
            self.data_config.task.key_prediction_query_idx_column,
            shuffle_rest_columns=True
        )

        return minibatch

    def get_query_idx(self, minibatch):
        return minibatch.get(self.data_config.task.key_prediction_query_idx_column, None)

    def fit(
        self,
        dataset : DBBRDBDataset,
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
            dataset.get_task(task_name).train_set, device, 'train')
        num_batches = len(train_loader)

        optimizer = self.create_optimizer()

        best_val_metric = float('-inf')
        counter = 0
        self.checkpoint(ckpt_path)
        for epoch in TimeBudgetedIterator(
            range(self.solution_config.epochs),
            self.solution_config.time_budget
        ):
            self.model.train()
            total_train_metric = 0.
            total_loss = 0.
            with tqdm.tqdm(train_loader, total=num_batches) as tq:
                t0 = time.time()
                for step, minibatch in enumerate(tq):
                    # Prepare data.
                    if self.data_config.task.task_type == DBBTaskType.retrieval:
                        minibatch = self.negative_sampling(minibatch)
                    minibatch = {key : val.to(model_device) for key, val in minibatch.items()}
                    feats = self.get_feats(minibatch)
                    keys = self.get_keys(minibatch)
                    labels = self.get_labels(minibatch)
                    query_idx = self.get_query_idx(minibatch)

                    # Forward.
                    logits = self.model(feats, keys)
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
                dataset.get_task(task_name).validation_set, device)
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
        table : Dict[str, np.ndarray],
        device : DeviceInfo
    ) -> float:
        metric_fn = get_metric_fn(self.data_config.task)
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)
        self.model.eval()

        eval_loader = self.create_dataloader(table, device, 'eval')
        num_batches = len(eval_loader)
        with torch.no_grad():
            logits_list = []
            labels_list = []
            query_idx_list = []
            for minibatch in tqdm.tqdm(eval_loader, total=num_batches):
                # Prepare data.
                minibatch = {key : val.to(model_device) for key, val in minibatch.items()}
                feats = self.get_feats(minibatch)
                keys = self.get_keys(minibatch)
                labels = self.get_labels(minibatch)
                query_idx = self.get_query_idx(minibatch)

                # Forward.
                logits = self.model(feats, keys)
                if query_idx is None:
                   query_idx_list = None
                else:
                   query_idx_list.append(query_idx)
                logits_list.append(logits)
                labels_list.append(labels)
        if query_idx_list is None:
            query_idx = None
        else:
            query_idx = torch.cat(query_idx_list).cpu()
        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
        return metric_fn(query_idx, logits, labels).item()

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
            TabularDatasetConfig, ckpt_path / 'data_config.yaml')
        self.model = self.create_model()
        self.model.load_state_dict(torch.load(ckpt_path / 'model.pt'))
