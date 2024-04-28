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


import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import logging
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from tqdm import tqdm
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def _run_one_device(
    proc_id : int,
    num_procs : int,
    data : np.ndarray,
    device : DeviceInfo,
    batch_size : int,
):
    model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[proc_id]

    tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base")
    max_sequence_length = 512

    model = model.to(model_device)

    num_batches = (len(data) + batch_size) // batch_size
    new_data = []
    with torch.no_grad():
        if proc_id == 0:
            iterator = tqdm(range(num_batches))
        else:
            iterator = range(num_batches)
        for i in iterator:
            text_batch = data[i*batch_size:(i+1)*batch_size]
            text_batch[text_batch == None] = ""
            input_ids = tokenizer(
                list(text_batch), return_tensors="pt", padding=True)["input_ids"]
            input_ids = input_ids[:,:max_sequence_length]
            input_ids = input_ids.to(model_device)
            embed_batch = model(input_ids).pooler_output
            new_data.append(embed_batch.cpu())
    new_data = torch.cat(new_data)
    return new_data

class DPRTextEmbeddingTransformConfig(pydantic.BaseModel):
    batch_size : int = 256

@column_transform
class DPRTextEmbeddingTransform(ColumnTransform):
    config_class = DPRTextEmbeddingTransformConfig
    name = "dpr_text_embedding"
    input_dtype = DBBColumnDType.text_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : DPRTextEmbeddingTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : 768,
        }

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> ColumnData:
        # Previous solution concat column name to data. Here we do not use it.
        data = column.data

        num_procs = 1 if len(device.gpu_devices) == 0 else len(device.gpu_devices)
        if num_procs > 1:
            logger.info("Spawn workers to generate embeddings using multi-GPUs.")
            ctx = mp.get_context('spawn')
            worklist = np.array_split(data, num_procs)
            with ctx.Pool(processes=num_procs) as pool:
                results = []
                for proc_id in range(num_procs):
                    rst = pool.apply_async(
                        _run_one_device,
                        (proc_id, num_procs, worklist[proc_id], device, self.config.batch_size)
                    )
                    results.append(rst)
                results = [rst.get() for rst in results]
            new_data = torch.cat(results)
        else:
            new_data = _run_one_device(0, 1, data, device, self.config.batch_size) 
        return [ColumnData(self.new_meta, new_data)]
