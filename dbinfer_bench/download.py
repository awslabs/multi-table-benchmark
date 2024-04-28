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


from typing import Tuple, Dict, Optional, List
from pathlib import Path
import os
import boto3
from tqdm import tqdm
import tarfile
import shutil
import yaml
import requests

__all__ = [
    "get_builtin_path_or_download",
    "download_or_get_path",
    "list_builtin",
]

DEFAULT_DOWNLOAD_CONFIG_FILE = Path(__file__).parent / 'download_config.yaml'

def list_builtin() -> List[str]:
    """List downloadable built-in datasets."""
    cfg = _get_download_cfg()
    return sorted(list(cfg['datasets'].keys()))

def download_or_get_path(dataset_name_or_path: str):
    cfg = _get_download_cfg()
    if dataset_name_or_path in cfg['datasets']:
        return get_builtin_path_or_download(dataset_name_or_path)
    else:
        return dataset_name_or_path

def get_builtin_path_or_download(
    dataset_name : str,
    version : Optional[str] = None,
) -> str:

    cfg = _get_download_cfg()

    assert dataset_name in cfg['datasets']

    # Use the PROJECT_HOME variable set by the conda environment as the default path.
    default_path = Path(os.environ.get("DBB_PROJECT_HOME", ".")) / 'datasets'
    data_home = Path(os.environ.get("DBB_DATASET_HOME", default_path))
    data_dir = data_home / dataset_name

    if version is None:
        version = cfg['datasets'][dataset_name]['version']
    version_file = data_dir / 'VERSION'

    download = True
    if os.path.exists(version_file):
        with open(version_file, 'rt') as f:
            local_ver = f.read().strip()
        if local_ver == version:
            download = False
        else:
            print(f"Request version is ({version}) but found local version ({local_ver}). Re-downloading...")
            shutil.rmtree(data_dir)

    

    if download:
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=False)

        source = cfg['source']
        if source.startswith("s3://"):
            _download_s3(cfg, dataset_name, version, data_home)
        else:
            _download_default(cfg, dataset_name, version, data_home)

        with open(version_file, 'w') as f:
            f.write(version)

    return data_dir

def _download_s3(
    cfg: Dict,
    dataset_name : str,
    version: str,
    data_home: Path
):
    source = cfg['source']
    parts = source[5:].split("/")
    bucket_name = parts[0]
    tarfilename = f"{version}-{dataset_name}.tar"
    prefix = '/'.join(parts[1:] + [tarfilename])

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    objs = list(bucket.objects.filter(Prefix=prefix))
    if len(objs) == 0:
        raise RuntimeError(f"Dataset {dataset_name} not available.")
    for obj in objs:
        print(f"Dowloading {obj.key} ...")
        download_path = data_home / tarfilename
        with tqdm(total=obj.size, unit='B', unit_scale=True) as progress:
            def progress_callback(bytes_transferred):
                progress.update(bytes_transferred)
            bucket.download_file(obj.key, download_path, Callback=progress_callback)

    print(f"Extracting {download_path} ...")
    with tarfile.open(download_path, 'r:*') as tar:
        tar.extractall(path=data_home)

def _download_default(
    cfg: Dict,
    dataset_name : str,
    version: str,
    data_home: Path
):
    tarfilename = f"{version}-{dataset_name}.tar"
    url = cfg['source'] + "/" + tarfilename

    req = requests.get(url, stream=True, verify=True)
    if req.status_code != 200:
        raise RuntimeError("Failed downloading url %s" % url)
    # Get the total file size.
    total_size = int(req.headers.get("content-length", 0))
    download_path = data_home / tarfilename
    with tqdm(
        total=total_size, unit="B", unit_scale=True
    ) as bar:
        with open(download_path, "wb") as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"Extracting {download_path} ...")
    with tarfile.open(download_path, 'r:*') as tar:
        tar.extractall(path=data_home)

def _get_download_cfg():
    cfg_file = os.environ.get("DBB_DOWNLOAD_CONFIG_FILE", DEFAULT_DOWNLOAD_CONFIG_FILE)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

        return cfg
