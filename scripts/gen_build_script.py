from pathlib import Path
import os

datasets = [
    'amazon',
    'avs',
    'diginetica',
    'mag',
    'outbrain-small',
    'retailrocket',
    'seznam',
    'stackexchange',
]

version = "20240304"
workspace = Path("datasets") / version

script = f"""#!/bin/bash

CMD="python -m dbinfer.main"

mkdir -p {workspace}

"""

for dataset_name in datasets:
    cmds = []
    data_to_tar = []
    # Step 1: transform for single table baselines and also for graph construction
    single_data = workspace / f"{dataset_name}-single"
    xform_cmd = f"$CMD preprocess {dataset_name} transform {single_data}"
    cmds.append(xform_cmd)
    data_to_tar.append(single_data)
    # Step 3: construct graph
    r2ne_data = workspace / f"{dataset_name}-r2ne"
    r2n_data = workspace / f"{dataset_name}-r2n"
    r2ne_cmd =  f"$CMD construct-graph {single_data} r2ne {r2ne_data}"
    r2n_cmd =  f"$CMD construct-graph {single_data} r2n {r2n_data}"
    cmds += [r2ne_cmd, r2n_cmd]
    data_to_tar += [r2ne_data, r2n_data]
    # Step 4: pre-DFS transform
    pre_dfs_data = workspace / f"{dataset_name}-pre-dfs"
    pre_dfs_cmd = f"$CMD preprocess {dataset_name} transform {pre_dfs_data} -c configs/transform/pre-dfs.yaml"
    cmds.append(pre_dfs_cmd)
    for depth in range(1, 4):
        # Step 5: DFS
        dfs_raw_data = workspace / f"{dataset_name}-dfs-{depth}-raw"
        dfs_cmd = f"$CMD preprocess {pre_dfs_data} dfs {dfs_raw_data} -c configs/dfs/dfs-{depth}-sql.yaml"
        # Step 6: post-DFS transform
        dfs_data = workspace / f"{dataset_name}-dfs-{depth}"
        post_dfs_cmd = f"$CMD preprocess {dfs_raw_data} transform {dfs_data} -c configs/transform/post-dfs.yaml"

        clean_up_cmd = f"rm -rf {dfs_raw_data}"
        
        cmds += [dfs_cmd, post_dfs_cmd, clean_up_cmd]
        data_to_tar.append(dfs_data)

    # Cleanup
    cmds.append(f"rm -rf {pre_dfs_data}")

    # Tar
    cmds.append(f"cd {workspace}")
    for data_path in data_to_tar:
        data_name = data_path.name
        tar_path = f"{version}-{data_name}.tar"
        tar_cmd = f"tar cvf {tar_path} {data_name}"
        cmds.append(tar_cmd)
    cmds.append(f"cd ..")

    script += f"""
echo ">>>>>> Building {dataset_name} ..."
"""
    script += '\n'.join(cmds)
    script += """
echo "<<<<<< Done"
"""

print(script)
with open('build_script.sh', 'w') as f:
    f.write(script)
