# DBInfer Benchmark (DBB)

DBInfer Benchmark (DBB) is a set of benchmarks for measuring machine
learning solutions over data stored as multiple tables.

## Install

Install DBInfer-Bench

```
pip install dbinfer-bench
```

## Getting datasets

To get a dataset,

```python
import dbinfer_bench as dbb 
dataset = dbb.load_rdb_data('diginetica')
```

See the full list of datasets and their data card in the accompanying paper.

| Dataset name  | Task names                          |
|:-------------:|:------------------------------------|
|`avs`          |`repeater`                           |
|`mag`          |`cite`, `venue`                      |
|`diginetica`   |`ctr`, `purchase`                    |
|`retailrocket` |`cvr`                                |
|`seznam`       |`charge`, `prepay`                   |
|`amazon`       |`rating`, `purchase`, `churn`        |
|`stackexchange`|`churn`, `upvote`                    |
|`outbrain-small`     |`ctr`                                |

## Understanding the dataset format

The dataset object obtained from `load_rdb_data` is of `DBBRDBDataset` class, which
contains the following properties:
* `metadata`: Metadata of the RDB dataset including table schema, relationships
  (primary key, foreign key), time column information, etc.
* `tables`: The RDB table data. Each table is a collection of
  columnar values stored as a dictionary of NumPy arrays.
* `tasks`: A list of tasks associated with the dataset.

A dataset can have multiple associated tasks. Each task is an `DBBRDBTask` object
with the following members:
* `metadata`: Task metadata including the prediction type, evaluation metric, etc.
* `train_set`, `validation_set`, `test_set`: Training, validation and test samples
  associated with the task. Similar to a row instance in tables, each sample can
  have heterogenous input features (e.g., a product can have name and price). Hence,
  samples are also stored as a dictionary of NumPy arrays.

See [this tutorial](./notebooks/dataset_guide.ipynb) for a walkthrough of the above concepts.

## Running baselines

The repository provides implementations of various baselines including popular
tabular models w/ or w/o auto-feature-engineering methods and Graph Neural Networks.
Since the running them consists of multiple steps such as data preprocessing, featurization,
graph construction and training, we package them into a python package `dbinfer`
with each step modularized as a commandline tool.

First, create the conda environment by:

```
bash conda/install-ubuntu-deps.sh
bash conda/create_conda_env.sh
```

It will setup an conda environment `dbinfer-gpu` or `dbinfer-cpu` depending
on your input. To update an existing environment according to upstream changes,
pass the `-o` option to the script to recreate the conda yaml, and then use
`conda env update --name <ENV_NAME> --file <CONDA_YAML>`.

Then, add a command-line alias to bashrc:

```
alias dbinfer='python -m dbinfer.main'
```

Then, create a workspace for saving model checkpoints:

```
mkdir workspace
```

### Methods using only a single table

We recommend using the preprocessed data (using data name `<DATASET>-single`) to
save preparation efforts.

<details>
  <summary>Click here to see the preprocessing details.</summary>

  ```bash
  # Dummy table creation, data normalization, featurization, etc.
  dbinfer preprocess <DATASET> transform <DATASET>-single
  ```
</details>

To train a baseline model,

```bash
dbinfer fit-tab <DATASET>-single <TASK>                \
  tabnn                      # model architecture  \
  -c <MODEL_CONFIG_YAML>     # model config        \
  -p workspace               # workspace path
```

Model configuration files (produced by HPO) are stored under `hpo_results/<DATASET>/<TASK>/`.

For example, to run the MLP baseline for the `ctr` task of `diginetica`:
```bash
dbinfer fit-tab diginetica-single ctr tabnn -c hpo_results/diginetica/ctr/single-mlp.yaml -p workspace
```

### Methods based on Deep Feature Synthesis (DFS)

We recommend using the preprocessed data (using data name `<DATASET>-dfs-<DEPTH>`) to
save preparation efforts. Note that when the depth is equal to one, only tables adjacent
to the target table are used to augment features, which corresponds to the "simple join"
baselines in our paper.

<details>
  <summary>Click here to see the preprocessing details.</summary>

  ```bash
  # Post-dfs processing including dummy table creation, key mapping, etc.
  dbinfer preprocess <DATASET> transform <DATASET>-pre-dfs -c configs/transform/pre-dfs.yaml
  # Run Deep Feature Synthesis
  dbinfer preprocess <DATASET>-pre-dfs dfs <DATASET>-post-dfs -c configs/dfs/dfs-<DEPTH>.yaml
  # Post-dfs processing including data normalization, extra featurization, etc.
  dbinfer preprocess <DATASET>-post-dfs transform <DATASET>-dfs-<DEPTH> -c configs/transform/post-dfs.yaml
  ```
</details>

To train a baseline model,

```bash
dbinfer fit-tab <DATASET>-dfs-<DEPTH> <TASK>           \
  tabnn                      # model architecture  \
  -c <MODEL_CONFIG_YAML>     # model config        \
  -p workspace               # workspace path
```

Model configuration files (produced by HPO) are stored under `hpo_results/<DATASET>/<TASK>/`.

For example, to run the DFS-2 + MLP baseline for the `ctr` task of `diginetica`:
```bash
dbinfer fit-tab diginetica-dfs-2 ctr tabnn -c hpo_results/diginetica/ctr/dfs-2-mlp.yaml -p workspace
```

### Graph Neural Networks

We use two graph construction algorithms, termed `r2n` and `r2ne`
to demonstrate the significance in such choice. Again, we recommend using the preprocessed
data (using data name `<DATASET>-<GRAPH_ALGO>`) to save preparation efforts.

<details>
  <summary>Click here to see the preprocessing details.</summary>

  ```bash
  # Dummy table creation, data normalization, featurization, etc.
  dbinfer preprocess <DATASET> transform <DATASET>-single
  # Graph construction.
  dbinfer construct-graph <DATASET>-single <GRAPH_ALGO> <DATASET>-<GRAPH_ALGO>
  ```
</details>

To train a baseline model,

```bash
dbinfer fit-gml <DATASET>-<GRAPH_ALGO> <TASK>          \
  <GNN_NAME>                 # GNN architecture    \
  -c <MODEL_CONFIG_YAML>     # model config        \
  -p workspace               # workspace path
```

Model configuration files (produced by HPO) are stored under `hpo_results/<DATASET>/<TASK>/`.

Available GNNs:
- `sage`: GraphSAGE
- `gat`: Graph Attention Network
- `hgt`: Heterogeneous Graph Transformer
- `pna`: Principle Neighborhood Aggregator

For example, to run the r2ne + GAT baseline for the `ctr` task of `diginetica`:
```bash
dbinfer fit-gml diginetica-r2ne ctr gat -c hpo_results/diginetica/ctr/r2ne-gat.yaml -p workspace
```

### Notes

* Some preprocessing steps (e.g., converting text fields into embeddings) may use multi-GPUs 
  automatically. To control the GPU devices in use, set the
  [`CUDA_VISIBLE_DEVICES` environment variable](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).
* The `dbinfer` command-line tool supports more than just `fit-gml` and `fit-tab`. Use `dbinfer --help`
  to see the full list of supported subcommands.
* The GNN solutions will try to utilize all CPU cores to accelerate data loading.
  To limit the number of CPU core resources used by each experiment, use the `NUM_VISIBLE_CPUS`
  environment variable. E.g., on a g4dn.metal instance with 96 cores and 8 GPUs, to run an experiment
  that consumes 1/8 of the resources:
  ```bash
  NUM_VISIBLE_CPUS=12 CUDA_VISIBLE_DEVICES=0 dbinfer fit-gml ...
  ```
* The default dataset download path is `<PROJECT_HOME>/datasets`. Use the `DBB_DATASET_HOME`
  environment variable to override that setting.
* You can also use the `dbinfer` commands to work with your local datasets. Just pass the local dataset path as the argument.
* When encountering the `received 0 items of ancdata` error, it means the solution may uses too
  much shared memory resources for data loading (a similar [PyTorch issue](https://github.com/pytorch/pytorch/issues/973)).
  Typically, limiting `NUM_VISIBLE_CPUS` will be useful. You could also try to
  increase the limit of open file descriptors (by default 1024).  Edit
  `/etc/security/limits.conf' and add the following two lines at the end to
  increase the limit to e.g. 16284:
  ```
  * hard nofile 16384
  * soft nofile 16384
  ```
  Afterwards, log out of your session and log back in again.  You can see if the change
  takes place by running `ulimit -n` in the command line.

## Cite Us

```
@article{dbinfer,
  title={4DBInfer: A 4D Benchmarking Toolbox for Graph-Centric Predictive Modeling on Relational DBs},
  author={Wang, Minjie and Gan, Quan and Wipf, David and Cai, Zhenkun and Li, Ning and Tang, Jianheng and Zhang, Yanlin and Zhang, Zizhao and Mao, Zunyao and Song, Yakun and Wang, Yanbo and Li, Jiahang and Zhang, Han and Yang, Guang and Qin, Xiao and Lei, Chuan and Zhang, Muhan and Zhang, Weinan and Faloutsos, Christos and Zhang, Zheng},
  journal={arXiv preprint arXiv:2404.18209},
  year={2024}
}
```

## Contributors

Thanks to the help from

* [Ning](https://github.com/NingLi670) on implementing graph construction.
* [Zhenkun](https://github.com/czkkkkkk) on implementing temporal sampling for GNNs.
* [Zunyao](https://github.com/DanielMao1) and Zhenkun on implementing a scalable DFS.
* [Jianheng](https://github.com/squareRoot3), [Yanlin](https://github.com/Lilyzhangyanlin), [Zizhao](https://github.com/SandMartex) on curating the datasets.
* Ning, [Yakun](https://github.com/Ereboas), [Yanbo](https://github.com/yanxwb) on implementing the baselines.
* [Jiahang](https://github.com/LspongebobJH), Jianheng on implementing AutoGluon-based solutions.
* And all the above for running the experiments and fixing countless bugs.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
