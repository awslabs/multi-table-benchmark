#!/bin/bash

readonly CUDA_VERSIONS="11.7"
readonly TORCH_VERSION="1.13.1"
readonly PYTHON_VERSION="3.9"
readonly DGL_VERSION="2.1a240205"

usage() {
cat << EOF
usage: bash $0 OPTIONS
examples:
  bash $0 -c
  bash $0 -g 11.7
  bash $0 -g 11.7 -p 3.9
  bash $0 -g 11.7 -p 3.9 -t 1.13.0

Create a conda environment for the project.

OPTIONS:
  -h           Show this message.
  -c           Create dev environment in CPU mode.
  -d           Only display environment YAML file instead of creating it.
  -f           Force creation of environment (removing a previously existing 
               environment of the same name).
  -g           Create environment in GPU mode with specified CUDA version,
               supported: ${CUDA_VERSIONS}.
  -o           Save environment YAML file to specified path.
  -p           Create environment based on specified python version.
  -s           Run silently which indicates always 'yes' for any confirmation.
  -t           Create environment based on specified PyTorch version such
               as ${TORCH_VERSION}.
  -l           Create environment based on specified DGL version such as ${DGL_VERSION}.
EOF
}

validate() {
  values=$(echo "$1" | tr "," "\n")
  for value in ${values}
  do
    if [[ "${value}" == $2 ]]; then
      return 0
    fi
  done
  return 1
}

confirm() {
  echo "Continue? [yes/no]:"
  read confirm
  if [[ ! ${confirm} == "yes" ]]; then
    exit 0
  fi
}

# Parse flags.
while getopts "cdfg:ho:p:st:" flag; do
  case "${flag}" in
    c)
      cpu=1
      ;;
    d)
      dry_run=1
      ;;
    f)
      force_create=1
      ;;
    g)
      cuda_version=${OPTARG}
      ;;
    h)
      usage
      exit 0
      ;;
    o)
      output_path=${OPTARG}
      ;;
    p)
      python_version=${OPTARG}
      ;;
    s)
      always_yes=1
      ;;
    t)
      torch_version=${OPTARG}
      ;;
    l)
      dgl_version=${OPTARG}
      ;;
    :)
      echo "Error: -${OPTARG} requires an argument."
      exit 1
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ -n ${cuda_version} && ${cpu} -eq 1 ]]; then
  echo "Only one mode can be specified."
  exit 1
fi

if [[ -z ${cuda_version} && -z ${cpu} ]]; then
  usage
  exit 1
fi

if [[ -z "${torch_version}" ]]; then
  torch_version=${TORCH_VERSION}
fi

if [[ -z "${dgl_version}" ]]; then
  dgl_version=${DGL_VERSION}
fi

# Set up CPU mode.
if [[ ${cpu} -eq 1 ]]; then
  torchversion=${torch_version}"+cpu"
  dgl_package_link="https://data.dgl.ai/wheels-test/repo.html"
  name="dbinfer-cpu"
fi

# Set up GPU mode.
if [[ -n ${cuda_version} ]]; then
  if ! validate ${CUDA_VERSIONS} ${cuda_version}; then
    echo "Error: Invalid CUDA version."
    usage
    exit 1
  fi

  echo "Confirm the installed CUDA version matches the specified one."
  [[ -n "${always_yes}" ]] || confirm

  torchversion=${torch_version}"+cu"${cuda_version//[-._]/}
  dgl_package_link="https://data.dgl.ai/wheels-test/cu"${cuda_version//[-._]/}"/repo.html"
  dgl_version=${dgl_version}"+cu"${cuda_version//[-._]/}
  name="dbinfer-gpu"
fi

# Set python version.
if [[ -z "${python_version}" ]]; then
  python_version=${PYTHON_VERSION}
fi

echo "Confirm you are excuting the script from your project root directory."
echo "Current working directory: ${PWD}"
[[ -n "${always_yes}" ]] || confirm

# Prepare the conda environment yaml file.
rand=$(echo "${RANDOM}" | md5sum | head -c 20)
mkdir -p /tmp/${rand}
yaml_path="/tmp/${rand}/env.yml"
cp conda/env.yml.template ${yaml_path}
sed -i "s|__NAME__|${name}|g" ${yaml_path}
sed -i "s|__PYTHON_VERSION__|${python_version}|g" ${yaml_path}
sed -i "s|__TORCH_VERSION__|${torchversion}|g" ${yaml_path}
sed -i "s|__PROJECT_HOME__|${PWD}|g" ${yaml_path}
sed -i "s|__DGL_PACKAGE_LINK__|${dgl_package_link}|g" ${yaml_path}
sed -i "s|__DGL_VERSION__|${dgl_version}|g" ${yaml_path}

# Ask for final confirmation.
echo "--------------------------------------------------"
cat ${yaml_path}
echo "--------------------------------------------------"
echo "Create a conda enviroment with the config?"
[[ -n "${always_yes}" ]] || confirm

# Save YAML file to specified path
if [[ -n "${output_path}" ]]; then
  cp ${yaml_path} ${output_path}
  echo "Environment YAML file has been saved to ${output_path}."
fi

# Create conda environment.
if [[ -z "${dry_run}" ]]; then
  conda_args=""
  if [[ -n "${force_create}" ]]; then
    conda_args="${conda_args} --force "
  fi
  conda env create -f ${yaml_path} ${conda_args}
else
  echo "Running in dry mode, so creation of conda environment is skipped."
fi

# Clean up created tmp conda environment yaml file.
rm -rf /tmp/${rand}
exit 0
