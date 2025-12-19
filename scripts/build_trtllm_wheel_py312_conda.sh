#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate the Python 3.12 conda environment.
if [[ -f "/workspace/aimo/miniconda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/workspace/aimo/miniconda/etc/profile.d/conda.sh"
  conda activate tensorrt-3.12
else
  echo "ERROR: conda.sh not found; cannot activate tensorrt-3.12 env." >&2
  exit 1
fi

# Point build to CUDA 12.8.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Normalize CLI flags: build_wheel.py expects --cuda_architectures.
args=()
ucx_override=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-architectures=*)
      args+=("--cuda_architectures=${1#*=}")
      ;;
    --cuda-architectures)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --cuda-architectures expects a value." >&2
        exit 1
      fi
      args+=("--cuda_architectures=${1}")
      ;;
    *)
      if [[ "$1" == *"ENABLE_UCX"* ]]; then
        ucx_override="1"
      fi
      args+=("$1")
      ;;
  esac
  shift
done

if [[ -z "${ucx_override}" ]]; then
  args+=("--extra-cmake-vars" "ENABLE_UCX=OFF")
fi

# Use the active environment without creating a nested venv.
exec python "${ROOT_DIR}/scripts/build_wheel.py" --no-venv "${args[@]}"
