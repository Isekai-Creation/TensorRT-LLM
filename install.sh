#!/usr/bin/env bash
set -euxo pipefail
cd /kaggle/working/TensorRT-LLM

export TRTLLM_SKIP_REQUIREMENTS="${TRTLLM_SKIP_REQUIREMENTS:-1}"

DATASET_ID="${TRT_KAGGLE_DATASET_ID:-mnhhang/tensorrt-10-14-1-48}"
DATASET_VERSION="${TRT_KAGGLE_DATASET_VERSION:-1}"
DATASET_DIR="${TRT_KAGGLE_TRT_DIRNAME:-TensorRT-10.14.1.48}"
TRT_ROOT="${TRT_ROOT:-}"
if [[ -z "${TRT_ROOT}" && -d "/kaggle/input/tensorrt-10-14-1-48/${DATASET_DIR}" ]]; then
  TRT_ROOT="/kaggle/input/tensorrt-10-14-1-48/${DATASET_DIR}"
elif [[ -z "${TRT_ROOT}" ]]; then
  echo "[install] Downloading TensorRT dataset from Kaggle" >&2
  tmp_root_file="$(mktemp)"
  KAGGLEHUB_DISABLE_PROGRESS=1 python3 - <<'PY' "${tmp_root_file}"
import os
import pathlib
import sys

try:
    import kagglehub
except Exception as exc:
    raise SystemExit(
        "kagglehub is required to download TensorRT. Install it with: pip install kagglehub"
    ) from exc

dataset = os.environ.get("TRT_KAGGLE_DATASET_ID", "mnhhang/tensorrt-10-14-1-48")
path = kagglehub.dataset_download(dataset)
pathlib.Path(sys.argv[1]).write_text(path)
PY
  TRT_ROOT="$(cat "${tmp_root_file}")"
  rm -f "${tmp_root_file}"
  if [[ -d "${TRT_ROOT}/${DATASET_DIR}/lib" ]]; then
    TRT_ROOT="${TRT_ROOT}/${DATASET_DIR}"
  fi
fi
if [[ -z "${TRT_ROOT}" ]]; then
  echo "ERROR: TRT_ROOT is not set or TensorRT dataset is missing." >&2
  exit 1
fi

TRT_SDK=/kaggle/working/trt_sdk
DRV=/kaggle/working/driver_shims

mkdir -p "$DRV"
ln -sf /usr/local/nvidia/lib64/libcuda.so.1      "$DRV/libcuda.so"
ln -sf /usr/local/nvidia/lib64/libnvidia-ml.so.1 "$DRV/libnvidia-ml.so"

mkdir -p "$TRT_SDK/lib"
ln -sfn "$TRT_ROOT/include" "$TRT_SDK/include"
for base in libnvinfer libnvinfer_plugin libnvonnxparser libnvinfer_lean libnvinfer_dispatch libnvinfer_vc_plugin; do
  versioned_so=""
  for candidate in "${TRT_ROOT}/lib/${base}.so."*; do
    if [[ -f "${candidate}" ]]; then
      versioned_so="${candidate}"
      break
    fi
  done
  if [[ -n "${versioned_so}" ]]; then
    ln -sf "${versioned_so}" "$TRT_SDK/lib/${base}.so.10"
    ln -sf "${versioned_so}" "$TRT_SDK/lib/${base}.so"
  fi
done

export LD_LIBRARY_PATH="$TRT_SDK/lib:$DRV:/usr/local/nvidia/lib64:${TRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$TRT_SDK/lib:$DRV:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs:${TRT_ROOT}/lib:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$TRT_SDK:$DRV:/usr/local/nvidia:/usr/local/cuda:${CMAKE_PREFIX_PATH:-}"
# Set the include path globally for the notebook session
NVTX_INC=/kaggle/working/cuda-12.8/nsight-systems-2024.6.2/target-linux-x64/nvtx/include
export CPATH="${CPATH:-}:$NVTX_INC"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}:$NVTX_INC"

TMPDIR=/dev/shm/tmp

mkdir -p "$TMPDIR"
export TMPDIR="$TMPDIR"

NCCL_ROOT="$(python3 - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    candidate = Path(root) / "nvidia" / "nccl"
    if (candidate / "lib").is_dir():
        print(candidate)
        break
PY
)"
if [[ -n "${NCCL_ROOT}" ]]; then
  export NCCL_ROOT
  export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${NCCL_ROOT}/lib:${LIBRARY_PATH:-}"
  export CMAKE_PREFIX_PATH="${NCCL_ROOT}:${CMAKE_PREFIX_PATH:-}"
fi

python3 ./scripts/build_wheel.py \
  --no-venv \
  # --clean --clean_wheel \
  --cuda_architectures "90-real" \
  --trt_root "$TRT_SDK" \
  -D CMAKE_PREFIX_PATH="$TRT_SDK" \
  -D CUDAToolkit_ROOT=/usr/local/cuda \
  -D FAST_BUILD=ON \
  -D BUILD_DEEP_EP=OFF \
  -D BUILD_DEEP_GEMM=OFF \
  -D BUILD_FLASH_MLA=OFF \
  -D ENABLE_UCX=OFF \
  -D NVTX_DISABLE=ON
