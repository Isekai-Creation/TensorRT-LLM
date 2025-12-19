#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-}}"
QFORMAT="${QFORMAT:-${3:-full_prec}}"

if [[ -z "${MODEL_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: MODEL_DIR=... OUTPUT_DIR=... $0 [model_dir] [output_dir] [qformat]" >&2
  exit 1
fi

CALIB_DATASET="${CALIB_DATASET:-cnn_dailymail}"
CALIB_SIZE="${CALIB_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-512}"
TOKENIZER_MAX_SEQ_LENGTH="${TOKENIZER_MAX_SEQ_LENGTH:-2048}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
if [[ -z "${KV_CACHE_DTYPE}" ]]; then
  KV_CACHE_DTYPE="$(python - <<'PY'
import os

kv = "fp8"
force = os.environ.get("FORCE_NVFP4", "").strip()

try:
    import torch
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major >= 10:
            try:
                import modelopt.torch.quantization as mtq
                if hasattr(mtq, "NVFP4_KV_CFG"):
                    kv = "nvfp4"
            except Exception:
                pass
except Exception:
    pass

if force:
    kv = "nvfp4"

print(kv)
PY
)"
fi

echo "Using kv_cache_dtype=${KV_CACHE_DTYPE}"

IS_MXFP4="$(MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import json
import pathlib
import os

model_dir = pathlib.Path(os.environ.get("MODEL_DIR", "")).expanduser()
cfg_path = model_dir / "config.json"
if not cfg_path.is_file():
    print("0")
    raise SystemExit(0)
try:
    data = json.loads(cfg_path.read_text())
except Exception:
    print("0")
    raise SystemExit(0)

qcfg = data.get("quantization_config") or {}
print("1" if qcfg.get("quant_method") == "mxfp4" else "0")
PY
)"

if [[ "${IS_MXFP4}" == "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  if [[ "${KV_CACHE_DTYPE}" == "nvfp4" ]]; then
    KV_CACHE_ALGO="NVFP4"
  else
    KV_CACHE_ALGO="FP8"
  fi
  YAML_PATH="${OUTPUT_DIR}/kv_cache_quant.yaml"
  cat > "${YAML_PATH}" <<YAML
quant_config:
  kv_cache_quant_algo: ${KV_CACHE_ALGO}
YAML
  echo "Detected MXFP4 pre-quantized GPT-OSS; skipping export."
  echo "Use: trtllm-serve ${MODEL_DIR} --extra_llm_api_options ${YAML_PATH}"
  exit 0
fi

python examples/quantization/quantize.py \
  --model_dir "${MODEL_DIR}" \
  --qformat "${QFORMAT}" \
  --kv_cache_dtype "${KV_CACHE_DTYPE}" \
  --calib_dataset "${CALIB_DATASET}" \
  --calib_size "${CALIB_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_max_seq_length "${CALIB_MAX_SEQ_LENGTH}" \
  --tokenizer_max_seq_length "${TOKENIZER_MAX_SEQ_LENGTH}" \
  --tp_size "${TP_SIZE}" \
  --pp_size "${PP_SIZE}" \
  --cp_size "${CP_SIZE}" \
  --device "${DEVICE}" \
  --device_map "${DEVICE_MAP}" \
  --output_dir "${OUTPUT_DIR}"
