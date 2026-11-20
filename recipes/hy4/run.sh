#!/usr/bin/env bash
# recipes/hy4/run.sh - Tencent Hy4-flash from the BF16 checkpoint to a low-bit GGUF, step by step (a stub: the steps
# are the ones checked on existing GGUFs - see README.md for what was and was not run).
#
#   ./run.sh download | convert | calib | imatrix | quantize | check | all
#
# Needs ./setup.sh first (third_party/llama.cpp with patches/llama-hy4.diff, the GPU build, .venv) and
# `source trench.env`. Set WORK to a folder with ~920 GB free (download 440.5 GB + BF16 GGUF ~438 GB + the result).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/mnt/data/hy4}
HF_DIR=$WORK/hf                      # tencent/Hy4-flash
BF16=$WORK/Hy4-flash-BF16.gguf       # the converter's output (one file, ~438 GB)
CALIB=$WORK/calib-hy4.txt
IMATRIX=$WORK/Hy4-flash.imatrix.gguf
OUT=$WORK/Hy4-flash-IQ2_XXS-hy4.gguf
LLAMA_DIR=${LLAMA_DIR:-$ROOT/third_party/llama.cpp}
LLAMA_BIN=${LLAMA_BIN:-$ROOT/third_party/llama.cpp-gpu/build-gpu/bin}
PY=${PY:-$ROOT/.venv/bin/python}

step=${1:-all}
run() { [ "$step" = all ] || [ "$step" = "$1" ]; }

if run download; then
  # 148 safetensors shards, 440,520,833,910 bytes (model.safetensors.index.json); the vision tower and the dfly
  # drafter are in there too and are skipped by the converter
  hf download tencent/Hy4-flash --local-dir "$HF_DIR" \
    --include "*.safetensors" --include "*.json" --include "tokenizer*" --include "chat_template.jinja"
fi

if run convert; then
  # the converter needs a transformers that knows the checkpoint's tokenizer class (TokenizersBackend: 5.x)
  "$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF_DIR" --outtype bf16 --outfile "$BF16"
fi

if run calib; then
  # tools/imatrix.py's mix (en / ja / zh / code / math / tools) at 7x the small budgets (~1M tokens on DeepSeek's
  # tokenizer), then more Japanese: append your own Japanese text files (CALIB_JA="a.txt b.txt") to weigh it up
  "$PY" "$ROOT/tools/imatrix.py" calib "$CALIB" --size=small --scale=7
  for f in ${CALIB_JA:-}; do cat "$f" >> "$CALIB"; done
fi

if run imatrix; then
  # dense part on the GPU, experts from the host (op offload); -c 2048: the indexer's selection does not start
  # before 2048 keys, so multi-sequence batches are fine
  "$PY" "$ROOT/tools/imatrix.py" run "$BF16" "$CALIB" "$IMATRIX" --llama-bin="$LLAMA_BIN" --gpu --ub=8192
  "$PY" "$ROOT/tools/imatrix.py" stats "$IMATRIX"
fi

if run quantize; then
  # experts IQ2_XXS, the Engram table IQ4_NL (llama-quantize's hy_v4_flash default, named here to be explicit), every
  # other matrix Q8_0 (the hy_v4_flash floor for non-expert tensors). Japanese-sensitive upgrades: the first and the
  # last MoE layers' down projections one step up (a starting point, not measured on this model yet)
  "$LLAMA_BIN/llama-quantize" --imatrix "$IMATRIX" \
    --tensor-type per_layer_token_embd=iq4_nl \
    --tensor-type 'blk\.(1|2|3|39|40|41)\.ffn_down_exps=iq3_xxs' \
    "$BF16" "$OUT" IQ2_XXS "$(nproc)"
fi

if run check; then
  # KL against the BF16 GGUF on a Japanese and an English token file (comma-separated ids, BOS 120000 first)
  for ids in ${CHECK_IDS:-}; do
    "$ROOT/tools/teacher" ref "$BF16" "$WORK/ref" "$ids" --gpu
    "$ROOT/tools/teacher" kl "$OUT" "$WORK/ref" "$ids" --gpu
  done
fi
