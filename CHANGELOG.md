# Changelog

Every release is on GitHub (Releases) with these notes. Update: `git pull`, then `./setup.sh` (it rebuilds only what
changed).

## v0.5.0 - 2026-11-21

**The k-quants and Q8_0 on the GPU, and the GPU by default.**

- `gpuq`: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (with an imatrix: ggml's `quantize_row_*_impl` search; without one: the
  `*_ref` search, as `ggml_quantize_chunk` picks) and Q8_0, byte-identical to ggml. IQ4_NL / IQ4_XS without an
  imatrix too. `gpuq_quantize_batch` / `gpuq_requantize_batch` quantize many matrices a launch (each its own imatrix
  row); `gpuq_can_quantize(type, with_imatrix)` says what the library takes. Rebuild with `./build.sh` (or
  `./setup.sh`).
- `requant_experts.py` and `convert_hf.py` quantize on the GPU by default: they find `libgpuq_cuda.so` or `libgpuq.so`,
  and without one (or without a GPU) print why and use the CPU. `--cpu` forces the CPU, `--gpu=N` picks the device.
  Experts go to the GPU in batches (`GPUQ_BATCH_MB`, default 2048) with the next batch read while the GPU works; big
  2-D tensors in row batches (Hy4-flash's 118 GiB per-layer embedding). With `--dump-device=cuda` the LDLQ loop now
  runs on the GPU without `--gpu` (`--ldlq-cpu` keeps the CPU loop).
- Checked on an RTX 5090: `test_gpuq.py` all 15 types identical on random weights (with zero, constant, one-signed and
  tiny blocks) and on Hy4-flash BF16 experts; Hy4-flash re-made from its BF16 GGUF in A's layout (IQ2_XXS / IQ3_XXS
  experts, Q8_0 elsewhere, IQ4_NL embedding table) in 5 min 22 s against 51 min 33 s for llama-quantize on 32
  threads, every one of the 1,380 tensors identical to llama-quantize's file.
- ROCm: the self-test (random weights, all 15 types, both imatrix paths) passes on a Radeon 8060S iGPU (gfx1151);
  real weights and the R9700 not run yet. Not run yet either: `convert_hf.py` end to end with the new defaults.

## v0.4.0 - 2026-11-20

**Hy4-flash (`hy_v4_flash`) in trench-gguf's llama.cpp copy**: convert, imatrix, quantize and check Tencent's
Hy4-flash with the usual steps ([recipes/hy4](recipes/hy4/README.md)).

- `patches/llama-hy4.diff` (applied by `setup.sh` to `third_party/llama.cpp` and, before the expert-upload patch, to
  `third_party/llama.cpp-gpu`): the `hy_v4_flash` arch (text only), ggml type 43 `F8_E4M3` (plain E4M3, Maya's Engram
  table), `llama-quantize` rules (Engram table IQ4_NL by default, other non-expert matrices at least Q8_0), the
  `HYV4FlashForCausalLM` converter. Names and keys match the published GGUFs; Maya-XS and nace Q3_K_M load as they are.
- Checked against Abyss on the same ids (KL over the whole vocabulary): Maya-XS mix459 0.118 / 93.4% same top token,
  ja2k (2,046 positions) 0.106 / 88.4%; nace Q3_K_M mix459 0.083 / 93.7%, ja2k 0.062 / 91.8%. Against the numpy
  reference on a tiny random model with the real layout: KL 0.000002 (16 tokens); on Maya-XS's first 192 mix459
  tokens against that reference: llama.cpp 0.021, Abyss 0.017 (llama.cpp against Abyss 0.029).
- `llama-imatrix` on Maya-XS (4 chunks of 512, experts from the host, 5090): 814 entries, every one of the 123 expert
  tensors (41 MoE layers x gate / up / down) included; 23.4 s per pass of 2,048 tokens, peak RSS 47.9 GB.
- `llama-quantize` on Maya-XS (experts copied, Engram table FP8 -> IQ4_NL, dense Q6_K -> Q8_0): 100.7 -> 75.9 GiB in
  22 min on 16 CPU threads; the result loads and runs (mix459 KL 0.175 against Abyss on the original, rounding an
  already-FP8 table twice, and `output` fell to Q5_K; `output` now follows the Q8_0 floor too, checked with
  `--dry-run`).
- `tools/teacher cmp`: KL against another engine's full log-softmax dump (Abyss's `STRATA_GLM_SCORE_DUMP`).
- Not run yet (beta): the converter on the full 440 GB checkpoint (tested on a synthetic checkpoint with the real layout and
  by range reads of real tensors against nace's GGUF).

## v0.3.1 - 2026-11-04

The tools behind [zephel01/Qwen3.8-Flash-Next-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-12G32G-GGUF)
and the DeepSeek V4.1 Compact JA256 GGUF.

- `tools/requant_experts.py` reads NVFP4 sources: the per-expert `.scale` is folded into the weights; the `.scale` /
  `.input_scale` tensors of re-made experts are left out of the output.
- `tools/hessian_dump --parse-special`: special-token text in the calibration file becomes those tokens, for a
  calibration that starts with the model's own chat transcripts. A plain-text calibration left a Qwen3.8 build ending
  turns right after `</think>` with no answer (and thinking 3,000+ tokens on short questions); chat transcripts fixed
  both at about the same KL (0.327 -> 0.339).
- `hessian_dump`: a last MoE layer pruned to no rows (Qwen3.8 layer 47) counts as short and triggers the
  `--all-logits` pass.
- `tools/merge_experts.py`: a layer item may name the tensors to take (`5-19:down,30:gate+up`).
- `tools/prune_experts.py` (keep the listed experts, byte for byte) and `tools/requant_table.py` (re-quantize a big
  table on the GPU; reads unsloth's FP8 type 43, an MXFP8 row layout).

## v0.3.0 - 2026-11-03

**LDLQ: better rounding at the same types and bytes, on the GPU.** The tools behind
[zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF),
with the steps in the README (LDLQ walkthrough).

- `tools/hessian_dump`: one llama.cpp pass over a calibration text writes every layer's expert input, the routing ids
  and weights (`--dense` also the inputs of the dense tensors). Built by `setup.sh`.
- `tools/requant_experts.py --dump=<dir>`: error-feedback rounding (GPTQ / block-LDLQ around ggml's own quantizers,
  `tools/ldlq.py`) with per-expert Hessians from the dump; `--order=act`, `--damp`, `--damp-iq1`, `--dense-dump` for
  2-D tensors (k-quants too), `--only-matched` (only the re-made tensors, for `merge_experts.py`). With `--gpu
  --dump-device=cuda` the whole loop runs on the GPU (DeepSeek V4.1, 120 expert tensors: 77 min on an RTX 5090).
  DeepSeek V4.1 at unsloth UD-IQ1_M's exact layout: KL 0.194 (plain rounding 0.404, UD-IQ1_M 0.198; 8 sequences).
- gpuq: a device-pointer API (quantize / dequantize super-blocks on the caller's stream, ggml's `to_float` of the
  i-quants on the GPU).
- CUDA checked on an RTX 5090: the self-test is byte-identical for all 9 types and the 4 GPU-read sources.
- `setup.sh --torch-gpu [--torch-index cu130|rocm7.2]`: the converter's torch version built for the GPU.
- `tools/gguf_set_split_no.py`, `recipes/deepseek-v4.1-1017-ldlq/` (the 132G / 139G rules files).
- `hessian_dump` makes its out dir, and when llama.cpp prunes the last layer to the output rows (OLMoE, Qwen MoE) it
  starts again with a logit for every token (`--all-logits`); `requant_experts.py` stops on a `--dump` without
  `info.txt` instead of rounding plain with no word.

## v0.2.1 - 2026-10-30

- `tools/imatrix.py calib --size=micro|tiny|small|medium --scale=N`: a bigger calibration text from the same pinned
  dataset (`--size=small --scale=7`: 4,239,316 bytes, ~1M tokens on DeepSeek's tokenizer).
- `tools/imatrix.py stats <imatrix.gguf>`: tokens per expert (min, 1st percentile, median, how many under 1,000). The
  ~146K-token imatrix of DeepSeek V4 Flash 0731 leaves 9.4% of its experts under 1,000 tokens.
- README: the `teacher kl` example is a real run now (DeepSeek V4 Flash 0731).

## v0.2.0 - 2026-10-29

**More types on the GPU, the source read on the GPU, Hugging Face conversion, imatrix on the GPU, a quality check.**

- GPU quantizer: IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS and IQ4_NL join IQ1_S, IQ1_M, IQ2_XXS and IQ2_S - 9 types, all
  byte-identical to ggml in the self-test, 2-21x one CPU thread a type (AI PRO R9700).
- Sources in F32, F16, BF16, Q8_0 and MXFP4 are dequantized on the GPU too (`gpuq_requantize`); the result is the
  same bytes as the CPU path.
- `tools/requant_experts.py`: rules may name 2-D tensors (attention, shared experts, embeddings), so a BF16-attention
  source comes out at a published file's size type for type. A 2-D tensor whose imatrix does not fit (DeepSeek V4's
  grouped `attn_output_a`) is quantized without it when the type allows.
- `tools/rules_from_gguf.py`: a published GGUF's layout as a rules file (`--all` for every tensor, `--shift=N` one step
  smaller or bigger, `--sizes=` the size on your source).
- `tools/convert_hf.py`: safetensors (BF16 / F16 / FP8) straight to a quantized GGUF through llama.cpp's own
  converter, no full-size intermediate; new architectures work the day llama.cpp supports them.
- `tools/imatrix.py`: calibration text from eaddario/imatrix-calibration at a pinned revision (byte budgets per
  language, the same text for every model), llama-imatrix with flags that work for models bigger than RAM, `--gpu`,
  `--ub`, saved once at the end.
- `patches/llama-expert-upload.diff` (setup.sh applies it to its own copy of llama.cpp): experts reach the GPU through
  pinned buffers. Without it the ROCm imatrix of a model bigger than RAM stalled; DeepSeek V4 Flash 0731 (150 GB on a
  94 GB machine): CPU 46 s per 1,000 tokens, patched GPU 16 s.
- `tools/teacher.cpp`: reference top-32 log-probs, then KL / 99th percentile / same top token, on the CPU or the GPU
  (Qwen3.8 UD-IQ4_XS, 628 tokens: 30.4 s on the CPU, 12.5 s on the GPU).
- `setup.sh`: 8 steps (the patched GPU llama.cpp, teacher, the converter's packages); the tools read `LIBGGML` /
  `GPUQ_LIB` from `trench.env` (a system libggml of another version gave other IQ1_M bytes).

## v0.1.0 - 2026-10-29

**First release: GPU i-quants byte-identical to ggml.**

- IQ1_S, IQ1_M, IQ2_XXS, IQ2_S on the GPU (ROCm; CUDA builds), ggml's quantizers line for line.
- `tools/requant_experts.py` (re-quantize the routed experts of any GGUF ggml reads, by regex rules, with an imatrix),
  `tools/merge_experts.py` (mix two GGUF sets per layer), `setup.sh`.
- llama-quantize's fallback types for rows a type cannot hold; tensors without imatrix data quantized without it where
  the type allows.
- A copy-paste DeepSeek V4.1 Flash walkthrough and a supported-models table in the README.
