# Hy4-flash (hy_v4_flash): from the BF16 checkpoint to a low-bit GGUF

[Japanese](README.ja.md)

Tencent's [Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (`HYV4FlashForCausalLM`, 219B parameters, 42 layers,
512 experts, 16 active) is not in upstream llama.cpp. `patches/llama-hy4.diff` adds it to trench-gguf's own llama.cpp
copy (`./setup.sh` applies it to `third_party/llama.cpp` and `third_party/llama.cpp-gpu`), so the usual pipeline works:
convert, imatrix, quantize, quality check. Text only (no vision tower, no dfly drafter).

The arch name, tensor names and metadata keys are the ones of the published hy_v4_flash GGUFs
(peasantsmith/Hy4-Flash-Maya, nace-ai/Hy4-flash-GGUF): those files load as they are.

## What the patch adds

| Part | What |
| --- | --- |
| `hy_v4_flash` model (`src/models/hy-v4-flash.cpp`) | 4 hyper-connection streams; gated MLA (q LoRA 1024, shared 512 latent + 64 rope key, a learned sink per head, a sigmoid gate on every output element, post-attention norm); 30 sliding-window layers (64 heads, 512 keys, RoPE 1e4) and 12 indexed layers (32 heads, RoPE 1e7, the compressed indexer: groups of 4 keys pooled by a gated softmax, the top 511 groups + the query's own once a query sees more than 2048 keys); layer 0 dense, the others the latent MoE (sigmoid router on the hidden state, experts on a 1536 projection, swiglu clamp 10, shared expert); Engram on layers 1/11/21/31 (2/3-gram hashes into 16 rows of 96 per token) |
| ggml `F8_E4M3` (type 43) | plain OCP E4M3 bytes, the type of Maya's Engram table (upstream ggml's numbering; unsloth's DeepSeek V4.1 "type 43" is a different, MXFP8 layout, which `tools/requant_table.py` reads) |
| `llama-quantize` rules | experts follow the file type; `per_layer_token_embd` (the Engram table, 63.4G values) IQ4_NL by default, `--tensor-type per_layer_token_embd=<type>` picks another; every other matrix at least Q8_0; hyper-connection mixers, indexer, Engram biases / scales stay F32 |
| converter (`conversion/hy_v4_flash.py`) | `HYV4FlashForCausalLM` -> `hy_v4_flash` GGUF: kv_b split into k_b / v_b, gate_up split, the Engram key / value projections fused, norm_key x norm_query folded into the gate, the four Engram tables streamed into one table, the hash primes checked against the config |
| `tools/teacher cmp` | KL over the whole vocabulary against a full log-softmax dump (Abyss's `STRATA_GLM_SCORE_DUMP`) |

Limits: the indexer's group selection (positions 2048 and later) assumes one sequence whose KV cells are its positions
0, 1, 2, ... in order (perplexity / imatrix chunks, a fresh generation); a batch with several sequences past 2048
stops with an error. Exact score ties go to the earlier group.

## Checked

See the worklog numbers in the [CHANGELOG](../../CHANGELOG.md) entry; in short:

- **Against Abyss on the same token ids** (Maya-XS and nace Q3_K_M, mix459 and a 2,048-token Japanese text): the
  per-position KL over the whole vocabulary and the same-top-token share:

  | File, text | Positions | KL mean (median / 99th pct) | Same top token | nll llama.cpp / Abyss |
  | --- | --- | --- | --- | --- |
  | Maya-XS IQ2_XXS, mix459 | 457 | 0.118 (0.0006 / 1.95) | 93.4% | 1.639 / 1.620 |
  | Maya-XS IQ2_XXS, ja2k | 2046 | 0.106 (0.022 / 1.68) | 88.4% | 4.116 / 4.125 |
  | nace Q3_K_M, mix459 | 457 | 0.083 (0.0002 / 1.72) | 93.7% | 1.655 / 1.734 |
  | nace Q3_K_M, ja2k | 2046 | 0.062 (0.011 / 0.87) | 91.8% | 3.832 / 3.805 |

  Not zero: the two engines do the same math in different orders and precisions (Abyss token by token with its own
  kernels, llama.cpp in 512-token batches with F16 KV), and at 2 bits the router flips experts on near ties. Neither
  side's nll is consistently lower. On the first 192 tokens of mix459 with Maya-XS, both engines against the numpy
  reference (`ref/hy4.py`, float64, the same file): llama.cpp KL 0.021 / 98.4% same top, Abyss 0.017 / 97.4%,
  llama.cpp against Abyss 0.029: each engine is as close to the reference as to the other. Positions 0-136 (the Japanese part of mix459) agree closer (mean KL 0.011 / 0.008)
  than the English and code after it (0.16 / 0.11).
- **Against a numpy reference** (Abyss's `ref/hy4.py`) on a tiny random model with the checkpoint's layout (33
  layers, 4 Engram layers, the indexer selecting from 16 keys): KL 0.000002 at 16 tokens; at 60 tokens median
  0.000001, the largest 0.003 where two experts' router scores differ by 7e-5 (summation order picks the other one).
- **Converter**: a synthetic checkpoint with the real names and layouts converts and loads; every fused / split
  tensor equals the expected transform bit for bit; the transforms were checked on the real checkpoint by range
  reads of layer 1 and 11 against nace's GGUF (norms / F32 tensors identical, Q8_0 tensors correlation 0.99999,
  expert gate / up / down 0.988 against Q3_K: the other orders 0.00). **Not run on the full 440 GB checkpoint.**

## Steps (`run.sh`)

| Step | Disk | Time |
| --- | --- | --- |
| download | 440.5 GB (index.json total_size) | 440.5 GB at your line speed (1 Gbps: about 1 h) |
| convert (`--outtype bf16`) | ~438 GB (219B parameters x 2 bytes; 0.4 GB of them F32) | not measured: reads 440 GB, writes 438 GB |
| calib | ~2 MB | minutes (downloads eaddario/imatrix-calibration) |
| imatrix (GPU, experts from the host) | ~1 GB | not measured on BF16; see the CHANGELOG for the Maya-XS rate |
| quantize (experts IQ2_XXS, table IQ4_NL, rest Q8_0) | ~82 GB: experts 148.6G x 2.06 bit = 38.3 GB, table 63.4G x 4.5 bit = 35.6 GB, the rest 6.9G x 8.5 bit = 7.3 GB | not measured |

The Engram table gets no imatrix (it is read by row, not multiplied), which is why it gets a type that does not need
one. The `--tensor-type` upgrades in `run.sh` for Japanese are a starting point, not a measured choice.
