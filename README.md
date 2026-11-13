# nemotron-3.5-asr-0.6b-325mb

A **323.59 MB** mixed-precision GGUF of [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — a 600M cache-aware streaming RNNT model — that keeps **14.21% weighted WER/CER across 17 FLEURS locales**, against **12.98%** for the fp16 base: **+1.23 points**.

**Download:** [`nemotron-3.5-asr-0.6b-325mb.gguf`](https://huggingface.co/xiaoxiaojian/nemotron-3.5-asr-0.6b-325mb-gguf) · 323,593,760 bytes (323.59 MB decimal / 308.6 MiB) · `sha256 5d8c5e0afce2b9ad9a62d5927078c4a6c902f6de4e2bf0bd83b75b6206143776`

Converted from `nvidia/nemotron-3.5-asr-streaming-0.6b` at revision [`ea30d66`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/ea30d66debe3740a08b573244286791d423d6b3e), targeting `transcribe.cpp` 0.4.2.

> **Unofficial community quantization. Not affiliated with or endorsed by NVIDIA.** `nemotron` is a NVIDIA trademark, used here descriptively. The weights inherit **OpenMDW-1.1** — the full text is in [`LICENSE-OpenMDW-1.1`](LICENSE-OpenMDW-1.1), and [`NOTICE`](NOTICE) records exactly what was changed. The scripts and patch in this repository are MIT ([`LICENSE-CODE`](LICENSE-CODE)).

> **Verify before use.** The export pipeline is not byte-deterministic: two runs of the identical configuration produced byte-different files of the same size. Size alone does not identify the artifact — only the sha256 above does. `sha256sum -c SHA256SUMS` takes a second.

> **The 325 MB budget is self-imposed.** It is not an external standard, not a round number from any paper, and not a property of the base model. Inside this project the line moved 220 → 295 → 320 → 325 MB as the accuracy cost of each step became measurable. Treat "325 MB" as a design target that this artifact meets, not as a claim that 325 MB is a meaningful threshold for anyone else.

---

## What it is

| | |
|---|---|
| Base model | `nvidia/nemotron-3.5-asr-streaming-0.6b` (600M params, 40 language-locales, RNNT) |
| Container | GGUF, 653 tensors |
| Types present | `TQ1_0`, `Q4_0`, `Q8_0`, `F32` |
| Size | 323.59 MB decimal (308.6 MiB) — **0.5072 bytes per weight** over 637,991,968 elements |
| Languages evaluated | 17 (see table) |
| Runtime | [`transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp) 0.4.2 + a 10-line patch, CPU only |
| Licence | weights OpenMDW-1.1 · code MIT |

The quantization is **not uniform**. Tensors were assigned per-module widths from a sensitivity analysis, then exported so that ggml's existing types carry the plan directly — no custom kernels, no runtime dequantization. All four types are stock ggml types.

### What is actually inside the file

GGUF metadata lists every tensor and its type — this is a property of the container, not a disclosure. Anyone who loads the file sees it:

| type | tensors | elements | share |
|---|---:|---:|---:|
| `TQ1_0` (≈1.69 bpw) | 106 | 164,626,432 | 25.8% |
| `Q4_0` (4.5 bpw) | 155 | 433,061,888 | 67.9% |
| `Q8_0` (8.5 bpw) | 14 | 39,575,552 | 6.2% |
| `F32` | 378 | 728,096 | 0.1% |
| **total** | **653** | **637,991,968** | 100% |

Two things worth noting, because they are the difference between this file and a one-size-fits-all build:

- The 106 ternary tensors are **not** scattered at random: 89 are attention projections and 17 are `ff1.linear2` (the 4096→1024 down-projection). The mirror-image `ff1.linear1` (1024→4096) is kept at `Q4_0` in every layer.
- The 378 `F32` tensors are norm weights, biases and small embeddings — 0.1% of elements. They cost almost nothing and are not worth quantizing.

## Accuracy — FLEURS test, 11,947 utterances, 39.54 h

`WER` for 14 locales, `CER` for `zh-CN` / `ja-JP` / `ko-KR`. Weighted by utterance count. `Δ` is quantized minus fp16, in points (lower is better; positive means the quantized model is worse).

| locale | n | fp16 | 325mb | Δ (pt) | metric |
|---|---:|---:|---:|---:|---|
| `zh-CN` | 945 | 20.01 | 20.22 | **+0.21** | CER |
| `pt-BR` | 919 | 8.91 | 9.40 | +0.49 | WER |
| `es-419` | 908 | 6.54 | 6.91 | +0.37 | WER |
| `it-IT` | 865 | 6.41 | 6.86 | +0.45 | WER |
| `de-DE` | 862 | 11.05 | 12.78 | +1.73 | WER |
| `vi-VN` | 857 | 15.12 | 16.25 | +1.13 | WER |
| `ru-RU` | 775 | 13.46 | 14.72 | +1.26 | WER |
| `pl-PL` | 758 | 20.43 | 22.77 | +2.34 | WER |
| `uk-UA` | 750 | 16.39 | 19.15 | +2.76 | WER |
| `tr-TR` | 743 | 16.13 | 17.73 | +1.60 | WER |
| `fr-FR` | 676 | 11.62 | 13.26 | +1.64 | WER |
| `ja-JP` | 650 | 14.19 | 15.77 | +1.58 | CER |
| `en-US` | 647 | 10.84 | 11.15 | +0.31 | WER |
| `ar-EG` | 428 | 16.35 | 17.79 | +1.44 | WER |
| `hi-IN` | 418 | 9.22 | 10.56 | +1.34 | WER |
| `ko-KR` | 382 | 9.12 | 9.74 | +0.62 | CER |
| `nl-NL` | 364 | 14.63 | 17.75 | **+3.12** | WER |
| **weighted** | **11,947** | **12.98** | **14.21** | **+1.23** | — |

Reading the table:

- **Best locale is Chinese**: `zh-CN` CER moves 20.01 → 20.22, **+0.21 pt**. The 4096-wide `feed_forward1.linear1`, the most sensitive tensor family in this architecture, is kept at a wider width precisely for this.
- **Worst locale is Dutch**: `nl-NL` +3.12 pt. No locale exceeds +3.12, i.e. nothing collapses.
- **5 of 11,947 transcripts are empty** (0.04%): 1 each in `en-US`, `de-DE`, `hi-IN`, and 2 in `ja-JP`. No repetition loops and no `<lang-XX>` tag leakage were observed anywhere in the run.
- **The aggregate is not a cherry-pick**: the weighted number is computed over the full test split of all 17 locales, not over a subset that flatters the result.

### Why 17 locales and not 40

The base model advertises 40 language-locales. FLEURS — the only multilingual set with per-locale test splits large enough to be meaningful — covers 17 of them. The remaining 23 are **untested here**, not "supported and fine". If you need those, evaluate before you ship.

## Speed and memory

Measured on the patched runtime, CPU only, `--n-ctx 1024`, batch mode:

| | |
|---|---|
| Hardware | AMD EPYC 9K65, **4 threads** |
| Throughput | **RTFx 14.7×** (20 clips / 188.6 s audio in 12.83 s wall) |
| Peak RSS | **537 MB** |
| Disk | 324 MB |

Two counter-intuitive facts worth knowing before you benchmark it yourself:

- **More threads is worse, badly.** The same 11.6 s clip: 2 threads 1.72 s, **4 threads 1.13 s**, 8 threads 1.49 s, **32 threads 19.67 s — 17× slower**, with 28% of cycles lost to throttling in `cpu.stat`. Set threads to the real CPU quota, not to `nproc`.
- **`--n-ctx` matters only in `--batch` mode.** Single-file inference ignores it (the model has no KV ceiling to lower — transcribe.cpp documents `n_ctx` as a no-op for this architecture). In batch mode the default equals the model maximum, the decoder state leaks across items, and one bad sample drags the whole run to the ceiling: a 7.1 s clip took 20+ minutes. `--n-ctx 1024` fixed it (8 items in 11.6 s). That is a workaround for a batch-mode bug, not a property of this file.

### No GPU

`TQ1_0` has **no CUDA kernel** in ggml, and Metal **rejects the type**. There is no GPU path today. CPU and Vulkan are the options. If you need GPU, this is the wrong artifact — pick a Q4/Q5 build of the same base model instead.

## How to run

```bash
# 1. Build transcribe.cpp with the patch (the patch is required — TQ1_0 is not
#    accepted by the loader for linear tensors without it)
git clone https://github.com/handy-computer/transcribe.cpp.git
bash scripts/r2_build.sh

# 2. Transcribe
./build/bin/transcribe-cli \
    -m nemotron-3.5-asr-0.6b-325mb.gguf \
    -l zh-CN -q \
    audio.wav

# 3. Verify the file you downloaded
sha256sum -c SHA256SUMS
```

The patch does two things, both minimal: it adds `GGML_TYPE_TQ1_0` to the linear-tensor whitelist, and it routes 48 one-by-one convolutions through the linear getter instead of the conv getter. See `USAGE.md` for the full walkthrough and for reproducing the accuracy table.

## What this is not

- **Not a 40-locale model.** 17 evaluated, 23 untested.
- **Not faster than the base model on GPU.** There is no GPU path.
- **Not bit-exact reproducible on export.** Two runs of the identical configuration produced byte-different files of identical size. Only the published sha256 denotes the evaluated artifact — if you re-export, you get "a" file, not "the" file.
- **Not a claim of beating every other quantization.** Comparisons with other community builds are only meaningful when measured on the same audio, the same runtime, and the same metric. Numbers published elsewhere often use English-only WER or a different normalization; they are not directly comparable to the 17-locale weighted figure above, and no such comparison is asserted here.
- **Not affiliated with NVIDIA.** See the notice at the top.

## Citation

```bibtex
@misc{nemotron-asr-06b-325mb,
  title  = {nemotron-3.5-asr-0.6b-325mb: 17-locale mixed-precision GGUF},
  year   = {2026},
  note   = {Quantized from nvidia/nemotron-3.5-asr-streaming-0.6b.
            Weights: OpenMDW-1.1. Code: MIT. Unofficial, not affiliated with NVIDIA.},
}
```

Released 2026-11-13. If you find a number here that you cannot reproduce, that is a bug — please open an issue with the audio, the command, and the runtime commit.
