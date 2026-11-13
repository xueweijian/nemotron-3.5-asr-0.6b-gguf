# Usage

Everything here runs **CPU only**. There is no GPU path for `TQ1_0` — see the limitations section at the end.

## 0. What you need

| | |
|---|---|
| C++ toolchain | C++17, `cmake` ≥ 3.20, `ninja` |
| Python | 3.10+ (only for the evaluation driver; no third-party packages) |
| Disk | ~1.5 GB for the runtime source + build, 324 MB for the model |
| RAM | 537 MB peak measured; 1 GB is comfortable |

## 1. Get the model

```bash
huggingface-cli download xiaoxiaojian/nemotron-3.5-asr-0.6b-325mb-gguf \
    nemotron-3.5-asr-0.6b-325mb.gguf --local-dir .

# always verify before use:
sha256sum -c SHA256SUMS
# expect: nemotron-3.5-asr-0.6b-325mb.gguf: OK
```

Do not skip the checksum. The export pipeline is not byte-deterministic — a file of exactly the right size can still be a different file.

## 2. Build the runtime

`transcribe.cpp` needs one patch before it will accept this file. The patch is 29 lines with 10 changed lines (+7/−3) and does two things:

1. adds `GGML_TYPE_TQ1_0` to `TRANSCRIBE_QUANT_LINEAR_TYPES` — without it the loader rejects the type for linear tensors and you get a `nullptr` weight;
2. routes 48 one-by-one convolutions from `GET_CONV` to `GET_LIN` — `GET_CONV` has its own whitelist that does not include `TQ1_0`.

```bash
git clone https://github.com/handy-computer/transcribe.cpp.git
cd transcribe.cpp
patch -p1 -i ../patches/r2/0001-tq1-linear-and-2d-pointwise.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTRANSCRIBE_CUDA=OFF -DTRANSCRIBE_METAL=OFF -DTRANSCRIBE_VULKAN=OFF
cmake --build build -j"$(nproc)" --target transcribe-cli
```

Or, from this repository, `bash scripts/r2_build.sh` — it copies the source to a scratch dir, applies every patch in `patches/r2/` idempotently, builds, and self-checks the binary (including that a missing model produces the *expected* error rather than a crash). Override `SRC_SRC` / `BUILD_SRC` if you cloned elsewhere.

## 3. Transcribe

The path below assumes the manual build from §2, run from inside `transcribe.cpp/`. If you used `scripts/r2_build.sh`, the binary is where the script printed it (`$TMPDIR/nemotron-build/tc-src/build/bin/transcribe-cli` by default) — read it from `build/build.json` if you lost the line.

```bash
./build/bin/transcribe-cli \
    -m nemotron-3.5-asr-0.6b-325mb.gguf \
    -l zh-CN \
    -q audio.wav
```

### The two flags that matter

**`--n-ctx 1024` — only when you use `--batch`.** Single-file inference ignores it: transcribe.cpp documents `n_ctx` as a no-op for this architecture, and there is no KV ceiling to lower. In batch mode the default equals the model maximum, decoder state leaks across items, and one bad sample drags the whole run to the ceiling: a 7.1 s clip took 20+ minutes. Measured: the same binary with the default lands in single-digit RTFx, with `1024` it reaches 14.7× (8 items in 11.6 s). That is a workaround for a batch-mode bug, not a property of this file.

**`-l` must be a locale the runtime's prompt dictionary knows.** Two FLEURS tags need remapping — `es-419 → es`, `ar-EG → ar`. The other 15 locales in this release work as-is. If you pass an unknown tag the CLI does not fail loudly; it falls back and the transcript quality silently changes. That is the single most common way to get wrong numbers.

### Thread count

Set threads to the **real CPU quota**, never to `nproc`. On a container with a 4-core quota, `nproc` may report 32; using 32 makes it ~17× **slower** (28% of cycles lost to throttling). Measured on one 11.6 s clip: 2 threads 1.72 s, **4 threads 1.13 s**, 8 threads 1.49 s, 32 threads 19.67 s.

## 4. Reproduce the accuracy table

The table in `README.md` is FLEURS test, 17 locales, 11,947 utterances, weighted by utterance count.

```bash
# 1. get FLEURS test audio for the 17 locales and write a manifest:
#    one JSON object per line: {"locale": "zh-CN", "audio": "/abs/path.wav", "text": "reference"}
#    (any extraction works; the driver only needs those three fields)

# 2. run
python3 scripts/52_r2_formal.py \
    --gguf nemotron-3.5-asr-0.6b-325mb.gguf \
    --label b325 \
    --manifest /path/to/manifest.jsonl \
    --jobs 8 --threads 4 \
    --json-out formal_b325.json
```

Output is per-locale `{n, primary, seconds, empty, fp16, delta_points}` plus a weighted total. `primary` is CER for `zh-CN` / `ja-JP` / `ko-KR` and WER elsewhere, chosen by `pzq.eval.should_use_cer`, which keys off the locale prefix. `--threads` is passed straight to the CLI.

For a 5-minute smoke test instead of the full 40 hours of audio, `ci/assets/eval340_ids.json` fixes a 340-utterance subset (17 locales × 20). It was calibrated against the full set: 14.18% on the subset vs 14.21% on the full set, a 0.03-point gap. Good enough to detect a broken build, not good enough to publish a number.

The published per-locale results are in [`eval/formal_b323.json`](eval/formal_b323.json) — `{locale, n, primary, seconds, empty, fp16, delta_points}` for all 17 locales, the same numbers as the table in `README.md`. Compare your run against it field by field; a matching `primary` per locale means the build and the audio are the same.

## 5. Limitations you should read before deploying

- **No GPU.** `TQ1_0` has no CUDA kernel; Metal rejects the type outright; Vulkan has one but is untested here. If you need GPU throughput, use a Q4/Q5 build of the same base model.
- **17 of 40 locales tested.** The other 23 are unmeasured, not "probably fine".
- **5 empty transcripts out of 11,947** (0.04%), spread over `en-US`, `de-DE`, `hi-IN` and `ja-JP`. Rare, but a production pipeline needs a fallback.
- **Worst locale is `nl-NL` at +3.12 points** over fp16. If Dutch is your main language, the trade may not be worth it.
- **The quantization is lossy by design.** 14.21% vs 12.98% weighted. If you need fp16 accuracy, this is not that.
