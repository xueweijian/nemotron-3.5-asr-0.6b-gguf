#!/usr/bin/env python3
"""在 FLEURS test 全量 11,947 条上评任意 GGUF（transcribe.cpp 路径）。

为什么另写一个而不是改 ``51_r2_wer8.py``：那个是「8 条抽检、判坏没坏」，这个是
「17 locale × 11,947 条、判几点」，两者的输出契约（per-locale + 加权总）和失败
处理（单条失败不能整轮停）都不同，混成一个函数会让两边都难读。

口径与正式集一致（``docs/research/STEP2_FORMAL_EVAL.md`` §3）：per-locale、
``pzq.eval.score`` 自写归一。中文必须用 kebab（``zh-CN``）—— ``should_use_cer``
按 ``-`` 切分，写 ``cmn_hans_cn`` 会让中文被当 WER 算（债 #29）。

**线程数必须按 cgroup 配额填，不是 nproc。** 本沙盒 `nproc` 报 32，但
`/sys/fs/cgroup/cpu.max` 是 `400000 100000`（**4 核**）。同一条 11.6 s 音频实测：
2 线程 1.72 s / **4 线程 1.13 s** / 8 线程 1.49 s / **32 线程 19.67 s**（慢 17×，
`cpu.stat` 里 28% 的周期被 throttle）。所以默认 4。

两个 locale 的 language 标签要重映射：``es-419→es``、``ar-EG→ar``，因为
transcribe.cpp 的 prompt dictionary 只认这两个短标签（实测；其余 15 个 kebab
原样可用）。

用法：

    python3 scripts/52_r2_formal.py \\
        --gguf /tmp/r2fix/model-b315.gguf --label b315 \\
        --manifest /tmp/fleurs_full/manifest.jsonl \\
        --jobs 8 --threads 4 \\
        --json-out docs/research/evidence/r2/formal_b315.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pzq.eval import score  # noqa: E402  （必须在 sys.path 之后）

DEFAULT_CLI = Path(os.environ.get("TRANSCRIBE_CLI", "/tmp/tc-build/build/bin/transcribe-cli"))
MB = 1e6

# transcribe.cpp prompt dictionary 不认的标签（实测）
LANG_ALIAS = {"es-419": "es", "ar-EG": "ar"}


def load_manifest(path: Path) -> dict[str, list[dict]]:
    """-> {locale(kebab): [{audio(绝对路径), text}, ...]}，保持 manifest 原顺序。

    抽取 kernel 写的 ``audio`` 是**相对 out 目录**的（``audio/xxx.wav``），
    直接喂 CLI 会全部打不开、静默跳过并返回非 0 —— 这里统一补成绝对路径。
    """
    base = path.parent
    by: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            a = Path(d["audio"])
            d["audio"] = str(a if a.is_absolute() else (base / a))
            by.setdefault(d["locale"], []).append(d)
    return by


def run_locale(cli: Path, gguf: Path, locale: str, items: list[dict],
               threads: int, work: Path, n_ctx: int,
               locale_timeout: int = 0) -> dict:
    """单 locale：写列表 → CLI --batch --batch-jsonl → 解析。

    ``--n-ctx`` 不是可选项，是必需品：默认是模型上限，而 ``--batch`` 模式下
    decoder 状态会跨条残留，个别样本被拖进坏状态后一路跑到上限（实测
    ``ar_eg_00019.wav`` 7.1 s 音频跑了 20+ 分钟；单独跑同一文件只要 186 ms）。
    加 1024 后同一批 8 条 11.6 s 跑完。1024 token 对 FLEURS 短句（最长 23.8 s）
    绰绰有余。
    """
    lst = work / f"{locale}.list"
    out = work / f"{locale}.jsonl"
    lst.write_text("\n".join(it["audio"] for it in items) + "\n", encoding="utf-8")

    # 断点续跑：已完成（行数够）就跳过
    if out.is_file():
        done = sum(1 for _ in out.open(encoding="utf-8", errors="replace"))
        if done >= len(items) + 1:                     # +1 = batch_header
            pass
        else:
            out.unlink(missing_ok=True)

    if not out.is_file():
        lang = LANG_ALIAS.get(locale, locale)
        cmd = [str(cli), "-m", str(gguf), "-l", lang, "-q", "--n-ctx", str(n_ctx),
               "--batch", str(lst), "--batch-jsonl", "--threads", str(threads)]
        t0 = time.time()
        # 默认公式按「单条 1.3s ×3 余量」估；b315 在部分 locale（tr/vi）会
        # RNNT 发散到 ~4s+/条，743 条就把 2972s 的公式上限顶爆（p6 实测）。
        # Kaggle kernel 12h 上限，由调用方显式给 7200。
        to = locale_timeout if locale_timeout > 0 else max(900, len(items) * 4)
        try:
            with out.open("w", encoding="utf-8") as fh:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE,
                                   text=True, timeout=to)
        except subprocess.TimeoutExpired:
            return {"locale": locale, "ok": False, "error": f"超时 {to}s", "n": len(items)}
        dt = time.time() - t0
        if p.returncode != 0:
            return {"locale": locale, "ok": False,
                    "error": (p.stderr or "")[-400:], "n": len(items)}
    else:
        dt = 0.0

    hyps: dict[str, str] = {}
    for line in out.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "file" in d and "text" in d:
            hyps[d["file"]] = d["text"]

    refs, outs = [], []
    for it in items:
        if it["audio"] not in hyps:
            continue
        refs.append(it["text"])
        outs.append(hyps[it["audio"]])
    if not refs:
        return {"locale": locale, "ok": False, "error": "一条都没解析到", "n": len(items)}

    sc = score(refs, outs, locale=locale)
    return {
        "locale": locale, "ok": True, "lang_used": lang,
        "n": len(refs), "n_expected": len(items),
        "wer": sc["wer"], "cer": sc["cer"], "primary": sc["primary"],
        "empty": sum(1 for h in outs if not h.strip()),
        "seconds": round(dt, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GGUF 在 FLEURS test 全量上的 per-locale 评测")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--manifest", default="/tmp/fleurs_full/manifest.jsonl")
    ap.add_argument("--cli", default=str(DEFAULT_CLI))
    ap.add_argument("--jobs", type=int, default=8, help="并发进程数（按 locale）")
    ap.add_argument("--threads", type=int, default=4,
                    help="每个进程的 CPU 线程；**别按 nproc 填**，见下方说明")
    ap.add_argument("--locale-timeout", type=int, default=0,
                    help="单 locale 超时秒数；0=自动 max(900, n*4)（发散 locale 会被顶爆）")
    ap.add_argument("--limit-per-locale", type=int, default=0, help="调试用；0=全量")
    ap.add_argument("--n-ctx", type=int, default=1024,
                    help="decoder KV 上限；必须设，否则发散样本会跑到模型上限")
    ap.add_argument("--work", default="/tmp/r2formal")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    gguf = Path(args.gguf)
    if not gguf.is_file():
        raise SystemExit(f"模型不存在：{gguf}")
    cli = Path(args.cli)
    if not cli.is_file():
        raise SystemExit(f"CLI 不存在：{cli}")

    by = load_manifest(Path(args.manifest))
    if args.limit_per_locale:
        by = {k: v[: args.limit_per_locale] for k, v in by.items()}
    total = sum(len(v) for v in by.values())
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    print(f"[正式集] {gguf.name}  {gguf.stat().st_size / MB:.2f} MB  "
          f"{len(by)} locale / {total} 条  jobs={args.jobs} threads={args.threads}", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_locale, cli, gguf, loc, items, args.threads, work,
                          args.n_ctx, args.locale_timeout): loc
                for loc, items in sorted(by.items())}
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            if r.get("ok"):
                print(f"  {r['locale']:8s} n={r['n']:4d} "
                      f"WER={r['wer']*100:6.2f}%  CER={r['cer']*100:6.2f}%  "
                      f"primary={r['primary']*100:6.2f}%  空串={r['empty']:3d}  {r['seconds']:6.1f}s",
                      flush=True)
            else:
                print(f"  {r['locale']:8s} 失败：{r.get('error','')}", flush=True)

    rows.sort(key=lambda r: r["locale"])
    good = [r for r in rows if r.get("ok")]
    n_tot = sum(r["n"] for r in good)
    # 加权总：primary 按条数加权（与 33/34 的分片 aggregate 同口径）
    overall = sum(r["primary"] * r["n"] for r in good) / n_tot if n_tot else float("nan")

    print(f"\n[汇总] {len(good)}/{len(rows)} locale 成功  n={n_tot}/{total}  "
          f"加权 primary={overall*100:.2f}%  墙钟 {(time.time()-t0)/60:.1f} min")
    print("\n| locale | n | WER% | CER% | primary% | 空串 |")
    print("|---|---|---|---|---|---|")
    for r in good:
        print(f"| {r['locale']} | {r['n']} | {r['wer']*100:.2f} | {r['cer']*100:.2f} "
              f"| {r['primary']*100:.2f} | {r['empty']} |")

    if args.json_out:
        payload = {
            "label": args.label, "gguf": str(gguf),
            "gguf_mb": round(gguf.stat().st_size / MB, 2),
            "manifest": args.manifest, "n_total": n_tot, "n_expected": total,
            "overall_primary": overall,
            "jobs": args.jobs, "threads": args.threads,
            "locales": rows,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"\n→ {args.json_out}")
    return 0 if len(good) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
