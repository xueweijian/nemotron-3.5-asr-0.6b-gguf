# nemotron-3.5-asr-0.6b-325mb（中文文档）

[English README](README.md) · 本文是 [`README.md`](README.md) 的中文版，数字与结论以英文版为准。

[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) 的 **323.59 MB** 混合精度 GGUF。基座是 600M 参数的 cache-aware 流式 RNNT 模型；本件在 **17 个 FLEURS 语种**上的加权 WER/CER 为 **14.21%**，fp16 基座为 **12.98%**，退化 **+1.23 个百分点**。

**下载：** [`nemotron-3.5-asr-0.6b-325mb.gguf`](https://huggingface.co/xiaoxiaojian/nemotron-3.5-asr-0.6b-325mb-gguf) · 323,593,760 字节（323.59 MB 十进制 / 308.6 MiB）· `sha256 5d8c5e0afce2b9ad9a62d5927078c4a6c902f6de4e2bf0bd83b75b6206143776`

**323.59 MB——比 fp16 源文件（1277.03 MB）小 3.95 倍，精度代价 +1.23 个百分点。** 折算下来每个权重 0.5072 字节，而 fp16 是 2.0 字节。这张卡上的每个数字都是在本机实测的，用仓库里的文件都能复现：体积是字节数，精度是下面逐语种的表，速度是固定 20 条批次上的墙钟时间。

由 `nvidia/nemotron-3.5-asr-streaming-0.6b` revision [`ea30d66`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/ea30d66debe3740a08b573244286791d423d6b3e) 转换而来，目标运行时为 `transcribe.cpp` 0.4.2。

> **非官方社区量化，与 NVIDIA 无隶属关系，也未获其背书。** `nemotron` 是 NVIDIA 的商标，此处仅作描述性使用。权重继承 **OpenMDW-1.1**，全文见 [`LICENSE-OpenMDW-1.1`](LICENSE-OpenMDW-1.1)，[`NOTICE`](NOTICE) 逐项记录了改动内容。本仓库内的脚本与补丁为 MIT 许可（[`LICENSE-CODE`](LICENSE-CODE)）。

> **用前先核对。** 导出流程不具备字节确定性：同一份配置跑两次，得到字节不同、体积相同的两个文件。**体积不能用来识别产物，只有上面那串 sha256 可以。** `sha256sum -c SHA256SUMS` 一秒即可完成。

> **325 MB 这条线是我自己定的。** 它既不是外部标准，也不是哪篇论文里的整数，更不是基座的属性。在项目内部，这条线随着每一步的精度代价被量出来，先后走过 220 → 295 → 320 → 325 MB。请把「325 MB」理解成**这件产物满足的一个设计目标**，而不是在宣称 325 MB 对其他人有任何门槛意义。

---

## 这是什么

| | |
|---|---|
| 基座 | `nvidia/nemotron-3.5-asr-streaming-0.6b`（600M 参数，40 个语言-地区，RNNT） |
| 容器 | GGUF，653 个张量 |
| 含有的类型 | `TQ1_0`、`Q4_0`、`Q8_0`、`F32` |
| 体积 | 323.59 MB 十进制（308.6 MiB）—— 637,991,968 个元素，**平均 0.5072 字节/权重** |
| 已评测语种 | 17（见下表） |
| 运行时 | [`transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp) 0.4.2 + 29 行补丁（改动 10 行），仅 CPU |
| 许可 | 权重 OpenMDW-1.1 · 代码 MIT |

量化**不是均匀的**。先用敏感度分析给每个模块分配位宽，再按这份方案导出，让 ggml 现有的类型直接承载分配结果——没有自定义 kernel，也没有运行时反量化。四种都是 ggml 原生类型。

### 文件里到底是什么

GGUF 元数据会列出每个张量及其类型，这是容器本身的属性，不构成「披露」——任何人加载这个文件都能看到：

| 类型 | 张量数 | 元素数 | 占比 |
|---|---:|---:|---:|
| `TQ1_0`（≈1.69 bpw） | 106 | 164,626,432 | 25.8% |
| `Q4_0`（4.5 bpw） | 155 | 433,061,888 | 67.9% |
| `Q8_0`（8.5 bpw） | 14 | 39,575,552 | 6.2% |
| `F32` | 378 | 728,096 | 0.1% |
| **合计** | **653** | **637,991,968** | 100% |

有两点值得单独说明，它们正是本件与「一刀切」构建方式的差别所在：

- 106 个 ternary 张量**不是**随机撒的：其中 89 个是注意力投影，17 个是 `ff1.linear2`（4096→1024 的下投影）。与之镜像的 `ff1.linear1`（1024→4096）在每一层都保留 `Q4_0`。
- 378 个 `F32` 张量是 norm 权重、偏置和小 embedding，合计只占 0.1% 的元素。它们几乎不占体积，量化它们没有收益。

## 精度 — FLEURS test，11,947 条，39.54 小时

14 个语种用 `WER`，`zh-CN` / `ja-JP` / `ko-KR` 用 `CER`。按发音条数加权。`Δ` 为量化减 fp16，单位是百分点（越低越好；正数表示量化后更差）。

| 语种 | 条数 | fp16 | 325mb | Δ（点） | 指标 |
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
| **加权** | **11,947** | **12.98** | **14.21** | **+1.23** | — |

怎么读这张表：

- **表现最好的是中文**：`zh-CN` CER 从 20.01 到 20.22，**+0.21 点**。4096 宽的 `feed_forward1.linear1` 是这个架构里最敏感的张量族，正是为了它才保留了更宽的位宽。
- **最差的是荷兰语**：`nl-NL` +3.12 点。没有任何语种超过 +3.12，也就是说没有哪个语种崩掉。
- **11,947 条里有 5 条空输出**（0.04%）：`en-US` 1 条、`de-DE` 1 条、`hi-IN` 1 条、`ja-JP` 2 条。整轮运行未观察到重复循环，也未观察到 `<lang-XX>` 标签泄漏。
- **加权值不是挑出来的**：这个数字在全部 17 个语种的完整 test split 上计算，不是挑了好看的子集。

### 为什么测了 17 个，而不是 40 个

基座宣称覆盖 **40 个语言-地区**：19 个 transcription-ready、13 个 broad-coverage、8 个 adaptation-ready（tokenizer 认识，但需微调才能稳定转写）。**本件保留了全部 40 个。** 量化只改变数值精度——不改架构、不改 tokenizer、不改语言提示核。基座能转写的，这个文件也能转写。

收窄的是**实测范围**，不是能力。实际评测了 17 个：

- 19 个 transcription-ready 里的 15 个；
- 2 个 broad-coverage：`pl-PL`、`zh-CN`。

FLEURS 覆盖 40 个里的 39 个（只缺 `nn-NO`），所以这是**预算决定**——17 个语种已经是 39.54 小时音频、11,947 条。剩下 23 个**在这里没测过**，不等于「支持且没问题」。需要的话，请用自己的音频先评测再上线。

那 23 个为什么仍然能用：**决定「转写哪个语种」的张量没有被压到 TQ1_0。** `pred.embed`（640×13088）、两个 `prompt.mlp` 矩阵、整个 RNN-T joint（`joint.enc` / `joint.pred` / `joint.out`）都是 **Q8_0**；所有归一化层的权重与偏置是 **F32**。1.69 bpw 的三值类型只用在编码器的注意力投影和中间 FFN 层上。

## 速度与内存

在打过补丁的运行时上实测，仅 CPU，`--n-ctx 1024`，`--batch` 模式：

实测环境：打过补丁的运行时，仅 CPU，`--n-ctx 1024`、`--batch`、`--threads 4`，20 条音频共 188.6 秒：

| | |
|---|---|
| 硬件 | **4 vCPU**（cgroup 配额 `400000 100000`），内存上限 8 GiB；实测过两台宿主：Intel Xeon 6983P-C、AMD EPYC 9K65（Zen 5） |
| 吞吐 | Xeon 上 **RTFx 10.9×**（188.6 秒耗时 17.34 秒）· EPYC 上 **14.7×**（12.83 秒） |
| 峰值 RSS | **539 MB** → 这个文件能在 **1 GB** 内存的机器上跑（f16 版光权重就要 1.28 GB） |
| 磁盘 | 324 MB |

这里的 `RTFx` 指「整批音频总时长 ÷ 总墙钟时间」，含 I/O，**不能**和单条流式的 RTF 直接比，而且**依赖宿主**：同一个文件、同样线程数、同一批音频，在两颗不同的服务器 CPU 上分别是 10.9× 和 14.7×。规划容量请按低的那个算。

**这是服务器核，不是笔记本。** 两个数字都是「4 线程对 4 vCPU 配额」——线程数与真实配额 1:1。要当心这类机器上的 `nproc`：它报 32（宿主核数），而配额只有 4，脚本若默认取 `nproc` 就是 8 倍超额，掉到 `RTFx 0.36×`。**那个数字量的是配置错误，不是硬件能力。** 目前没有消费级笔记本的实测数据，请自己测过再规划容量。

另外两条反直觉的实测：

- **线程越多越慢，而且慢得很厉害。** 同一条 11.6 秒音频：2 线程 1.72 秒，**4 线程 1.13 秒**，8 线程 1.49 秒，**32 线程 19.67 秒——慢了 17 倍**，`cpu.stat` 里 28% 的周期丢在限流上。线程数要按真实 CPU 配额设，不要按 `nproc` 设。
- **`--n-ctx` 只在 `--batch` 模式下有意义。** 单文件推理会忽略它（这个模型没有可下调的 KV 上限，transcribe.cpp 官方文档也写明 `n_ctx` 对该架构是 no-op）。而 batch 模式下默认值等于模型上限，decoder 状态会在样本之间串味，一条坏样本就能把整轮拖到上限：一条 7.1 秒的音频跑了 20 多分钟。`--n-ctx 1024` 解决了这个问题（8 条 11.6 秒跑完）。**这是对 batch 模式缺陷的规避，不是这个文件的属性。**

### 量化换来的是体积，不是速度

同一台机器（Xeon 6983P-C）、同一运行时、同一批 20 条音频（188.6 秒）、`--threads 4`。A/B 交替跑，各 4 次，第一次作为冷启动（页缓存）丢弃，取其余 3 次中位数：

| 构建 | 体积 | 耗时 | RTFx |
|---|---|---|---|
| fp16 源文件 | 1277.03 MB | 14.98 秒 | 12.59× |
| **本件** | **323.59 MB** | **17.34 秒** | **10.88×** |

小 3.95 倍的文件**慢 15.8%**，不是快。原因（推断，未做消融）：这颗 Xeon 有 AVX-512 + F16C，f16 点积走原生指令；而 `TQ1_0`/`Q4_0` 每次 matmul 前都要把 F32 激活量化成 `Q8_K`。4 倍体积换来的是**磁盘和内存**——权重 324 MB 而不是 1.28 GB，峰值 539 MB 而不是必须装得下整个 fp16 文件——**不是 CPU 时间**。在带宽吃紧的设备上排序可能反过来，但未实测，本仓没有这类数据。

### 没有 GPU

`TQ1_0` 在 ggml 里**没有 CUDA kernel**，Metal **直接拒收该类型**。目前不存在 GPU 路径，可选的是 CPU 和 Vulkan。如果你需要 GPU，这件产物不合适——请换同基座的 Q4/Q5 版本。

## 怎么跑

两个入口，同样三步。全流程不需要 GPU。

| 你从哪来 | 你已经有什么 | 还缺什么 |
|---|---|---|
| Hugging Face（本页） | 模型文件 + `scripts/`、`patches/`、`USAGE.md` | `transcribe.cpp` |
| GitHub | `scripts/`、`patches/`、`USAGE.md` | 模型文件 |

```bash
# 0. 下载模型（约 324 MB）——已有则跳过
huggingface-cli download xiaoxiaojian/nemotron-3.5-asr-0.6b-325mb-gguf \
    nemotron-3.5-asr-0.6b-325mb.gguf --local-dir .
sha256sum -c SHA256SUMS            # 期望：nemotron-3.5-asr-0.6b-325mb.gguf: OK

# 1. 带补丁编译 transcribe.cpp（补丁是必需的——不打补丁，加载器不接受
#    TQ1_0 用于线性张量）
git clone https://github.com/handy-computer/transcribe.cpp.git
bash scripts/r2_build.sh
# 脚本会打印 "cli    : <路径>"，下面这行从构建记录里把路径读回来
CLI=$(python3 -c "import json;print(json.load(open('build/build.json'))['binaries']['transcribe-cli'])")

# 2. 转写
"$CLI" -m nemotron-3.5-asr-0.6b-325mb.gguf -l zh-CN -q audio.wav
```

**二进制不在 `./build/bin/`**——`r2_build.sh` 在临时副本里构建，默认位置是 `$TMPDIR/nemotron-build/tc-src/build/bin/transcribe-cli`。上面那行 `CLI=` 是最可靠的取法。如果你是手工构建（在 `transcribe.cpp/` 内 `cmake -S . -B build`），那 `./build/bin/transcribe-cli` 就是对的。

依赖：C++17 工具链、`cmake` ≥ 3.20、`ninja`。实测峰值内存 539 MB。

补丁只做两件事，都很小：把 `GGML_TYPE_TQ1_0` 加进线性张量的白名单；让 48 个 1×1 卷积走线性 getter 而不是 conv getter。完整流程见 `USAGE.md`，其中也包含复现精度表的方法。

## 这不是什么

- **不是一个 40 语种模型。** 17 个语种在**全量 test split** 上实测——11,947 条、39.54 小时，每条都在同一份 manifest 上有 fp16 基线。另 23 个未测，不是「大概率没问题」。
- **在 GPU 上不比基座快。** 没有 GPU 路径。
- **导出不能字节级复现。** 同一份配置跑两次，得到字节不同、体积相同的文件。只有公布的 sha256 指向被评测的那一件——你自己重新导出，得到的是「某个文件」，不是「这个文件」。
- **不比较跨构建的精度。** 别处公布的 WER 用的是不同音频、不同运行时版本、不同归一化方式，与上面的 17 语种加权值不可直接比较，本仓不作任何此类断言。本卡唯一拿来对照的构建，是导出它的那个 fp16 源文件，同一份 manifest：12.98% → 14.21%。
- **与 NVIDIA 无关。** 见文首声明。

## 许可与署名

- **模型权重** —— [OpenMDW-1.1](LICENSE-OpenMDW-1.1)，继承自
  [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
  revision `ea30d66`。模型材料版权归 NVIDIA Corporation。本件是非官方派生，
  与 NVIDIA 无隶属、无赞助、无背书关系。[`NOTICE`](NOTICE) 逐项记录了改动：
  格式（NeMo → GGUF）、精度（按模块分配 TQ1_0 / Q4_0 / Q8_0 / F32）、未重训、未加数据。
  OpenMDW-1.1 要求再分发时随附协议副本，故保留全文。
- **脚本与补丁** —— MIT（[`LICENSE-CODE`](LICENSE-CODE)）。运行时补丁派生自
  [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
  （MIT，版权归 The transcribe.cpp authors），该声明已保留在 `LICENSE-CODE` 中。
- **评测数据** —— WER/CER 在 [FLEURS](https://huggingface.co/datasets/google/fleurs)
  （CC-BY-4.0）上计算。本仓只分发句子的 **ID** 与参考文本的 SHA-256
  （[`ci/assets/eval340_ids.json`](ci/assets/eval340_ids.json)），不分发音频，也不分发转写文本。
  若你自行从 FLEURS 重建 manifest，CC-BY-4.0 约束的是你那一份副本。
- **输出** —— 不设限。OpenMDW-1.1 原文写明：对使用模型材料生成的任何输出，
  不施加任何使用、修改或分享方面的限制或义务。**你用这个模型转写出来的文字，
  权利归你。**
- **无担保** —— 按「现状」提供。ASR 输出必然有错（评测集加权 WER/CER 14.21%），
  不可用于医疗、法律或安全关键场景。

## 联系方式

- 技术问题：在本仓库开 issue。
- 商业授权 / 合作：[imysql@foxmail.com](mailto:imysql@foxmail.com)
- 欢迎就 ASR 量化与端侧部署方向交流合作。

## 引用

```bibtex
@misc{nemotron-asr-06b-325mb,
  title  = {nemotron-3.5-asr-0.6b-325mb: 17-locale mixed-precision GGUF},
  year   = {2026},
  note   = {Quantized from nvidia/nemotron-3.5-asr-streaming-0.6b.
            Weights: OpenMDW-1.1. Code: MIT. Unofficial, not affiliated with NVIDIA.},
}
```

发布于 2026-11-13。如果你发现本文某个数字复现不出来，那是个 bug——请带上音频、命令和运行时 commit 开 issue。
