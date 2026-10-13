# 图像 SDR 转 HDR 开发项目 · 实施及初验报告

> **报告类型**：实施情况说明 + **初验**（Preliminary Verification）
> **⚠️ 本报告不是最终验收报告。项目尚未完成完整验收测试，不得据此签署验收通过。**
> 报告日期：2026-10-13　｜　测试基准日：2026-10-12　｜　项目阶段：开发完成 / 待完整验收
> 编制依据：`图像SDR转HDR的开发项目立项.md`、`图像SDR转HDR项目立项书.md`、`IMPLEMENTATION_PLAN.md`、`photo_sdr_to_hdr.py`、`bench_photo_sdr_to_hdr.py`、`TEST_COVERAGE.md`
> 编制环境：8 逻辑核 / 7.7GB 内存 / 纯 CPU（无 CUDA）；Python 3.12.9 / Pillow 12.3.0 / numpy 2.5.0
> **当前状态：❌ 未通过验收 —— 已完成实施与初步验证，7 项验收前置条件未满足（见第五章）**
> **完整验收测试分三阶段递进执行：阶段一 CPU（测试准备 + 全量 CPU 实测）→ 阶段二 T4（GPU 通用通路）→ 阶段三 L40（Ada 专属项），详见第七章**

---

## 〇、报告性质声明

| 项 | 说明 |
|---|---|
| 报告目的 | 记录立项以来的**实施完成度**与**初步验证结果**，作为启动完整验收测试的输入 |
| 覆盖范围 | 已验证部分：4 个后端（hdrconv / pyultrahdr / gmnet / sdr2hdr）的端到端转换、格式矩阵、EXIF、并行、干跑、确定性、质量指标 |
| **未覆盖范围** | ① `unet-onnx` / `lumi-*` / `ultrahdr` 共 5 个后端名（4 个实现）零端到端验证；② 学习型后端无质量指标；③ 真实照片回归；④ 千张级压力测试；⑤ 真实 HDR 显示器主观评估 |
| 结论效力 | 仅支持"进入完整验收测试阶段"，**不支持验收通过** |
| 验收路径 | 完整验收按 **CPU → T4 → L40** 三阶段递进，每阶段设独立准出条件 |
| 后续文件 | 三阶段验收测试完成后汇总出具《图像SDR转HDR项目验收报告》 |

---

## 一、项目背景与实施概述

### 1.1 立项背景

立项需求为：在 PhoUtils 工具集内创建统一的 **SDR → HDR 批量转换 CLI 工具**，将立项文档列举的多个开源 HDR 项目统一封装进单一脚本，要求参数完整、区分不同项目模型、支持文件/文件夹批量处理，并遵循"先给出执行方案再进行具体开发设计"的流程约束。

### 1.2 实施完成度总览

| 阶段 | 内容 | 状态 |
|---|---|:---:|
| Phase 1 | 基础框架：CLI 解析、文件发现、路径生成、后端基类与注册表、模型目录、并发框架、日志/报告、干跑 | ✅ 完成 |
| Phase 2 | 核心后端：hdrconv、unet-onnx、pyultrahdr、子进程基类（sdr2hdr/ultrahdr）、统一输出格式层 | ✅ 代码完成 |
| Phase 3 | 高级后端：LumiPic 三变体、GMNet、ultrahdr 封装、EXIF A+C 与色彩空间转换 | ✅ 代码完成 |
| Phase 4 | 打磨与文档：多模型对比、设备与资源预估、综合基准、质量指标模块、README | ✅ 完成 |

**实施层面结论**：立项书约定的 4 个阶段交付物已全部产出，代码路径完整、无占位实现。

### 1.3 交付物清单

```
PhoUtils/
├── photo_sdr_to_hdr.py            # 主 CLI（469 行）
├── bench_photo_sdr_to_hdr.py      # 综合基准（781 行，11 组 A–K）
├── IMPLEMENTATION_PLAN.md         # 执行方案（立项前置交付物）
├── TEST_COVERAGE.md               # 测试覆盖报告
├── 图像SDR转HDR的开发项目立项.md   # 立项需求源
├── 图像SDR转HDR项目立项书.md       # 项目立项书（本项目的正式立项文件）
├── backends/  base.py / pyultrahdr_backend.py / hdrconv_backend.py
│              gmnet_backend.py / lumipic_backend.py / unet_onnx_backend.py
│              subprocess_backend.py（sdr2hdr + ultrahdr）
└── utils/     file_io.py / device.py / parallel.py / logging.py / report.py
               exr_io.py / png16.py / exif.py / itm.py / hdr_metrics.py
```

---

## 二、立项需求追溯

### 2.1 CLI 参数实施核对（立项硬性要求 10 项）

| 立项要求 | 实现（`photo_sdr_to_hdr.py`） | 实施 |
|---|---|:---:|
| `--input` / `-i`（可多个） | L64-67，`nargs="+"`, `required=True` | ✅ |
| `--output` / `-o`（可自动） | L68-71，默认 `_hdr_output/` | ✅ |
| `--models` 区分不同项目 | L74-80，`choices=9 个后端名`，默认 `hdrconv` | ✅ |
| `--workers` / `--threads` | L83-87，三别名 `-w/--workers/--threads`，0=auto | ✅ |
| `--device` auto\|cuda\|cpu（原文 `--devide`） | L90-95，实现为 `--device`，新增 `mps` | ✅ 超集 |
| `--suffix` | L98-101，默认 `_hdr` | ✅ |
| `--recursive` / `-r` | L103-107 | ✅ |
| `--dry-run`（默认行为即预览） | L110-113；L225 `dry_run = not apply` | ✅ |
| `--log`（`-`=stdout） | L204-209，默认 `-` | ✅ |
| `--help` | argparse 内置 + `epilog=__doc__` | ✅ |

**附加参数（超出立项）**：`--apply`、`--format`(7 值)、`--quantization`、`--colorspace`、`--model-dir`、`--keep-exif/--no-keep-exif`、`--sidecar/--no-sidecar`、`--gain-map-method`、`--target-nits`、`--sdr2hdr-mode`、`--ultrahdr-binary`、`-v/--version`。

### 2.2 开源项目落地形态（立项清单 11 项）

| 立项项目 | 落地形态 | 后端标识 | 本环境验证 |
|---|---|---|:---:|
| pyultrahdr | `backends/pyultrahdr_backend.py` 纯 Python 重实现 | `pyultrahdr` | ✅ 25/26 |
| hdr-conversion | `backends/hdrconv_backend.py` 封装 | `hdrconv` | ✅ 21/21 |
| GMNet | `backends/gmnet_backend.py`（免 BasicSR） | `gmnet` | ✅ 19/21 |
| LumiPic | `backends/lumipic_backend.py` 三变体 | `lumi-qwen` / `lumi-klein4b` / `lumi-klein9b` | ⏭ 未验证 |
| SDR-to-HDR-Reconstruction-System | `backends/unet_onnx_backend.py` | `unet-onnx` | ⏭ 未验证 |
| sdr2hdr | `backends/subprocess_backend.py` | `sdr2hdr` | 🔶 6/20（能力边界） |
| libultrahdr | `backends/subprocess_backend.py` | `ultrahdr` | ⏭ 未验证 |
| gainmap-js | 范围外：TypeScript / three.js Web 生态，与 Python CLI 定位不符 | — | — |
| HDR Image Transcoder | 部分落地：`--format avif` 经 hdrconv；AVIF/JXL/HEIF 互转未实现 | — | 🔶 |
| gainforge | 范围外：HDR→SDR 反向色调映射，与项目目标方向相反 | — | — |
| Awesome-Gain-Maps | 范围外：资源汇总列表，非可集成软件 | — | — |

**结果**：7 个项目以独立或扩展后端落地（9 后端名 / 8 实现），4 项判定范围外并给出理由，无静默丢弃。**其中 4 个实现仅有代码、未经端到端验证**。

### 2.3 批量处理能力实施核对

| 能力 | 实现 | 已验证 |
|---|---|:---:|
| 单文件 / 文件夹批量 | `collect_input_files()` | ✅ |
| 多输入路径 | `-i` `nargs="+"` | ✅ |
| 递归子目录 | `-r` | ✅ |
| 输出目录结构保持 | `make_output_path()` 按 `input_root` 相对化 | ✅ |
| 多后端对比（文件名带后端名） | `make_output_path(..., backend_name=)` | ✅ |
| 多进程并行 | `run_parallel_backends()` + `resolve_workers()` | 🔶 功能正确，加速比未验证 |
| 干跑零写入 | `dry_run` 分支提前 return | ✅ |
| 默认干跑 + `--apply` 才写盘 | L221-225 互斥校验 | ✅ |
| 报告落盘 | `ReportGenerator.generate()` | ✅ |

---

## 三、初步验证结果

### 3.1 测试方法学评价

`bench_photo_sdr_to_hdr.py`（11 组 93 项）达到可作为验收测试框架的质量，具备升格为完整验收测试的基础：

| 优点 | 说明 |
|---|---|
| 确定性素材 | 固定种子 42，512×384，渐变+亮斑+σ=8 噪声+细密纹理（为 VMAF 区分度专门设计） |
| 双层指标 | 无参考（rt-VMAF/SSIM/PSNR/DR）+ 全参考（∆ICtCp / PU21 vs 合成 ITM GT） |
| 指标自校准 | PU21 映射点对照 MATLAB 参考逐点一致；VMAF 区分度验证（86.5 / 85.6 / 地板 97.43） |
| 跳过语义分离 | 能力边界记 SKIP、环境阻塞记 SKIP 并打印原因，不污染失败率 |
| 资源占用测量 | K 组用 `ru_maxrss` 采子进程峰值 RSS |
| 双产物报告 | `bench_report.json`（机器可读）+ `bench_report.md`（人读） |

### 3.2 总体结果

| 指标 | 数值 |
|---|---|
| 注册后端 | 9 名称 / 8 实现 |
| 测试组 | 11 组（A–K） |
| 总测试项 | 93 |
| ✅ 通过 | **71** |
| ❌ 失败 | **0** |
| ⏭ 跳过 | 22（14 能力边界 + 8 环境阻塞） |
| 总耗时 | 98.5s（512×384，CPU） |

**初步结论：已执行路径零失败**。但 93 项中仅 71 项产生结论，**覆盖率 76%**，不构成完整验收。

### 3.3 覆盖矩阵（后端 × 测试组）

图例：✅ 全通过 ｜ 🔶 部分通过 ｜ ⏭ 环境阻塞 ｜ — CLI 级不重复计入

| 后端 \ 组 | A | B | C | D | E | F | G | H | I | J | K | 覆盖率 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **hdrconv** | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | ✅ | — | **21/21 (100%)** |
| **pyultrahdr** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **25/26 (96%)** |
| **gmnet** | ✅ | 🔶 | 🔶 | 🔶 | ✅ | — | — | ✅ | ✅ | ✅ | — | **19/21 (90%)** |
| **sdr2hdr** | ✅ | 🔶 | ⏭ | ⏭ | ✅ | — | — | ⏭ | ⏭ | ⏭ | — | **6/20 (30%)** |
| **unet-onnx** | ⏭ | ⏭ | ⏭ | ⏭ | ⏭ | — | — | ⏭ | ⏭ | ⏭ | — | **0/21** ❌ |
| **lumi-qwen** | ⏭ | ⏭ | ⏭ | ⏭ | ⏭ | — | — | ⏭ | ⏭ | ⏭ | — | **0/21** ❌ |
| **lumi-klein4b** | ⏭ | ⏭ | ⏭ | ⏭ | ⏭ | — | — | ⏭ | ⏭ | ⏭ | — | **0/21** ❌ |
| **lumi-klein9b** | ⏭ | ⏭ | ⏭ | ⏭ | ⏭ | — | — | ⏭ | ⏭ | ⏭ | — | **0/21** ❌ |
| **ultrahdr** | ⏭ | ⏭ | ⏭ | ⏭ | ⏭ | — | — | ⏭ | ⏭ | ⏭ | — | **0/21** ❌ |

**统计口径提醒**：矩阵按后端列给出"已通过 / 该后端理论测试项"的分母。**已实际执行并产生结论的仅 71 项；22 项跳过中，14 项为能力边界（预期内）、8 项为环境阻塞（需在具备依赖的环境补测）。**

### 3.4 测试组与验证点对应

| 组 | 名称 | 验证点 | 判定标准 | 初验结果 |
|---|---|---|---|:---:|
| A | 后端可用性 | 优雅降级、依赖检测 | 不可用记 SKIP + 原因（环境因素不算失败） | ✅ |
| B | 输出格式矩阵 | `--format` 7 格式结构合法 | EXR 可读且峰值 >250 nits；JPEG 有 MPF/XMP；PNG 16bit | ✅ |
| C | 增益图方法 | `--gain-map-method` 4 值生效 | 峰值、dr_max_stops、boost_mids 采集 | ✅ |
| D | 目标峰值亮度 | `--target-nits` 4 档 | 250 < 峰值 ≤ 目标×1.10，记录截断率 | ✅ |
| E | EXIF 保留矩阵 | EXIF A+C 方案 | JPEG 有 EXIF APP1；EXR header 有 cameraMake/isoSpeed；PNG 有 eXIf；no-keep 无泄漏 | ✅ |
| F | 并行扩展 | `--workers` 1/4/auto | 退出码 0 + 输出数正确 | 🔶 功能通过，加速比未验证 |
| G | 干跑安全 | 默认干跑零写入 | 退出码 0 且输出目录不存在 | ✅ |
| H | 确定性 | 可复现承诺 | 同输入两次输出字节完全相同 | ✅ |
| I | 无参考客观质量 | 真实 HDR + SDR 保真 | rt-VMAF / rt-SSIM / rt-PSNR / DR / base 保真 | 🔶 3 后端已测，学习型后端未测 |
| J | 全参考客观质量 | vs 合成 ITM GT | ∆ICtCp PSNR(BT.2124) + PU21 PSNR/SSIM | 🔶 同上 |
| K | 资源占用 | 内存可控 | 子进程峰值 RSS，退出码 0 | 🔶 仅代表后端 |

### 3.5 客观质量指标实测（已测 3 后端）

| 后端 | rt_VMAF ↑ | rt_SSIM ↑ | rt_PSNR(dB) ↑ | DR stops | boost_shadows | boost_highlights | ∆ICtCp PSNR(dB) ↑ | PU21 SSIM ↑ |
|---|---|---|---|---|---|---|---|---|
| pyultrahdr | 100 | 0.921 | 20.1 | 2.96 | 0.61 | 1.95 | 111.0 | 1.000 |
| hdrconv | 100 | 0.921 | 20.1 | 2.96 | 0.61 | 1.95 | 111.0 | 1.000 |
| gmnet | 100 | 0.950 | 22.5 | 2.43 | 0.08 | 0.54 | 43.4 | 0.967 |

**指标解读**：

1. **真实 HDR 已达成**（对已测 3 后端）：DR 2.43–2.96 stops，峰值 >250 nits，输出非格式改名。
2. **两条技术路线各有其位**：pyultrahdr / hdrconv 用 ITM 解析曲线，提升激进（高光 +1.95 stops）、与合成 GT 同源（∆ICtCp 111dB），rt_SSIM 0.921 反映色调映射对 SDR 外观的改变量；gmnet 用学习型增益图，提升保守（高光 +0.54 stops）、SDR 保真更高（rt_SSIM 0.950），∆ICtCp 43.4dB 差异为"学习方法 ≠ 解析曲线"的预期内差异。
3. **rt_VMAF=100 的解读**：VMAF 对平滑色调变化不敏感（已文档化于 `utils/hdr_metrics.py`），须结合 SSIM 阅读。该特性由测试自身识别并记录。
4. **指标局限**：合成 GT 为共享 ITM 曲线，对学习型方法天然不利；**尚未在真实照片集上建立质量基线**。

### 3.6 单元级验证（开发期已完成）

| 模块 | 验证内容 | 结果 |
|---|---|---|
| `utils/hdr_metrics.py` | PU21 映射点对照 MATLAB 参考（100nits→256.38 / 10000→595.39 / 0.005→0） | ✅ 逐点一致 |
| | SSIM（相同=1.0000，σ=0.05 噪声=0.9849）、∆ICtCp（相同=inf，2% 噪声=75.2dB） | ✅ |
| | VMAF 区分度（局部损坏 86.5 / JPEG q30 85.6 / 地板 97.43） | ✅ |
| `utils/exif.py` | TIFF IFD 偏移解析、SHORT 对 RATIONAL、EXR 属性映射 | ✅ 7 字段 |
| `utils/exr_io.py` | half-float 读写往返、通道布局（B,G,R）、header 属性 | ✅ |
| `utils/png16.py` | 16bit 结构、eXIf chunk、归一化语义 | ✅ |
| `backends/pyultrahdr_backend.py` | UltraHDR 段结构（SOI+EXIF+XMP+MPF+base+gainmap）、EXIF 顺序 | ✅ 字节级 |
| `backends/gmnet_backend.py` | 检查点加载（无 missing/unexpected 键）、渲染公式复刻 | ✅ |
| `backends/subprocess_backend.py` | ICC 嵌入字节数（13300/9176）、EXIF+ICC 共存 | ✅ |

### 3.7 性能参考

512×384 / CPU：pyultrahdr · hdrconv ~20ms/张，gmnet ~0.2s/张（不含模型加载），sdr2hdr ~50ms/张。4 进程池对 4 张小图无加速（进程池启动开销 > 收益，与 photo_dedup 实测结论一致）——属已解释现象，**千张级加速比尚未实测**。

---

## 四、初验检查项（非验收门槛）

> 以下为初验阶段的形式与完整性检查，用于判断是否具备进入完整验收测试的条件，**不具备验收否决效力**。

| # | 初验检查项 | 现状 | 判定 |
|:--:|---|:---:|:---:|
| P1 | 立项 CLI 参数全部落地 | 10/10（+`mps` 超集） | ✅ |
| P2 | 立项开源项目均有落地形态或范围外判定 | 11/11 无遗漏 | ✅ |
| P3 | 4 个 Phase 里程碑交付物齐备 | 全部产出 | ✅ |
| P4 | 基准测试框架可复现 | 4 条复现命令已文档化 | ✅ |
| P5 | 质量指标实现经单元级自校准 | PU21/VMAF/SSIM/∆ICtCp 均验证 | ✅ |
| P6 | 默认干跑零写入 | G 组 0 文件泄漏 | ✅ |
| P7 | 全部后端代码路径可执行 | 9 名称 / 8 实现齐备 | ✅ |
| P8 | 已执行路径无失败项 | 71 通过 / 0 失败 | ✅ |

**初验判定：✅ 通过 —— 允许进入完整验收测试阶段。**

---

## 五、验收前置条件（当前未满足项）

以下 7 项为**完整验收的阻断项**，当前均未满足，是本项目尚不能验收通过的原因。「承接阶段」列标明该项由 §7 的哪个硬件阶段关闭。

| # | 阻断项 | 当前状态 | 承接阶段 | 关闭标准 | 建议责任 |
|:--:|---|---|:---:|---|---|
| **B1** | 5 个后端名（4 个实现）零端到端验证 | ⏭ 全部环境阻塞 | CPU/T4/L40 | 在具备依赖的环境重跑 A–K 组，无 FAIL（能力边界 SKIP 可接受） | 补测人 |
| **B1.1** | `unet-onnx` | OOM 被 OS 杀死 | **T4** | `pip install onnxruntime-gpu` + `--device cuda`（4K 秒级）；模型已就位（7.4MB） | 补测人 |
| **B1.2** | `lumi-klein4b` | 8GB 权重未下载 | **T4** | `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download`；~12GB 显存；Apache 2.0，最轻。**限 4bit/8bit 路径**（Turing 无 bf16） | 补测人 |
| **B1.3** | `lumi-qwen` | 54GB 权重未下载 | **L40** | bf16 需 ~24GB 显存且依赖 bf16 支持（Turing 不具备） | 补测人 |
| **B1.4** | `lumi-klein9b` | 17GB 权重未下载 | **L40** | 模型 + 激活 >16GB 且 bf16 路径；需授权 | 补测人 |
| **B1.5** | `ultrahdr` | 二进制未编译 | **CPU** | `cmake -G Ninja .. && ninja`；产物入 PATH 或设 `ULTRHDR_APP_PATH` 或 `--ultrahdr-binary`。**纯 CPU 后端，无需 GPU** | 补测人 |
| **B2** | 学习型后端无质量指标（I/J 组空白） | 缺口 | T4 + L40 | unet-onnx / klein4b 在 T4 采集；qwen / klein9b 在 L40 采集 | 补测人 |
| **B3** | 测试素材为合成图 | 512×384 合成 | **CPU** | ≥20 张真实照片（夜景 / 人像 / 高动态场景）回归，建立质量基线 | 补测人 |
| **B4** | 千张级压测未做 | 未做 | **CPU** | ≥1000 张批量，记录 workers 1/N/auto 加速比与峰值内存 | 补测人 |
| **B5** | 真实 HDR 显示器主观评估未做 | 未做 | **L40** | 在 HDR 显示器上对 UltraHDR / Gainmap 输出做主观评估记录（需 GPU 实时渲染输出） | 测试人 |
| **B6** | UltraHDR 合规性未做第三方校验 | 内部字节级验证 | **CPU** | 引入 Google 参考样本或 libultrahdr（已编译后端）交叉比对 | 补测人 |
| **B7** | 测试覆盖统计口径待核对 | 见 §3.3 提醒 | **CPU** | `TEST_COVERAGE.md` §1「8 项环境阻塞」与 §4 标题/覆盖矩阵反推（5 项 A 组）口径不一致，需统一后再签署 | 测试负责人 |

**已就位的缓解措施**：unet-onnx 已加 ≥5GB 可用内存预检，将不可捕获的 SIGKILL 转为明确报错（优雅降级）。

**阶段归属小结**：CPU 阶段承接 B1.5 / B3 / B4 / B6 / B7 五项；T4 阶段承接 B1.1 / B1.2 及 B2 的一半；L40 阶段承接 B1.3 / B1.4、B2 另一半与 B5。

---

## 六、已知缺口清单

| # | 缺口 | 影响 | 当前处理 |
|:--:|---|---|---|
| 1 | 学习型后端（unet-onnx / lumi-*）无本环境端到端质量数据 | I/J 组矩阵不完整 | 环境限制，非代码缺陷；B1 关闭后补全 |
| 2 | sdr2hdr 无质量指标（ICC 嵌入是显示技巧，不产生线性 HDR） | I/J 组不适用 | 能力本质决定，效果只能在真实 HDR 显示器上主观评估（B5） |
| 3 | VMAF 已知边界（单帧触发帧同步异常已规避；对平滑色调不敏感；地板 97.43） | 指标解释需谨慎 | 已文档化并有单元级区分度验证支撑 |
| 4 | 测试图为合成素材 | 与真实照片统计特性有差距 | 结构化可复现；B3 补真实照片回归 |
| 5 | F 组并行加速未体现（4 张小图 workers=4 与 1 持平） | 千张级加速比未知 | 已解释现象；B4 补压测 |

---

## 七、完整验收测试计划（三阶段：CPU → T4 → L40）

### 7.1 阶段划分原则

| 原则 | 说明 |
|---|---|
| **硬件唯一归属** | 每个测试项只归属一个阶段，避免重复劳动与结果冲突 |
| **CPU 优先** | 仅 CPU 即可完成的测试准备与实测，全部前置到 CPU 阶段做完 |
| **Ada 隔离** | 必须依赖 Ada Lovelace 架构特性的测试项，只在 L40 阶段执行 |
| **递进不重复** | 后一阶段不重跑前序已通过项，仅做回归抽查 + 新增能力测试 |
| **基线冻结** | 每阶段准出即冻结该阶段的测试集版本、参数与报告产物，作为后续比对基线 |

### 7.2 阶段总览

| 阶段 | 硬件 | 承接后端 | 承接阻断项 | 核心产出 | 状态 |
|:--:|---|---|---|---|:--:|
| **阶段一** | CPU（8 核 / 8GB，**已具备**） | hdrconv、pyultrahdr、sdr2hdr、ultrahdr、gmnet(CPU) | B1.5、B3、B4、B6、B7 | 真实照片基线表、CPU 全量 bench、压测报告、合规比对记录 | ⬜ 未启动 |
| **阶段二** | T4（NVIDIA T4 / 16GB 显存） | unet-onnx、lumi-klein4b、gmnet(GPU) | B1.1、B1.2、B2（部分） | GPU 通路验证报告、GPU 吞吐与显存基线、CPU↔GPU 一致性比对 | ⬜ 未启动 |
| **阶段三** | L40（Ada / 48GB 显存） | lumi-qwen、lumi-klein9b | B1.3、B1.4、B2（部分）、B5 | Ada 专属项全量报告、bf16 量化矩阵、主观评估记录 | ⬜ 未启动 |

> **硬件规格（已确认，2026-10-13）**：T4 = **NVIDIA T4（Turing SM75 / 16GB GDDR6）**，硬件不支持 bf16 —— 这是 Ada 专属测试项被隔离到 L40 的技术根因；L40 = **NVIDIA L40（Ada Lovelace SM89 / 48GB GDDR6）**。三阶段测试归属据此确定，不再重算。

### 7.3 阶段一：CPU（测试准备 + 全量 CPU 实测）

**定位**：本阶段承担**全部测试准备**工作，使后续两个阶段无需再准备素材；并完成所有 CPU 可闭环的实测项。

**测试准备（后续阶段直接复用，不重复做）**

| 编号 | 准备项 | 内容 | 产出 |
|:--:|---|---|---|
| P1 | 真实照片测试集 | ≥20 张，覆盖夜景 / 人像 / 高动态场景 / 纹理密集 / 低饱和；含带完整 EXIF 样本（相机/ISO/快门/光圈/焦距/GPS） | `testdata/real/` + 清单 + SHA256 + 版权说明 |
| P2 | 分辨率矩阵 | 512×384（现有基准）/ 1K / 2K / **2160×3840**（unet-onnx 固定输入）/ 8K 抽样 | `testdata/matrix/` |
| P3 | 合成图基线 | 沿用固定种子 42，作为可复现的确定性基准 | 已具备 |
| P4 | 编译 `ultrahdr_app` | `cmake -G Ninja .. && ninja`，产物入 PATH 或设 `ULTRHDR_APP_PATH` | 二进制 + 构建日志 |
| P5 | bench 扩展 | 新增 **L 组（真实照片质量基线）**；B 组扩展分辨率维度；新增 M 组（跨阶段一致性比对接口） | bench 修订版 |
| P6 | 参考样本 | Google UltraHDR 参考样本 + libultrahdr 产物，用作合规性比对基准 | `testdata/reference/` |

**实测内容**

| 编号 | 实测项 | 覆盖后端 | 对应组/阻断项 |
|:--:|---|---|---|
| R1 | CPU 全量基准（9 后端全注册，环境阻塞自动 SKIP） | 全部 | T2 全量 / B7 |
| R2 | `ultrahdr` 后端首次端到端（A–K 全组） | ultrahdr | **B1.5** |
| R3 | 真实照片集质量基线（I/J 组，≥20 张 × 全部可用后端） | hdrconv / pyultrahdr / gmnet / sdr2hdr | **B3**、B2 部分 |
| R4 | 千张级压测（workers 1/N/auto，加速比 + 峰值内存 + OOM 恢复） | CPU 后端 | **B4** |
| R5 | UltraHDR 合规性交叉校验（pyultrahdr ↔ libultrahdr ↔ Google 参考样本） | pyultrahdr / ultrahdr | **B6** |
| R6 | 统计口径统一 + `TEST_COVERAGE.md` 更新 | — | **B7** |
| R7 | 确定性 / 干跑 / EXIF / 格式 / 增益方法 / 峰值亮度全矩阵复测 | CPU 后端 | G/H/E/B/C/D |

**准出条件**

- R1–R7 全部完成，`bench_report.json` 中 FAIL = 0
- 真实照片基线表落盘，≥20 张 × ≥4 后端 指标齐全
- 压测报告：加速比与峰值内存记录在案，无 OOM
- 测试集版本冻结（清单 + 校验和），提交后续阶段复用
- **本阶段不得遗留任何「环境阻塞」类跳过项**

---

### 7.4 阶段二：T4（GPU 通用通路，不含 Ada 专属项）

**定位**：验证**所有需要 GPU 参与、但不依赖 Ada 架构特性**的测试组合。

**Turing 架构对本阶段的硬约束**（硬件已确认为 NVIDIA T4 / SM75 / 16GB）

| 约束 | 影响 | 应对 |
|---|---|---|
| **不支持 bf16** | `lumi-*` 默认 `--quantization none`（bf16 路径）在 T4 上**必然失败**；`torch.cuda.is_bf16_supported()` 返回 False | 本阶段所有 `lumi-*` 调用**必须显式传 `--quantization 4bit` 或 `8bit`**；`none` 档跳过并标注「待 L40 阶段」 |
| 16GB 显存上限 | `lumi-klein9b`（~17GB 权重）无法加载；单进程大模型推理接近上限 | 本阶段不测 klein9b / qwen；`klein4b`（~8GB）留足激活余量 |
| 多进程显存共享 | workers>1 时每个 worker 各占一份显存，易 OOM | 显存预算约束 worker 数收敛，由 R13 验证 |

> ⚠️ **执行提醒**：T4 上跑 `lumi-klein4b` 时若遗漏 `--quantization 4bit`，将因 bf16 不受支持而直接报错。此为**硬件能力边界，非代码缺陷**，但需在测试记录中明确标注以免误判。建议在 lumi 后端补 bf16 可用性预检（参照 unet-onnx 的内存预检做法），使报错信息可读。

**测试准备**

| 编号 | 准备项 | 内容 |
|:--:|---|---|
| P7 | GPU 运行时 | `onnxruntime-gpu`、CUDA 版 torch；验证 `CUDAExecutionProvider` 挂载成功 |
| P7.1 | bf16 能力确认 | `torch.cuda.is_bf16_supported()` → 预期 **False**，作为阶段二的基线断言记录在案 |
| P8 | 权重下载 | `lumi-klein4b`（~8GB，Apache 2.0）；HF 镜像加速 |
| P9 | 显存基线 | 各后端单进程显存占用实测，标定 `estimate_worker_memory()` 的 GPU 估算 |
| P10 | OOM 复现用例 | 构造显存不足场景，验证预检与降级行为 |

**实测内容**

| 编号 | 实测项 | 覆盖后端 | 说明 |
|:--:|---|---|---|
| R8 | `--device cuda` 设备解析与 providers 挂载 | 全部 GPU 后端 | 验证 auto→cuda 探测逻辑 |
| R9 | `unet-onnx` 端到端 A–K 全组（fp32 / fp16） | unet-onnx | **B1.1** 关闭；2160×3840 推理秒级 |
| R10 | `lumi-klein4b` 端到端（**4bit / 8bit 路径**） | lumi-klein4b | **B1.2** 关闭；`none`(bf16) 档受 Turing 限制跳过，待 L40 |
| R11 | `gmnet` CUDA 推理 + CPU/GPU 指标比对 | gmnet | 量化 CPU 阶段基线的硬件差异 |
| R12 | GPU 批量吞吐与显存峰值（workers 1/2/4/auto） | 全部 GPU 后端 | 单卡多进程显存共享行为 |
| R13 | worker 数收敛验证（显存预算 vs 核数） | — | 验证 `resolve_workers()` GPU 分支 |
| R14 | **CPU ↔ T4 输出一致性比对** | unet-onnx / gmnet | 见 §7.6 |
| R15 | bf16 缺失时的报错可读性验证 | lumi-klein4b | 不传 `--quantization` 时应给出明确错误，而非栈溢出/静默降级 |

**准出条件**

- `unet-onnx`、`lumi-klein4b` A–K 全组无 FAIL，I/J 组指标落盘
- `--device cuda` 路径验证通过，`CUDAExecutionProvider` 生效
- GPU 吞吐与显存峰值记录在案
- CPU↔T4 一致性比对完成并给出差异归因
- 报告中跳过项 100% 为能力边界（含 bf16/Turing 与 klein9b 显存两项，均已标注原因）

---

### 7.5 阶段三：L40（Ada 专属测试项）

**定位**：**只负责必须依赖 Ada Lovelace 架构的测试项**，不重复前两阶段的常规验证。

**Ada 专属测试项清单（本阶段存在的唯一理由）**

| 编号 | 测试项 | 必须 Ada 的原因 | 覆盖后端 | 关闭 |
|:--:|---|---|---|---|
| **A1** | bf16 精度路径（`--quantization none`） | T4 为 Turing SM75，**硬件不支持 bf16**；bf16 由 Ampere(SM80) 引入 | lumi-* | B2 |
| **A2** | `lumi-qwen` bf16 推理 | 权重 ~54GB + 推理需 ~24GB 显存，且依赖 bf16 | lumi-qwen | **B1.3** |
| **A3** | `lumi-klein9b` bf16 推理 | 模型 ~17GB + 激活 >16GB（T4 显存不足），需 bf16 | lumi-klein9b | **B1.4** |
| **A4** | 量化矩阵完整对照（none / 8bit / 4bit） | `none` 档即 bf16，缺 Ada 无法跑完整矩阵 | lumi-* | B2 |
| **A5** | fp16 ↔ bf16 数值一致性对照 | bf16 尾数位与累加行为需 Ada tensor core 才能比对 | lumi-* | — |
| **A6** | 大显存单卡多 worker 并发压测 | 48GB 显存才支撑多进程 GPU 推理并发 | 全部 GPU 后端 | B4 补充 |
| **A7** | L40 ↔ CPU/T4 指标差异归因 | 同输入跨代硬件的数值差异分析 | 全部 GPU 后端 | — |
| **A8** | HDR 显示器主观评估 | 需 GPU 实时渲染 UltraHDR / Gainmap 输出 | 全部 | **B5** |

**测试准备**

| 编号 | 准备项 | 内容 |
|:--:|---|---|
| P11 | 权重下载 | `lumi-qwen`（~54GB）、`lumi-klein9b`（~17GB，需授权）；存储需求 ≥150GB |
| P12 | 精度环境 | 确认 bf16 可用（`torch.cuda.is_bf16_supported()` 返回 True） |
| P13 | 主观评估物料 | 选取代表性输出（UltraHDR / Gainmap / EXR→显示器映射）+ 评分表 |

**实测内容**

| 编号 | 实测项 | 覆盖后端 |
|:--:|---|---|
| R16 | `lumi-qwen` 端到端 A–K 全组（bf16 主路径） | lumi-qwen |
| R17 | `lumi-klein9b` 端到端 A–K 全组（bf16 主路径） | lumi-klein9b |
| R18 | 量化矩阵全档对照（A4）+ 显存/耗时/质量三维记录 | lumi-* |
| R19 | 大显存并发压测（A6） | 全部 GPU 后端 |
| R20 | 跨阶段一致性比对与差异归因（A7） | 全部 |
| R21 | HDR 显示器主观评估（A8） | 全部 |
| R22 | 阶段性回归抽查（前两阶段已通过项的抽样复跑） | 全部 |

**准出条件**

- 9 个后端名全部完成端到端验证（A–K 组无 FAIL）
- Ada 专属项 A1–A8 全部完成并有记录
- 量化矩阵三档数据齐全，显存与质量权衡有结论
- 主观评估表完成，UltraHDR / Gainmap 输出在 HDR 显示器上确认可视
- 跨阶段一致性比对完成，差异全部有归因
- `TEST_COVERAGE.md` 更新为最终版，环境阻塞跳过项归零

---

### 7.6 跨阶段一致性比对规则

同一输入在三阶段分别执行转换后，比对规则如下：

| 类别 | 比对对象 | 判定标准 |
|---|---|---|
| **阶段内确定性** | 同阶段两次转换 | 输出字节**完全一致**（延续 H 组已有结论） |
| **同硬件不同后端** | 如 pyultrahdr ↔ ultrahdr（同 ITM 路线） | 核心指标差异 ≤5%，段结构一致 |
| **CPU ↔ T4** | gmnet、unet-onnx | rt_SSIM ≥0.95、rt_PSNR ≥35dB；差异归因于算子/精度 |
| **T4 ↔ L40** | lumi-klein4b 4bit 档 vs klein9b 4bit 档 | 记录差异，不设强门禁（模型不同） |
| **同模型跨卡** | lumi-klein4b 在 T4(4bit) vs L40(4bit/bf16) | 4bit 档差异 ≤容差；bf16 档为质量上限基线 |
| **量化档位** | none / 8bit / 4bit | 质量单调性：none ≥ 8bit ≥ 4bit（PSNR 不倒挂） |

> 容差具体数值待 T2 阶段首次跨卡比对后确定并记录。

### 7.7 阶段进度追踪表

| 阶段 | 计划起止 | 实际起止 | 完成度 | 产出位置 | 阻塞项 |
|---|---|---|:--:|---|---|
| 阶段一 CPU | 待排期 | — | 0% | — | — |
| 阶段二 T4 | 待排期（依赖阶段一 P1–P6 完成） | — | 0% | — | — |
| 阶段三 L40 | 待排期（依赖阶段二 P7–P10 完成） | — | 0% | — | — |

### 7.8 拟采用的验收判定准则（待验收会确认）

| 类别 | 准则 | 目标值 | 负责阶段 |
|---|---|---|:---:|
| 功能 | 立项 CLI 参数落地率 | 100%（10/10） | CPU |
| 功能 | 全后端端到端可用率 | 9/9 后端名可在目标环境运行 | CPU / T4 / L40 |
| 功能 | 批量与递归 | 单次运行 ≥1000 张无中断 | CPU |
| 质量 | bench 失败项 | 0 | 三阶段累计 |
| 质量 | 跳过项性质 | 100% 为能力边界，**无环境阻塞遗留** | 三阶段累计 |
| 质量 | 真实 HDR 判据 | 峰值 >250 nits、DR ≥2 stops | CPU（基线）/ T4 / L40 |
| 质量 | 确定性 | 同阶段两次转换字节一致 | CPU |
| 质量 | EXIF 保留 | keep/no-keep × 格式矩阵无泄漏 | CPU |
| 质量 | 真实照片回归 | ≥20 张真实照片质量基线齐全 | CPU |
| 质量 | 量化档位单调性 | none ≥ 8bit ≥ 4bit（PSNR 不倒挂） | L40 |
| 质量 | 跨阶段一致性 | CPU↔T4 rt_SSIM ≥0.95 且差异有归因 | T4 / L40 |
| 性能 | 千张级并行加速比 | ≥1.5×（相对串行） | CPU / L40 |
| 性能 | 峰值内存 | ≤可用内存 90%，无 OOM | CPU / L40 |
| 性能 | 显存峰值 | ≤显存预算，单卡多 worker 不 OOM | T4 / L40 |
| 合规 | UltraHDR 段结构 | 与 Google 参考样本一致 | CPU |
| 体验 | HDR 显示器主观评估 | UltraHDR / Gainmap 输出可视可接受 | L40 |
| 文档 | 复现命令可执行 | 全部命令在干净环境复现成功 | CPU |

---

## 八、初验签署

| 角色 | 姓名 | 日期 | 意见 |
|---|---|---|---|
| 项目负责人 | | | |
| 技术负责人 | | | |
| 测试负责人 | | | |
| 初验结论 | | | ☐ 同意进入完整验收测试　☐ 需整改后重新初验 |

> 本签署仅确认"实施完成、具备进入完整验收测试的条件"，**不构成项目验收通过**。

---

## 附录 A：复现命令（按验收阶段）

### A.1 通用 / 阶段一 CPU

```bash
# 全量基准（自动跳过环境不可用项，~98s）
python bench_photo_sdr_to_hdr.py --report-dir ./bench_out

# 冒烟子集（~10s）
python bench_photo_sdr_to_hdr.py --quick

# 限定后端
python bench_photo_sdr_to_hdr.py --backends pyultrahdr gmnet

# 编译 ultrahdr 后端后重跑（阶段一 B1.5）
cmake -G Ninja .. && ninja
export ULTRHDR_APP_PATH=<产物路径>
python bench_photo_sdr_to_hdr.py --backends ultrahdr --report-dir ./bench_ultrahdr

# 干跑验证（零写入）
python photo_sdr_to_hdr.py -i ./photos -o ./out --models pyultrahdr

# 实际执行
python photo_sdr_to_hdr.py -i ./photos -o ./out --models pyultrahdr --apply

# 千张级压测（阶段一 B4）
python bench_photo_sdr_to_hdr.py --backends pyultrahdr --report-dir ./bench_stress
```

### A.2 阶段二 T4

```bash
# 准备：GPU 运行时与权重
pip install onnxruntime-gpu
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download oumoumad/LumiPic v5b_step2000.safetensors

# unet-onnx 端到端（阶段二 B1.1）
python bench_photo_sdr_to_hdr.py --backends unet-onnx --report-dir ./bench_t4_unet

# lumi-klein4b（阶段二 B1.2；Turing 无 bf16，用 4bit）
python photo_sdr_to_hdr.py -i ./photos -o ./out --models lumi-klein4b \
    --device cuda --quantization 4bit --apply

# gmnet CUDA 通路 + CPU↔T4 一致性比对
python photo_sdr_to_hdr.py -i ./photos -o ./out --models gmnet --device cuda --apply
```

### A.3 阶段三 L40

```bash
# 准备：bf16 能力确认
python -c "import torch; print(torch.cuda.is_bf16_supported())"   # 须为 True

# lumi-qwen（阶段三 B1.3）
python bench_photo_sdr_to_hdr.py --backends lumi-qwen --report-dir ./bench_l40_qwen

# lumi-klein9b（阶段三 B1.4）
python bench_photo_sdr_to_hdr.py --backends lumi-klein9b --report-dir ./bench_l40_klein9b

# 量化矩阵全档对照（A4）
for q in none 8bit 4bit; do
  python photo_sdr_to_hdr.py -i ./photos -o ./out --models lumi-klein9b \
      --device cuda --quantization $q --apply --log ./quant_$q.log
done

# 全量最终回归（9 后端）
python bench_photo_sdr_to_hdr.py --report-dir ./bench_final
```

**报告产物**：`bench_out/bench_report.json`（机器可读）+ `bench_out/bench_report.md`（人读）
**运行产物**：输出目录下 `photo_sdr_to_hdr_report.*`

## 附录 B：关键实现位置索引

| 能力 | 位置 |
|---|---|
| CLI 参数解析 | `photo_sdr_to_hdr.py:54-227` |
| 默认干跑 / `--apply` 互斥 | `photo_sdr_to_hdr.py:221-225` |
| 启动横幅（系统/设备/并发） | `photo_sdr_to_hdr.py:230-286` |
| 每后端转换参数分发 | `photo_sdr_to_hdr.py:289-311` |
| 并行调度 | `utils/parallel.py:252` `run_parallel_backends()` |
| worker 数解析 | `utils/device.py:292` `resolve_workers()` |
| 内存预估 | `utils/device.py:263` `estimate_worker_memory()` |
| 后端注册表 | `backends/base.py:123` `BackendRegistry` |
| 报告生成 | `utils/report.py` `ReportGenerator` |
| 质量指标实现 | `utils/hdr_metrics.py` |
| 基准测试组 A–K | `bench_photo_sdr_to_hdr.py:167-612` |

## 附录 C：指标口径说明

| 指标 | 含义 | 方向 |
|---|---|---|
| rt_VMAF / rt_SSIM / rt_PSNR | tonemap 回环保真（SDR 外观是否被破坏） | ↑ |
| DR stops | 动态范围提升量 | ↑ 越高 HDR 增强越强 |
| boost_shadows / boost_highlights | 暗部/高光分域提升量（stops） | 场景相关 |
| ∆ICtCp PSNR | vs 合成 ITM GT 的色度保真（BT.2124） | ↑ |
| PU21 PSNR / SSIM | vs 合成 ITM GT 的亮度感知保真（Mantiuk & Azimi, PCS 2021） | ↑ |
| base_SSIM / base_PSNR | UltraHDR 主图（SDR 回退层）对原图的保真 | ↑ |