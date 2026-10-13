# 图像 SDR 转 HDR 开发项目 · 立项书

> **文件类型**：项目立项书（Project Charter）
> **项目编号**：PhoUtils-SDR2HDR-001
> **版本**：v1.0　｜　**编制日期**：2026-10-13
> **项目名称**：图像 SDR 转 HDR 批量转换工具（`photo_sdr_to_hdr.py`）
> **所属工程**：PhoUtils 照片实用工具集
> **项目性质**：工程应用类（复用开源 HDR 生态，自研调度与质量度量层）
> **项目状态**：🟡 开发完成，待完整验收测试
> **审批状态**：☐ 待审批　☐ 已批准　（批准日期：______）

---

## 一、项目背景与立项依据

### 1.1 背景

HDR 显示设备（OLED 手机、QD-OLED / Mini-LED 显示器、兼容 HDR 的流媒体与相册服务）正在快速普及，Android 14 起已原生支持 Ultra HDR 格式。但存量图片资产绝大部分仍是 SDR 8bit 素材，内容方面临"SDR → HDR"的批量转制需求：

- 手机相册、社交平台需要把 SDR 素材升级为 HDR，提升在 HDR 设备上的观感；
- 专业后期需要从 SDR 素材生成线性 HDR 中间件（EXR）用于合成；
- 存量图库迁移需要**可复现、可回退、零覆盖**的批处理管线，而非手工逐张处理。

### 1.2 立项依据

| 依据 | 内容 |
|---|---|
| 需求来源 | `图像SDR转HDR的开项目立项.md`（原始需求与候选项目清单） |
| 技术可行性 | HDR 生态已提供多套可用实现：Google libultrahdr（Android 原生参考库）、hdr-conversion（Python 格式封装）、pyultrahdr（纯 Python UltraHDR 编码器）、GMNet（ICLR 2025 Gain Map 学习方法）、LumiPic（扩散模型 LoRA）、U-Net SDR→HDR 重建等 |
| 生态成熟度 | Ultra HDR（JPEG/R｜ISO 21496-1 Gainmap）已有标准与参考实现；`hdr-conversion` 可 pip 安装，无需编译 |
| 工程价值 | PhoUtils 已沉淀 `photo_dedup.py` 的并发、日志、报告、安全性设计，可直接复用形成统一的 SDR→HDR 工具 |

### 1.3 项目定位

在 PhoUtils 内创建**一个统一 CLI 工具**，封装多个开源 HDR 实现，用统一参数、统一调度、统一报告屏蔽后端差异，支持文件/文件夹批量转换。定位为**工程编排层**而非算法发明层——自研重点放在参数体系、后端抽象、并发与资源调度、EXIF 透传、客观质量度量。

---

## 二、项目目标

### 2.1 总体目标

交付一个生产可用的 SDR→HDR 批量转换 CLI 工具，满足"**参数完整、模型可区分、批量可扩展、过程可观测、结果可回退**"五项工程要求。

### 2.2 量化目标

| 编号 | 目标 | 量化指标 |
|:--:|---|---|
| G1 | 统一 CLI 工具 | 单一入口 `photo_sdr_to_hdr.py`，参数体系覆盖下述 L1 清单 |
| G2 | 模型可区分 | `--models` 支持 ≥8 个后端名 / ≥7 个独立实现，含解析式与学习式两类路线 |
| G3 | 批量可扩展 | 单次运行支持多输入路径、目录递归、≥1000 张规模 |
| G4 | 过程可观测 | 结构化启动横幅 + 实时进度 + 落盘运行报告（成功率/失败明细/耗时/资源） |
| G5 | 结果可回退 | 默认干跑零写入；写操作需显式 `--apply`；原图不被修改 |
| G6 | 真实 HDR 输出 | EXR 输出峰值 >250 nits，动态范围提升 ≥2 stops |
| G7 | 元数据保真 | EXIF 方案 A+C：原生注入 + `.exif.json` 旁车，`--no-keep-exif` 无泄漏 |
| G8 | 可复现 | 同输入两次转换输出字节一致 |
| G9 | 质量可度量 | 自研 VMAF / PU21 / ∆ICtCp / SSIM / DR 指标，经参考实现校准 |
| G10 | 优雅降级 | 依赖或模型缺失时明确报错或降级，不静默失败、不整批中断 |

### 2.3 非目标（明确不做）

| 非目标 | 理由 |
|---|---|
| 研发新的 SDR→HDR 算法 | 定位为编排层，复用开源实现 |
| 训练 / 微调深度模型 | 复用公开预训练权重 |
| HDR→SDR 反向色调映射（gainforge 类） | 与项目目标方向相反 |
| Web / three.js 前端编码（gainmap-js 类） | 生态定位不符（本项目为 Python CLI） |
| 多格式互转码器（AVIF/JXL/HEIF 全互转） | 非核心需求；`--format avif` 经 hdrconv 附带支持 |
| HDR 显示器主观评价体系建设 | 需专用硬件，属验收阶段配合项 |

---

## 三、项目范围

### 3.1 范围内

| 模块 | 内容 |
|---|---|
| CLI 层 | 参数解析、校验、互斥检查、启动横幅、帮助文档 |
| 后端抽象 | `SDRToHDRBackend` 基类 + `BackendRegistry` 注册表 + 统一能力声明 |
| 后端实现 | pyultrahdr / hdrconv / gmnet / unet-onnx / LumiPic×3 / sdr2hdr / ultrahdr |
| 调度层 | 文件发现、输出路径生成、进程池并行、worker 数解析、内存/显存预估、设备选择 |
| 输出层 | EXR half-float 线性 / PNG 16bit / JPEG / UltraHDR 容器 / ISO 21496-1 Gainmap 容器 |
| 元数据层 | EXIF 解析与注入（JPEG APP1 / PNG eXIf / EXR header 属性）、旁车 JSON |
| 度量层 | VMAF / PU21 / ∆ICtCp / SSIM / PSNR / 动态范围 / 分域提升 / 截断率 |
| 可观测层 | 统一日志、进度条、运行报告生成 |
| 测试层 | 综合基准脚本 + 测试覆盖报告 |
| 文档层 | 立项书、执行方案、实施及初验报告、README |

### 3.2 范围外

见 §2.3 非目标。另：**真实 HDR 显示器主观评估**、**第三方 UltraHDR 合规性校验**属验收阶段环境与配合需求，由验收方提供条件。

### 3.3 候选开源项目评估与选型结论

| 候选项目 | 成熟度 | 集成成本 | 输出能力 | 选型结论 |
|---|---|---|---|---|
| **pyultrahdr** | 社区（~230 行） | 极低（纯 Python） | Ultra HDR JPEG | ✅ **首选**：零编译依赖，能力覆盖 UltraHDR 输出 |
| **hdr-conversion** | pip 可装 | 低 | UltraHDR / Gainmap / PQ-HLG | ✅ **首选**：格式封装合规性高 |
| **GMNet** | ICLR 2025 论文 | 中（PyTorch） | 线性 HDR（EXR）+ Gain Map | ✅ **首选**：学习型路线代表，SDR 保真更优 |
| **LumiPic** | 研究级 | 高（数十 GB 权重） | 浮点 HDR EXR，LogC3 | ✅ 纳入：专业后期路线，提供量化选项 |
| **SDR-to-HDR-Reconstruction-System** | 研究级 | 中（ONNX） | 线性 HDR | ✅ 纳入：轻量 U-Net，工程门槛低 |
| **sdr2hdr** | 社区（Rust） | 中（cargo） | ICC 嵌入（显示增强） | ✅ 纳入：作为"显示端技巧"补充路线，**不产线性 HDR**，验收需区分口径 |
| **libultrahdr** | 官方（Android 原生） | 高（需 C++ 编译） | Ultra HDR | ✅ 纳入：合规性基准，与 pyultrahdr 交叉验证 |
| HDR Image Transcoder | 社区 | 中 | Gainmap AVIF / AVIF-JXL-HEIF | 🔶 部分纳入：仅取 avif 输出路径 |
| gainmap-js | 官方（TS） | — | three.js 编码 | ❌ 范围外：Web 生态 |
| gainforge | 社区（Rust） | — | HDR→SDR | ❌ 范围外：方向相反 |
| Awesome-Gain-Maps | 资源列表 | — | — | ❌ 范围外：非软件 |

---

## 四、需求规格（L1 级）

### 4.1 CLI 参数规格

**必须实现**（立项硬性要求）：

| 参数 | 短选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `--input` | `-i` | str（可多个） | 必填 | 输入文件或目录 |
| `--output` | `-o` | str | 自动（`_hdr_output/`） | 输出目录 |
| `--models` | — | str（可多个） | `hdrconv` | 转换后端，可多选并行对比 |
| `--workers` | `-w` / `--threads` | int | 0（自动） | 0=按核数/内存自动，1=串行 |
| `--device` | — | choice | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `--suffix` | — | str | `_hdr` | 输出文件名后缀 |
| `--recursive` | `-r` | flag | False | 递归子目录 |
| `--dry-run` | — | flag | 默认开 | 预览模式，不写文件 |
| `--log` | — | str | `-` | 日志文件路径（`-`=stdout） |
| `--help` | `-h` | flag | — | 帮助 |

**计划扩展**（可后延）：`--apply`、`--format`、`--quantization`、`--colorspace`、`--model-dir`、`--keep-exif`、`--sidecar`、`--gain-map-method`、`--target-nits`、`--sdr2hdr-mode`、`--ultrahdr-binary`、`--version`。

### 4.2 功能性需求

| 编号 | 需求 | 优先级 |
|:--:|---|:--:|
| FR1 | 文件与文件夹批量输入，多路径，去重保序 | P0 |
| FR2 | 递归子目录扫描，输出保持相对目录结构 | P0 |
| FR3 | 多后端并行对比，输出文件名含后端标识 | P0 |
| FR4 | 进程池并行，worker 数按核数/内存/显存自动收敛 | P0 |
| FR5 | 设备自动选择（CUDA → MPS → CPU） | P0 |
| FR6 | 单文件失败不中断整批，错误入报告 | P0 |
| FR7 | 输出格式统一为内部线性 nits 语义（EXR half-float / Rec.709） | P0 |
| FR8 | EXIF 保留方案 A+C（原生注入 + 旁车） | P1 |
| FR9 | 增益图方法可配置（auto/linear/log/fixed） | P1 |
| FR10 | 目标峰值亮度可配置（500/1000/2000/4000 nits） | P1 |
| FR11 | LumiPic 量化选项（none/8bit/4bit）缓解显存压力 | P1 |
| FR12 | 模型自动下载至 `models/` 子目录，支持离线复用与 HF 镜像切换 | P1 |
| FR13 | 运行报告落盘（配置快照 + 逐文件结果 + 统计 + 耗时） | P1 |
| FR14 | SIGINT 优雅关闭，保留已完成结果 | P2 |
| FR15 | 磁盘空间预检与预警 | P2 |

### 4.3 非功能性需求

| 编号 | 需求 | 指标 |
|:--:|---|---|
| NFR1 | 性能 | 512×384 / CPU：解析式后端 ≤50ms/张；学习式 ≤1s/张（不含模型加载） |
| NFR2 | 并行收益 | 千张级批量下 workers=N 相对串行加速比 ≥1.5× |
| NFR3 | 内存 | 峰值 RSS ≤ 可用内存 90%，无 OOM；不可捕获的 SIGKILL 须转为预检报错 |
| NFR4 | 确定性 | 同输入两次转换输出字节一致 |
| NFR5 | 幂等安全 | 默认不写任何文件；输出同名自动加序号后缀，不覆盖 |
| NFR6 | 可移植 | Python ≥3.10，核心依赖仅 Pillow + numpy；重依赖按后端懒加载 |
| NFR7 | 可观测 | 关键决策（并发度、设备选择、跳过原因）均打印依据 |
| NFR8 | 合规性 | UltraHDR 输出段结构符合规范（SOI+EXIF+XMP+MPF+base+gainmap） |

---

## 五、技术方案概要

### 5.1 总体架构

```
┌─────────────────────────────────────────────────────────┐
│  photo_sdr_to_hdr.py  (CLI 入口 / 参数 / 横幅 / 报告)      │
└───────────────┬─────────────────────────────────────────┘
                │
    ┌───────────▼────────────┐
    │ utils/  调度与基础设施  │  file_io · device · parallel
    │                        │  logging · report
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │ backends/  后端实现     │  base(SDRToHDRBackend + Registry)
    │                        │  pyultrahdr · hdrconv · gmnet
    │                        │  unet_onnx · lumipic ×3
    │                        │  subprocess(sdr2hdr · ultrahdr)
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │ utils/  算法与度量      │  itm · exr_io · png16 · exif
    │                        │  hdr_metrics(VMAF/PU21/∆ICtCp/SSIM/DR)
    └────────────────────────┘
```

### 5.2 后端抽象接口

```python
class SDRToHDRBackend:
    name: str                     # 后端标识
    description: str              # 简要说明
    output_formats: List[str]     # 支持的输出格式
    def check_available(self) -> Tuple[bool, str]   # (可用, 原因)
    def prepare(self, device: str, **kwargs) -> None
    def convert(self, in_path: str, out_path: str, **kwargs) -> bool
    def convert_batch(self, paths: List[str], out_dir: str, **kwargs)
```

**关键设计**：以 `output_formats` 声明能力矩阵，使上层在批量调度前即可裁剪任务，避免运行时才失败。

### 5.3 批处理流水线

```
输入收集 → 文件发现(递归/扩展名过滤) → 能力矩阵裁剪 → 设备与并发度计算
→ [干跑预览并退出] → 进程池分发 → worker 内懒加载模型 → 结果收集
→ 报告生成 → 统计与退出码
```

### 5.4 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 输出格式默认值 | OpenEXR half-float 线性 Rec.709 | 开放、通用、可被后续合成流程消费；`--format` 可覆盖 |
| 默认执行模式 | 干跑预览，需 `--apply` 才写盘 | 与 PhoUtils "先看清楚再动手"原则一致 |
| EXIF 策略 | 方案 A+C：原生注入 + 旁车兜底 | 各格式注入能力不一，旁车保证零丢失 |
| 模型管理 | 自动下载至 `models/` 分类子目录，支持 HF 镜像 | 首次下载后续离线复用，批量任务稳定 |
| 并发模型 | 进程池 + worker 内懒加载模型 | 避免主进程加载后无法被 worker 共享 |
| 内存策略 | 按后端类型估算单进程占用，收敛 worker 数 | 防止 OOM 批量失败（历史教训：unet-onnx 被 OOM killer 杀死） |
| 指标自研 | VMAF/PU21/∆ICtCp 自行实现 | 符合 PhoUtils "薄封装、重工程"风格，避免引入外部指标库依赖 |
| UltraHDR 首选后端 | pyultrahdr 而非 libultrahdr | 零编译依赖即可覆盖 UltraHDR 输出；libultrahdr 作为合规性基准补充 |

---

## 六、实施计划与里程碑

| 阶段 | 内容 | 主要交付物 | 状态 |
|---|---|---|---|
| **P0 立项** | 需求分析、候选项目评估、执行方案 | 立项书、执行方案 | ✅ |
| **Phase 1 基础框架** | CLI 解析、文件发现、路径生成、后端基类与注册表、模型目录、并发框架、日志/报告、干跑 | `photo_sdr_to_hdr.py` 骨架、`utils/*` | ✅ |
| **Phase 2 核心后端** | hdrconv、unet-onnx、pyultrahdr、子进程基类（sdr2hdr/ultrahdr）、统一输出格式层 | `backends/*` | ✅ |
| **Phase 3 高级后端** | LumiPic 三变体、GMNet、ultrahdr 封装、EXIF A+C、色彩空间转换 | `backends/lumipic_backend.py`、`gmnet_backend.py` | ✅ |
| **Phase 4 打磨与文档** | 多模型对比、设备与资源预估、综合基准、质量指标模块、README | `bench_photo_sdr_to_hdr.py`、`utils/hdr_metrics.py` | ✅ |
| **P5 完整验收测试** | 解锁依赖、全量重跑、真实照片回归、压测、主观评估、合规校验 | 正式验收报告 | ⬜ **待启动** |
| **P6 验收与交付** | 验收会、签署、发布 | 验收报告、发布说明 | ⬜ 待启动 |

**当前进度**：P0–P4 完成（开发 100%），P5 待启动。**项目尚未通过验收。**

---

## 七、验收标准与验收计划

> 本节为**前置定义**，用于指导后续验收测试；项目当前**未满足**下列标准。

### 7.1 验收环境要求（三阶段硬件划分）

验收环境按硬件能力分为三个阶段，测试内容逐级递进；每项测试只归属一个阶段。

| 阶段 | 硬件 | 架构 / 显存 | 承接后端 | 当前具备 |
|:--:|---|---|---|:--:|
| **阶段一 CPU** | 8 核 / 8GB | CPU only | hdrconv、pyultrahdr、sdr2hdr、ultrahdr、gmnet(CPU) | ✅ |
| **阶段二 T4** | NVIDIA T4 | Turing SM75 / 16GB GDDR6 | unet-onnx、lumi-klein4b、gmnet(GPU) | ⬜ |
| **阶段三 L40** | NVIDIA L40 | **Ada Lovelace SM89** / 48GB GDDR6 | lumi-qwen、lumi-klein9b | ⬜ |

**公共要求**：存储 ≥150GB（lumi 权重合计 ~79GB + 缓存）；HDR 显示器 1000 nits+（阶段三主观评估用）；阶段三需 ≥32GB 显存 GPU 环境。

> **硬件划分的技术依据（已确认）**：T4 = NVIDIA T4，**Turing 架构 SM75 / 16GB GDDR6**，**硬件层面不支持 bf16**（bf16 由 Ampere SM80 引入，Ada SM89 增强）。因此凡依赖 bf16 精度路径、所需显存 >16GB 的测试项，一律归属 L40 阶段执行；其余需 GPU 参与的测试组合归属 T4。L40 = NVIDIA L40，**Ada Lovelace SM89 / 48GB GDDR6**。

### 7.2 验收判定准则

| 类别 | 准则 | 目标值 | 当前状态 | 负责阶段 |
|---|---|---|:---:|:---:|
| 功能 | 立项 CLI 参数落地率 | 100%（10/10） | ✅ 已满足 | CPU |
| 功能 | 全后端端到端可用率 | 9/9 后端名可运行 | ❌ 4/9 | CPU / T4 / L40 |
| 功能 | 批量与递归 | 单次运行 ≥1000 张无中断 | ❌ 未测 | CPU |
| 质量 | bench 失败项 | 0 | ✅ 已满足（71/0/22） | 三阶段累计 |
| 质量 | 跳过项性质 | 100% 能力边界，无环境阻塞遗留 | ❌ 有 8 项环境阻塞 | 三阶段累计 |
| 质量 | 真实 HDR | 峰值 >250 nits、DR ≥2 stops | ✅ 已满足（3 后端） | CPU / T4 / L40 |
| 质量 | 确定性 | 同阶段两次转换字节一致 | ✅ 已满足 | CPU |
| 质量 | EXIF 保留 | keep/no-keep × 格式矩阵无泄漏 | ✅ 已满足 | CPU |
| 质量 | 真实照片回归 | ≥20 张真实照片质量基线 | ❌ 未测 | CPU |
| 质量 | 量化档位单调性 | none ≥ 8bit ≥ 4bit（PSNR 不倒挂） | ❌ 未测 | **L40** |
| 质量 | 跨阶段一致性 | CPU↔T4 rt_SSIM ≥0.95 且差异有归因 | ❌ 未测 | T4 / L40 |
| 性能 | 并行加速比 | 千张级 ≥1.5× | ❌ 未测 | CPU / L40 |
| 性能 | 内存峰值 | ≤可用内存 90%，无 OOM | 🔶 仅小样本 | CPU |
| 性能 | 显存峰值 | ≤显存预算，单卡多 worker 不 OOM | ❌ 未测 | T4 / L40 |
| 合规 | UltraHDR 段结构 | 与 Google 参考样本一致 | 🔶 仅字节级内部验证 | CPU |
| 体验 | HDR 显示器主观评估 | UltraHDR / Gainmap 输出可视可接受 | ❌ 未测 | **L40** |
| 文档 | 复现性 | 干净环境复现全部命令成功 | 🔶 未做干净环境验证 | CPU |

### 7.3 验收测试计划（三阶段递进）

#### 阶段一：CPU —— 测试准备 + 全量 CPU 实测

本阶段承担**全部测试准备**，使后续阶段无需重复准备素材。

| 类别 | 内容 |
|---|---|
| **准备** | 真实照片集 ≥20 张（夜景/人像/高动态/纹理/低饱和，含完整 EXIF 样本）｜分辨率矩阵（512×384/1K/2K/2160×3840/8K）｜编译 `ultrahdr_app`（CMake+Ninja）｜bench 扩展（新增真实照片组、分辨率维度、跨阶段比对接口）｜Google UltraHDR 参考样本 |
| **实测** | `ultrahdr` 后端首次端到端｜CPU 全量基准（9 后端）｜真实照片质量基线（I/J 组）｜千张级压测（加速比 + 峰值内存 + OOM 恢复）｜UltraHDR 合规性交叉校验｜统计口径统一 |
| **通过条件** | FAIL = 0；真实照片基线表齐全；压测报告在案；测试集版本冻结；**不得遗留任何环境阻塞类跳过项** |

#### 阶段二：T4 —— GPU 通用通路（不含 Ada 专属项）

| 类别 | 内容 |
|---|---|
| **准备** | `onnxruntime-gpu` + CUDA 版 torch，验证 `CUDAExecutionProvider` 挂载｜下载 `lumi-klein4b`（~8GB）｜各后端显存基线标定｜OOM 复现用例 |
| **实测** | `--device cuda` 设备解析与 providers 挂载｜`unet-onnx` A–K 全组（fp32/fp16）｜`lumi-klein4b` A–K 全组（**4bit / 8bit 路径**）｜`gmnet` CUDA 推理与指标比对｜GPU 批量吞吐与显存峰值｜worker 数收敛验证｜CPU↔T4 一致性比对 |
| **通过条件** | unet-onnx / lumi-klein4b 无 FAIL；CUDA 通路验证通过；GPU 吞吐与显存记录在案；一致性比对有差异归因；跳过项为能力边界或明确标注「bf16 待 L40」 |

#### 阶段三：L40 —— Ada 专属测试项（不重复前序验证）

| 类别 | 内容 |
|---|---|
| **准备** | 下载 `lumi-qwen`（~54GB）、`lumi-klein9b`（~17GB，需授权）｜确认 `torch.cuda.is_bf16_supported()`｜主观评估物料与评分表 |
| **实测** | `lumi-qwen` / `lumi-klein9b` A–K 全组（bf16 主路径）｜量化矩阵全档对照（none/8bit/4bit）｜大显存单卡多 worker 并发压测｜跨阶段一致性比对与差异归因｜HDR 显示器主观评估｜前两阶段已通过项的回归抽查 |
| **通过条件** | 9 后端名全部端到端无 FAIL；Ada 专属项全部完成并有记录；量化矩阵三维数据齐全；主观评估确认；**环境阻塞跳过项归零** |

### 7.4 跨阶段一致性比对规则

| 比对类型 | 判定标准 | 负责阶段 |
|---|---|:---:|
| 阶段内确定性 | 输出字节完全一致 | CPU |
| 同硬件不同后端 | 核心指标差异 ≤5%，段结构一致 | CPU |
| CPU ↔ T4 | rt_SSIM ≥0.95、rt_PSNR ≥35dB，差异归因于算子/精度 | T4 |
| 同模型跨卡（4bit 档） | 差异 ≤容差；bf16 档为质量上限基线 | L40 |
| 量化档位单调性 | none ≥ 8bit ≥ 4bit（PSNR 不倒挂） | **L40** |

### 7.5 验收组织

| 角色 | 职责 | 建议人选 |
|---|---|---|
| 项目负责人 | 交付范围、进度、验收组织 | 待指派 |
| 技术负责人 | 架构与实现质量、后端选型 | 待指派 |
| 测试负责人 | 测试设计、bench 执行、口径裁定 | 待指派 |
| 补测执行 | 依赖解锁、重跑、回归 | 待指派 |
| 验收方 | 准则裁定、验收签署 | 待指派 |

---

## 八、质量保证策略

### 8.1 测试分层

| 层级 | 内容 | 工具 | 门禁 |
|---|---|---|---|
| 单元测试 | PU21 映射点对照 MATLAB 参考、VMAF 区分度、SSIM/∆ICtCp 边界、EXIF IFD 偏移、EXR/PNG 结构 | 手工校验 + 自校准脚本 | 指标实现变更必测 |
| 集成测试 | 后端端到端、CLI 全参数组合、报告生成 | `bench_photo_sdr_to_hdr.py` | 每次提交全量跑 |
| 回归测试 | 固定种子输出字节一致性、质量指标基线 | bench H 组 + 基线表 | 每次提交 |
| 压力测试 | 千张级批量、内存压力、显存溢出恢复 | 专用脚本 | 验收阶段 |
| 合规测试 | UltraHDR / ISO 21496-1 段结构与参考样本比对 | 参考样本 | 验收阶段 |

### 8.2 测试维度（bench A–K 共 11 组）

| 组 | 维度 | 判定标准 |
|---|---|---|
| A | 后端可用性矩阵 | 不可用记 SKIP + 原因（环境因素不算失败） |
| B | 输出格式矩阵 | EXR 可读且峰值 >250 nits；JPEG 有 MPF/XMP；PNG 16bit |
| C | 增益图方法 | auto/linear/log/fixed 生效，采集峰值与分域提升 |
| D | 目标峰值亮度 | 250 < 峰值 ≤ 目标×1.10，记录截断率 |
| E | EXIF 保留矩阵 | JPEG EXIF APP1 / EXR header / PNG eXIf；no-keep 无泄漏 |
| F | 并行扩展 | workers 1/4/auto 退出码 0 且输出数正确 |
| G | 干跑安全 | 退出码 0 且输出目录不存在（零文件写入） |
| H | 确定性 | 同输入两次转换输出字节完全相同 |
| I | 无参考客观质量 | rt-VMAF / rt-SSIM / rt-PSNR / DR / base 保真 |
| J | 全参考客观质量 | ∆ICtCp PSNR(BT.2124) + PU21 PSNR/SSIM |
| K | 资源占用 | 子进程峰值 RSS（ru_maxrss） |

### 8.3 缺陷分级

| 级别 | 定义 | 处理 |
|:--:|---|---|
| P0 致命 | 无法运行、数据损坏、批量中断 | 立即修复，阻断发布 |
| P1 严重 | 主流程功能失效、后端完全不可用 | 验收前必须修复 |
| P2 一般 | 边界条件、指标偏差、降级路径异常 | 计划内修复，可带条件验收 |
| P3 轻微 | 文案、命名、次要优化 | 记录待办 |

---

## 九、资源与依赖

### 9.1 核心依赖（强制）

```
Python >= 3.10
Pillow >= 10.0
numpy >= 1.24
tqdm >= 4.66        # 进度条
```

### 9.2 可选依赖（按后端懒加载）

| 后端 | 依赖 |
|---|---|
| hdrconv | `hdr-conversion` |
| unet-onnx | `onnxruntime` / `onnxruntime-gpu` |
| lumi-* | `torch`、`diffusers`、`transformers`、`peft`、`accelerate`、`safetensors`（4bit 需 `bitsandbytes`） |
| gmnet | 免 BasicSR（架构 vendor + 检查点下载） |
| pyultrahdr | 仅标准库 + Pillow |
| sdr2hdr | Rust CLI（`cargo install sdr2hdr`） |
| ultrahdr | 自行编译的 `ultrahdr_app` |

### 9.3 外部模型资产

| 后端 | 资产 | 大小 | 许可注意 |
|---|---|---|---|
| unet-onnx | `unet4k.onnx` | 7.4MB | 已就位 |
| gmnet | 预训练检查点 | 小 | 已就位 |
| lumi-klein4b | FLUX.2-klein-4B | ~8GB | Apache 2.0，**最轻，建议优先** |
| lumi-klein9b | FLUX.2-klein-9B | ~17GB | 需授权 |
| lumi-qwen | Qwen-Image-Edit-2511 | ~54GB | bf16 推理需 ~24GB 显存 |

### 9.4 硬件资源估算

| 后端 | 单进程基础内存 | 单张推理峰值 | 备注 |
|---|---|---|---|
| pyultrahdr | ~100 MB | +30 MB | 纯 Python，极轻量 |
| hdrconv | ~200 MB | +50 MB | 纯 CPU |
| sdr2hdr | ~50 MB | +20 MB | 子进程调用 |
| gmnet | ~2–4 GB | +1–2 GB | 显存/内存双占 |
| unet-onnx | ~500 MB（模型） | +数 GB（激活） | 2160×3840 固定输入，**OOM 高风险** |
| lumi-qwen | ~8–12 GB（bf16） | +4–8 GB | 建议 4bit 量化 |

---

## 十、风险分析与应对

| 编号 | 风险 | 类别 | 等级 | 概率 | 影响 | 应对措施 | 责任 |
|:--:|---|---|:--:|:--:|:--:|---|---|
| R1 | `unet-onnx` 在低内存机器被 OOM killer 杀死（SIGKILL 不可捕获） | 技术 | **高** | 高 | 高 | 已加 ≥5GB 可用内存预检，转为明确报错；GPU 机优先 | 开发 |
| R2 | LumiPic 权重合计 ~79GB，下载与存储成本高；部分需授权 | 资源 | **高** | 高 | 中 | 优先验证最轻的 `lumi-klein4b`；HF 镜像加速；提供 4bit 量化 | 测试 |
| R3 | `ultrahdr` 需 C++ 编译，CI 环境缺失导致该后端长期零验证 | 技术 | 中 | 中 | 中 | pyultrahdr 已覆盖同等输出能力作兜底；编译纳入验收环境 | 开发 |
| R4 | 合成测试图与真实照片统计特性差距，指标基线失真 | 质量 | **高** | 高 | 高 | 已列入验收阻断项 B3，须补 ≥20 张真实照片回归 | 测试 |
| R5 | 客观指标自研实现偏差（VMAF/PU21/∆ICtCp） | 质量 | 中 | 中 | 中 | 已用 MATLAB 参考点与区分度实验自校准；持续维护 | 开发 |
| R6 | VMAF 对平滑色调不敏感（rt_VMAF=100 失真） | 质量 | 中 | 已发生 | 中 | 已文档化边界；强制结合 SSIM/DR 解读 | 测试 |
| R7 | 小批量下进程池启动开销 > 收益，并行收益不可见 | 性能 | 中 | 已发生 | 低 | 已解释；千张级压测量化后再定默认策略 | 开发 |
| R8 | 合成 GT 基于共享 ITM 曲线，对学习型方法天然不利（∆ICtCp 43.4dB） | 质量 | 中 | 已发生 | 低 | 已在报告中说明属预期内差异；需引入真实 GT 或主观评估 | 测试 |
| R9 | `sdr2hdr` 不产线性 HDR，I/J 组不适配，易被误判为缺陷 | 质量 | 低 | 已发生 | 低 | 已在测试覆盖报告中明确标注为能力边界 | 测试 |
| R10 | 测试覆盖统计口径不一致（跳过项分解口径存疑） | 管理 | 中 | 已发生 | 低 | 列入验收前置项 B7，验收前统一裁定 | 测试负责人 |
| R11 | 学习库版本冲突（diffusers / peft / basicsr 互斥） | 技术 | 中 | 中 | 中 | 按后端懒加载隔离；建议独立虚拟环境 | 开发 |
| R12 | HDR 显示设备主观效果无法用客观指标完全代表 | 质量 | 中 | 高 | 中 | 验收含主观评估环节，需专用显示器 | 验收方 |
| **R13** | **T4（Turing SM75）不支持 bf16**，`lumi-*` 默认 `--quantization none` 在阶段二必然失败，易被误判为代码缺陷 | 技术 | **高** | 高 | 中 | 阶段二强制显式传 `--quantization 4bit/8bit`；建议在 lumi 后端补 bf16 可用性预检（参照 unet-onnx 内存预检），使报错可读 | 开发 |
| **R14** | T4 显存 16GB，`lumi-klein9b`（~17GB）无法加载，阶段归属误配会导致阶段二无法准出 | 资源 | 中 | 中 | 中 | 已明确 klein9b/qwen 归 L40 阶段；阶段二准备项含"记录该项为硬件能力边界" | 测试 |
| **R15** | 三阶段测试集/基线版本不一致，跨阶段比对失效 | 管理 | 中 | 中 | 中 | 每阶段准出即冻结测试集清单 + SHA256 + 参数版本（见《初验报告》§7.1 原则） | 测试负责人 |

---

## 十一、变更控制

| 变更类型 | 审批要求 | 文档要求 |
|---|---|---|
| 范围变更（增删后端 / 增删输出格式） | 项目负责人 + 技术负责人 | 更新本立项书 §3、§4 |
| 架构变更（后端接口、流水线结构） | 技术负责人 | 更新执行方案 |
| 验收准则变更 | 验收方 | 更新本立项书 §7.2 |
| 依赖基线升级（Python / Pillow / numpy 主版本） | 技术负责人 | 更新 README 环境要求 |
| 已知缺口转为遗留项 | 测试负责人 | 更新《实施及初验报告》 |

变更须记录：变更内容、原因、影响范围、验证方式、生效版本。

---

## 十二、项目里程碑与交付节点

| 节点 | 交付物 | 状态 |
|---|---|:--:|
| M1 立项完成 | 立项书 | ✅ |
| M2 方案完成 | 执行方案（`IMPLEMENTATION_PLAN.md`） | ✅ |
| M3 框架完成 | CLI + 调度 + 报告 | ✅ |
| M4 后端完成 | 8 个后端实现 | ✅ 代码完成（4 个未验证） |
| M5 度量与测试完成 | 指标模块 + 综合基准 | ✅ |
| **M6 初验完成** | **实施及初验报告** | ✅ **已出具**（结论：非验收通过，转入完整验收） |
| **M7a 阶段一 CPU** | 测试准备包、CPU 全量 bench、真实照片基线、压测报告、合规比对 | ⬜ **待启动** |
| **M7b 阶段二 T4** | GPU 通路报告、unet-onnx / lumi-klein4b 端到端、一致性比对 | ⬜ **待启动** |
| **M7c 阶段三 L40** | Ada 专属项报告、lumi-qwen / lumi-klein9b 端到端、量化矩阵、主观评估 | ⬜ **待启动** |
| **M8 验收通过** | 正式验收报告 + 签署 | ⬜ 待启动 |

---

## 十三、签署

| 角色 | 姓名 | 日期 | 签字 | 意见 |
|---|---|---|---|---|
| 编制人 | | | | |
| 项目负责人 | | | | |
| 技术负责人 | | | | |
| 测试负责人 | | | | |
| 资源审批 | | | | |
| 验收方 | | | | |

---

## 附录 A：立项单要素摘要

| 要素 | 内容 |
|---|---|
| 项目名称 | 图像 SDR 转 HDR 批量转换工具 |
| 项目编号 | PhoUtils-SDR2HDR-001 |
| 项目目标 | 交付统一 SDR→HDR 批量转换 CLI，封装 ≥8 个后端名 / ≥7 个实现 |
| 交付物 | `photo_sdr_to_hdr.py`、`backends/`(8)、`utils/`(10)、`bench_photo_sdr_to_hdr.py`、4 份文档 |
| 核心约束 | Python ≥3.10；核心依赖仅 Pillow+numpy；重依赖懒加载；默认干跑不写盘 |
| 验收原则 | 真实 HDR（峰值>250 nits、DR≥2 stops）+ 零失败 + 可复现 + 优雅降级 |
| 验收环境 | **三阶段递进**：CPU（准备+全量 CPU 实测）→ T4（GPU 通用通路）→ L40（Ada 专属项） |
| 当前状态 | 开发完成，**尚未验收通过**；7 项验收前置条件待关闭，分属三个验收阶段 |
| 下一节点 | M7 三阶段完整验收测试：M7a CPU → M7b T4 → M7c L40（详见《实施及初验报告》§5、§7） |

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| SDR | Standard Dynamic Range，标准动态范围（8bit） |
| HDR | High Dynamic Range，高动态范围 |
| Ultra HDR | Google 提出的 JPEG/R 扩展格式，JPEG 内嵌增益图，Android 14+ 原生支持 |
| Gain Map | 增益图，描述 SDR 像素到 HDR 亮度的映射关系 |
| ISO 21496-1 | 增益图 JPEG 国际标准 |
| EXR | OpenEXR，线性 HDR 中间格式，本项目用 half-float |
| ITM | Inverse Tone Mapping，逆色调映射（解析式 SDR→HDR 路线） |
| MPF | Multi-Picture Format，JPEG 内多图容器段，UltraHDR 必需 |
| VMAF / PU21 | 视频/图像质量感知客观指标；PU21 为 HDR 亮度感知指标 |
| ∆ICtCp | ITU-R BT.2124 色度差分，用于 HDR 色度保真评价 |
| PQ / HLG | 高动态范围传递函数（Perceptual Quantizer / Hybrid Log-Gamma） |
| Gainmap Method | 增益图计算方式：auto / linear / log / fixed |
| 能力边界跳过 | 后端本就不支持某能力，测试记 SKIP 而非 FAIL |
| 环境阻塞跳过 | 因缺依赖/模型/硬件而无法执行，记 SKIP 并附解锁路径 |