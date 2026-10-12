# photo_sdr_to_hdr.py 实现方案

## 📋 项目概述

基于 PhoUtils 的设计风格，创建一个统一的 **SDR → HDR 批量转换 CLI 工具**，集成多个开源项目的模型/后端，支持文件/文件夹批量处理，遵循"先预览、再执行、可回退"的工程原则。

---

## 🎯 核心需求

### CLI 参数规格（必须包含）

| 参数 | 短选项 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `--input` | `-i` | str (可多个) | **必填** | 输入文件或目录路径，支持多个 |
| `--output` | `-o` | str | 自动生成 | 输出目录或文件（单文件时） |
| `--models` | | str (可多个) | `hdrconv` | 模型后端选择，可指定多个并行对比 |
| `--workers` | `--threads` | int | 0 (自动) | 并行进程数，0=按核数/内存自动 |
| `--device` | | choice | `auto` | 设备：`auto`/`cuda`/`cpu`/`mps` |
| `--suffix` | | str | `_hdr` | 输出文件后缀 |
| `--recursive` | `-r` | flag | False | 递归处理子目录 |
| `--dry-run` | | flag | True | 预览模式（默认），不写文件 |
| `--apply` | | flag | False | 真正执行转换 |
| `--log` | | str | `-` | 日志文件路径，`-` 为 stdout |
| `--help` | `-h` | flag | - | 显示帮助 |

### 模型后端标识符（`--models` 可选值）

| 标识符 | 后端类型 | 说明 | 依赖 |
|--------|----------|------|------|
| `hdrconv` | Python原生 | hdr-conversion 库，格式转换/封装 | `pip install hdr-conversion` |
| `unet-onnx` | ONNX Runtime | SDR-to-HDR U-Net (unet4k.onnx) | `pip install onnxruntime` |
| `lumi-qwen` | PyTorch/Diffusers | LumiPic Qwen-Image-Edit LoRA | `pip install diffusers transformers peft` |
| `lumi-klein4b` | PyTorch/Diffusers | LumiPic FLUX.2-klein-4B LoRA | 同上 |
| `lumi-klein9b` | PyTorch/Diffusers | LumiPic FLUX.2-klein-9B LoRA (需授权) | 同上 |
| `gmnet` | PyTorch/BasicSR | GMNet ICLR2025 Gain Map 预测 | `pip install basicsr` |
| `sdr2hdr` | 子进程 | Rust CLI 工具 (ICC 嵌入) | `cargo install sdr2hdr` |
| `pyultrahdr` | Python原生 | 纯 Python UltraHDR 编码 (~230行，无需编译) | `pip install pillow` (仅标准库+Pillow) |
| `ultrahdr` | 子进程 | libultrahdr CLI (需自行编译) | 需编译 `ultrahdr_app` |

---

## 🏗️ 架构设计

### 1. 后端抽象接口

```python
# 所有后端实现统一接口
class SDRToHDRBackend:
    name: str                    # 后端标识符
    description: str             # 简要说明
    required_deps: List[str]     # 所需依赖包
    supports_batch: bool         # 是否支持批量推理
    output_formats: List[str]    # 支持的输出格式
    
    def check_available(self) -> Tuple[bool, str]:
        """检查后端是否可用，返回 (可用, 原因)"""
    
    def prepare(self, device: str, **kwargs) -> None:
        """初始化模型/环境，device 为 cuda/cpu/mps"""
    
    def convert(self, input_path: str, output_path: str, **kwargs) -> bool:
        """单张转换，返回成功与否"""
    
    def convert_batch(self, input_paths: List[str], output_dir: str, **kwargs) -> List[Tuple[str, bool]]:
        """批量转换（可选实现，默认串行调用 convert）"""
```

### 2. 后端实现策略

#### A. hdrconv 后端 (`backends/hdrconv_backend.py`)
- **功能**：UltraHDR / Adaptive Gainmap / PQ-HLG 格式封装
- **输入**：SDR 图片 (JPEG/PNG)
- **输出**：UltraHDR JPEG / Gainmap JPEG / AVIF HDR
- **核心 API**：`hdrconv.read()`, `hdrconv.write()`, `hdrconv.convert()`
- **注意**：项目说明"色彩转换可能不正确"，alpha 阶段

#### B. unet-onnx 后端 (`backends/unet_onnx_backend.py`)
- **模型**：`Model_onnx/unet4k.onnx` (需下载)
- **推理**：onnxruntime (CPU/CUDA)
- **预处理**：Resize 到 4096x4096? 归一化到 [0,1]
- **后处理**：色彩变换 + 线性叠加 (参考项目 color-transform)
- **输出**：EXR / PNG (16bit) / JPEG (需色调映射)

#### C. LumiPic 系列后端 (`backends/lumipic_backend.py`)
- **统一入口**：`inference.py` / `inference_klein.py`
- **模型**：自动下载 LoRA 权重 (HuggingFace `oumoumad/LumiPic`)
- **依赖**：diffusers, transformers, peft, torch
- **显存需求**：24-48GB (可量化)
- **输出**：OpenEXR (.exr, half-float, LogC3 编码)

#### D. GMNet 后端 (`backends/gmnet_backend.py`)
- **框架**：BasicSR (PyTorch)
- **模型**：需从 GitHub releases 下载预训练权重
- **推理**：预测 Gain Map + Qmax → 重建 HDR
- **脚本复用**：`scripts/render_sdr_gm_to_linear_hdr.py`

#### E. 子进程后端基类 (`backends/subprocess_backend.py`)
- 统一管理外部 CLI 工具调用
- 超时控制、错误捕获、进度回调
- 实现 `sdr2hdr` 和 `ultrahdr` 具体子类

#### F. pyultrahdr 后端 (`backends/pyultrahdr_backend.py`)
- **来源**：https://github.com/hanfeisun/pyultrahdr (纯 Python ~230行)
- **功能**：将任意 SDR 图像转换为字节兼容的 Ultra HDR JPEG
- **优势**：无需编译 libultrahdr，零 C++ 依赖，极易集成
- **核心逻辑**：读取 SDR 图像 → 计算简单增益图 → 写入 Ultra HDR JPEG 容器格式
- **输出**：Ultra HDR JPEG (.jpg)，兼容 Android/Chrome 等 HDR 显示器
- **局限**：非深度学习，增益图基于简单启发式，效果不如学习型方法

### 3. 批量处理流水线

```
输入收集 → 文件发现(递归/扩展名过滤) → 分组(按后端) → 并行推理 → 结果收集 → 写入/报告
```

**并发策略**（沿用 photo_dedup.py）：
- 自动探测 CPU 核数 + 可用内存
- 单进程显存/内存预估（模型加载 + 推理峰值）
- `workers = min(核数, 内存预算/单进程预估, 任务数)`
- 进程池 + 队列，失败重试/跳过不中断整批

### 4. 设备管理

```python
def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return requested  # cuda/cpu/mps 直传
```

- ONNX Runtime: `providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']`
- PyTorch: `.to(device)`
- 子进程: 通过环境变量或参数传递

---

## 📁 文件结构规划

```
PhoUtils/
├── photo_sdr_to_hdr.py          # 主入口脚本
├── backends/
│   ├── __init__.py
│   ├── base.py                  # 后端抽象基类
│   ├── hdrconv_backend.py
│   ├── unet_onnx_backend.py
│   ├── lumipic_backend.py
│   ├── gmnet_backend.py
│   └── subprocess_backend.py    # 子进程基类 + sdr2hdr/ultrahdr 实现
├── utils/
│   ├── __init__.py
│   ├── file_io.py               # 文件发现、扩展名、输出路径生成
│   ├── device.py                # 设备解析、显存探测
│   ├── parallel.py              # 进程池封装、任务分发
│   ├── logging.py               # 统一日志、进度条
│   └── report.py                # 运行报告生成
├── models/                      # 模型文件存放目录 (gitignore)
│   ├── unet4k.onnx
│   └── gmnet_pretrained.pth
└── tests/
    ├── test_backends.py
    └── test_cli.py
```

---

## 🔄 详细执行流程

### 主流程 `main()`

1. **参数解析** → 验证输入输出路径、模型后端可用性
2. **文件发现** → `collect_input_files(inputs, recursive, exts)` 返回有序文件列表
3. **后端初始化** → 对每个 `--models` 实例化后端，`check_available()` 过滤不可用
4. **设备解析** → `resolve_device(args.device)` 统一设备字符串
5. **并发度计算** → `resolve_workers(n_files, mem_per_worker, args.workers)`
6. **干跑预览** → 打印将处理的文件、后端、输出路径、预估资源
7. **执行转换** (仅 `--apply`)：
   - 创建输出目录
   - 进程池分发任务
   - 实时进度显示
   - 收集成功/失败统计
8. **生成报告** → 写入 `photo_sdr_to_hdr_report.txt`
9. **清理/总结** → 打印统计、耗时、失败文件列表

### 文件发现逻辑

```python
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
                  ".heic", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2")

def collect_input_files(paths: List[str], recursive: bool) -> List[str]:
    files = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(SUPPORTED_EXTS):
            files.append(p)
        elif os.path.isdir(p):
            walk = os.walk(p) if recursive else [(p, [], os.listdir(p))]
            for dirpath, _, filenames in walk:
                for f in sorted(filenames):
                    if f.lower().endswith(SUPPORTED_EXTS):
                        files.append(os.path.join(dirpath, f))
    return sorted(set(files))  # 去重保序
```

### 输出路径生成

```python
def make_output_path(input_path: str, output_root: str, suffix: str, 
                     input_root: str, ext_override: str = None) -> str:
    # 保持相对目录结构
    rel = os.path.relpath(input_path, input_root)
    stem, ext = os.path.splitext(rel)
    new_ext = ext_override or ext
    out_rel = f"{stem}{suffix}{new_ext}"
    return os.path.join(output_root, out_rel)
```

---

## ⚙️ 关键技术细节

### 1. 内存/显存预估

| 后端 | 单进程基础内存 | 单张推理峰值 | 备注 |
|------|----------------|--------------|------|
| hdrconv | ~200 MB | +50 MB | 纯 CPU，轻量 |
| unet-onnx | ~500 MB (模型) | +200 MB | ONNX Runtime 会预分配 |
| lumi-qwen | ~8-12 GB (bf16) | +4-8 GB | 需量化选项 |
| gmnet | ~2-4 GB | +1-2 GB | BasicSR 框架 |
| pyultrahdr | ~100 MB | +30 MB | 纯 Python，极轻量 |
| sdr2hdr | ~50 MB | +20 MB | 子进程调用 Rust CLI |

**策略**：`--mem-per-worker` 手动指定，或按模型类型自动估算。

### 2. 进程池隔离

- **模型加载在主进程**？不行，显存不共享
- **方案**：每个 worker 进程内懒加载模型（首次任务时加载）
- 使用 `multiprocessing` + `initializer` 传递配置，worker 内全局缓存模型实例

```python
# worker 内全局缓存
_WORKER_MODEL_CACHE = {}

def worker_init(backend_name, device, model_kwargs):
    global _WORKER_MODEL_CACHE
    backend = create_backend(backend_name)
    backend.prepare(device, **model_kwargs)
    _WORKER_MODEL_CACHE[backend_name] = backend

def worker_convert(job):
    input_path, output_path, backend_name, kwargs = job
    backend = _WORKER_MODEL_CACHE[backend_name]
    return backend.convert(input_path, output_path, **kwargs)
```

### 3. 多模型对比模式

当 `--models` 指定多个后端时：
- 每个输入文件对每个后端都生成一个输出
- 输出文件名包含后端标识：`photo_hdr_unet-onnx.exr`, `photo_hdr_lumi-qwen.exr`
- 生成对比报告，便于评估效果

---

## 📝 日志与报告设计

### 控制台输出示例

```
$ python photo_sdr_to_hdr.py -i ./photos -o ./hdr_output --models unet-onnx lumi-qwen --apply

目录：./photos (递归: 是)
扫描到 42 张图片
后端：unet-onnx (ONNX Runtime CUDA), lumi-qwen (Diffusers CUDA, 4-bit量化)
设备：cuda (NVIDIA RTX 3080, 10GB)
并行：4 进程（核数 8，内存上限 6，单进程预估 2.5 GB）
输出：./hdr_output/ (后缀 _hdr, 格式 .exr)
动作：执行转换

[unet-onnx] 初始化模型... 完成 (1.2s)
[lumi-qwen] 初始化模型... 完成 (8.5s, 量化加载)

转换进度：████████████████████ 42/42 (100%)
  unet-onnx: 42 成功, 0 失败
  lumi-qwen: 40 成功, 2 失败 (显存不足)

统计：共 84 任务，成功 82，失败 2，耗时 145.3s
报告：photo_sdr_to_hdr_report.txt
```

### 报告文件格式

```
photo_sdr_to_hdr Report - 2026-10-12 14:30:15
===========================================
输入目录: /mnt/d/Photos
输出目录: /mnt/d/Photos/hdr_output
后端: unet-onnx, lumi-qwen
设备: cuda
并行进程: 4
后缀: _hdr
递归: 是

文件列表 (42):
  1. IMG_0001.jpg -> IMG_0001_hdr_unet-onnx.exr ✓ | IMG_0001_hdr_lumi-qwen.exr ✓
  2. IMG_0002.jpg -> IMG_0002_hdr_unet-onnx.exr ✓ | IMG_0002_hdr_lumi-qwen.exr ✓
  ...

失败详情:
  IMG_0035.jpg (lumi-qwen): CUDA out of memory
  IMG_0036.jpg (lumi-qwen): CUDA out of memory

统计: 总任务 84, 成功 82, 失败 2, 总耗时 145.3s
```

---

## 🛡️ 安全性与鲁棒性

1. **默认干跑**：无 `--apply` 不写任何文件
2. **输出不覆盖**：同名文件自动加 `_1`, `_2` 后缀
3. **单文件失败不中断**：记录错误继续处理下一个
4. **模型下载缓存**：HuggingFace cache 复用，离线可用
5. **信号处理**：`SIGINT` 优雅关闭进程池，保存已完成结果
6. **磁盘空间检查**：预估输出大小，空间不足时预警

---

## 🧪 测试策略

| 测试层级 | 内容 | 工具 |
|----------|------|------|
| 单元测试 | 后端接口、文件发现、路径生成 | pytest |
| 集成测试 | 端到端 CLI 调用、多后端并行 | pytest + 临时目录 |
| 回归测试 | 固定种子图片的输出一致性 | 校验哈希/指标 |
| 压力测试 | 大批量(1000+)、内存压力、显存溢出恢复 | 手动/脚本 |

---

## 📦 依赖管理

### 核心依赖（必须）
```
Pillow >= 10.0
numpy >= 1.24
tqdm >= 4.66  # 进度条
```

### 可选依赖（按后端懒加载）
```
# hdrconv
hdr-conversion >= 0.1.6

# unet-onnx
onnxruntime >= 1.18  # 或 onnxruntime-gpu

# lumipic 系列
torch >= 2.3
diffusers >= 0.30
transformers >= 4.40
peft >= 0.11
accelerate >= 0.30
safetensors >= 0.4

# gmnet
basicsr >= 1.4  # 需从源码安装或特定版本

# 子进程后端无额外 Python 依赖
```

**安装建议**：提供 `requirements.txt` (核心) + `requirements-full.txt` (所有后端)

---

## 🚀 实施里程碑（已全部完成）

### Phase 1: 基础框架 ✅
- [x] CLI 参数解析（含新增 `--format`/`--quantization`/`--colorspace`/`--model-dir`/`--keep-exif`/`--gain-map-method`/`--target-nits`/`--sdr2hdr-mode`/`--ultrahdr-binary`）
- [x] 文件发现、输出路径生成（支持格式后缀覆盖，ultrahdr/gainmap → .jpg）
- [x] 后端抽象基类、注册机制（`BackendRegistry` 工厂）
- [x] 模型目录管理（自动创建、分类下载、完整性校验）
- [x] 并发框架（复用 photo_dedup.py 的 resolve_workers/进程池，worker 内模型隔离）
- [x] 日志/报告系统
- [x] 干跑模式完整走通

### Phase 2: 核心后端实现 ✅
- [x] hdrconv 后端（hdr-conversion 库，ISO 21496-1/UltraHDR 规范封装）
- [x] unet-onnx 后端（unet4k.onnx 自动下载，内存预检，CUDA 可选）
- [x] pyultrahdr 后端（纯 Python，零依赖，Ultra HDR JPEG 输出 + EXIF 注入）
- [x] 子进程后端基类 + sdr2hdr（ICC vendor）+ ultrahdr（CLI，优雅降级）
- [x] 统一输出格式层（EXR half-float / PNG16 / JPEG，内部线性 nits）

### Phase 3: 高级后端 ✅
- [x] LumiPic 三变体（qwen/klein4b/klein9b，量化选项，HF 镜像自动切换，懒加载）
- [x] GMNet 后端（架构 vendor + 检查点自动下载，无需 BasicSR）
- [x] ultrahdr 后端（libultrahdr CLI 封装）
- [x] EXIF 保留（A+C）与色彩空间转换

### Phase 4: 打磨与文档 ✅
- [x] 多模型对比模式（`--models a b c`，输出带后端名后缀）
- [x] 设备自动选择、内存/显存预估
- [x] `bench_photo_sdr_to_hdr.py` 综合基准（11 组 93 项）
- [x] 客观质量指标模块 `utils/hdr_metrics.py`（VMAF/PU21/∆ICtCp/SSIM/DR）
- [x] README 更新

### 实测结果（2026-10-12，8 逻辑核 / 7.7GB 内存 / CPU）

**Benchmark：71 通过 / 0 失败 / 22 跳过（环境原因）**

| 后端 | 状态 | 无参考质量（I 组） | 全参考质量（J 组，vs ITM GT） |
|------|------|--------------------|------------------------------|
| pyultrahdr | ✅ 可用 | rt_SSIM 0.921 / PSNR 20.1dB / VMAF 100 / DR 2.96 stops | ∆ICtCp 111dB / PU21 SSIM 1.000 |
| hdrconv | ✅ 可用 | 同上（同 ITM 曲线） | 同上 |
| gmnet | ✅ 可用 | rt_SSIM 0.950 / PSNR 22.5dB / VMAF 100 / DR 2.43 stops | ∆ICtCp 43.4dB / PU21 SSIM 0.967 |
| sdr2hdr | ✅ 可用 | （ICC 嵌入，无线性 HDR 输出） | — |
| unet-onnx | ⏭ 内存不足跳过 | （需 ≥5GB 可用内存或 CUDA） | — |
| lumi-qwen/klein4b/klein9b | ⏭ 模型未下载跳过 | （首次使用需下载基础模型） | — |
| ultrahdr | ⏭ 未编译跳过 | （需 ultrahdr_app 二进制） | — |

**性能参考**（512×384 测试图，CPU）：pyultrahdr/hdrconv ~20ms/张，gmnet ~0.2s/张（不含模型加载），sdr2hdr ~50ms/张；4 进程并行池对 4 张小图无加速（进程池开销 > 收益，与 photo_dedup 的结论一致）。

---

## ✅ 已确认决策点

1. **输出格式统一**：**默认统一为 OpenEXR (.exr, half-float, 线性 Rec.709)**，提供 `--format` 覆盖（支持 `exr`/`jpg`/`avif`/`png`）
2. **LumiPic 量化选项**：**暴露 `--quantization {none,8bit,4bit}`**，默认 `none`（bf16），显存不足时建议 `4bit`
3. **模型自动下载**：**首次运行自动下载到项目 `models/` 子目录**，分类保存（如 `models/unet/`, `models/lumipic/`, `models/gmnet/`），支持离线复用
4. **EXIF 保留**：**方案 A+C（全量透传 + 旁车兜底）**：
   - JPEG/AVIF/PNG → 原生 EXIF 注入（JPEG 在 SOI 后插 EXIF APP1，UltraHDR 与 XMP 注入同理）
   - EXR → 关键字段（相机/ISO/快门/光圈/焦距/GPS）写 header 属性 + 完整 EXIF JSON 存 `exifData` 属性
   - 所有格式同时写 `.exif.json` sidecar（可用 `--no-sidecar` 关闭）
5. **色彩空间处理**：输入假设 sRGB；输出默认**线性 Rec.709 (EXR)** / **PQ 编码 (JPEG/AVIF UltraHDR)**；`--colorspace` 可选 `linear-rec709`/`pq-rec2020`/`hlg-rec2020`

### EXIF 技术细节（A+C 落实方式）

| 输出格式 | 主通道 | 实现 |
|----------|--------|------|
| JPEG (UltraHDR) | EXIF APP1 段 | `SOI + EXIF_APP1 + XMP_APP1 + MPF_APP2 + baseJPEG[2:] + gainmapJPEG` |
| AVIF | metadata item | 通过 `--format avif` 时经 hdrconv 写入（其封装管道路径） |
| PNG | `eXIf` chunk | Pillow `save(exif=...)` 原生支持 |
| EXR | header 属性 | `cameraMake` / `isoSpeed` / `exposureTime` / `fNumber` / `focalLength` / `gpsLatitude` 等 + `exifData` (JSON 全量) |
| Sidecar | `.exif.json` | 所有格式默认输出，`--no-sidecar` 关闭 |

> 注意：UltraHDR JPEG 的 EXIF APP1 必须插在 XMP APP1 **之前**（兼容性更好，参考 Google 参考样本布局）；pyultrahdr 的 `build_ultra_hdr` 已采用 `SOI + XMP + MPF + base` 布局，我们在其前面再插入 EXIF 段即可。

---

## 📋 更新后的 CLI 参数规格

| 参数 | 短选项 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `--input` | `-i` | str (可多个) | **必填** | 输入文件或目录路径，支持多个 |
| `--output` | `-o` | str | 自动生成 | 输出目录或文件（单文件时） |
| `--models` | | str (可多个) | `hdrconv` | 模型后端选择，可指定多个并行对比 |
| `--workers` | `--threads` | int | 0 (自动) | 并行进程数，0=按核数/内存自动 |
| `--device` | | choice | `auto` | 设备：`auto`/`cuda`/`cpu`/`mps` |
| `--suffix` | | str | `_hdr` | 输出文件后缀 |
| `--recursive` | `-r` | flag | False | 递归处理子目录 |
| `--dry-run` | | flag | True | 预览模式（默认），不写文件 |
| `--apply` | | flag | False | 真正执行转换 |
| `--log` | | str | `-` | 日志文件路径，`-` 为 stdout |
| `--help` | `-h` | flag | - | 显示帮助 |
| `--format` | | choice | `exr` | 输出格式：`exr`/`jpg`/`avif`/`png`/`ultrahdr` |
| `--quantization` | | choice | `none` | LumiPic 量化：`none`/`8bit`/`4bit` |
| `--colorspace` | | choice | `linear-rec709` | 输出色彩空间：`linear-rec709`/`pq-rec2020`/`hlg-rec2020` |
| `--model-dir` | | str | `./models` | 模型存储目录 |
| `--keep-exif` | | flag | True | 保留 EXIF 元数据 |
| `--gain-map-method` | | choice | `auto` | pyultrahdr 增益图方法：`auto`/`linear`/`log`/`fixed` |
| `--target-nits` | | int | `1000` | 目标峰值亮度 (UltraHDR/Gainmap 用) |

---

## 📚 参考资源

- photo_dedup.py: 并发策略、日志、报告、安全性设计
- hdr-conversion: https://github.com/Jackchou00/hdr-conversion
- SDR-to-HDR-Reconstruction-System: https://github.com/zhuiber/SDR-to-HDR-Reconstruction-System
- GMNet: https://github.com/qtlark/GMNet
- LumiPic: https://github.com/oumad/LumiPic
- sdr2hdr: https://github.com/guofei9987/sdr2hdr
- libultrahdr: https://github.com/google/libultrahdr
- pyultrahdr: https://github.com/hanfeisun/pyultrahdr (纯 Python UltraHDR 编码器)