以下是针对图像SDR转HDR的开源项目及其官方 GitHub 地址与一句话推荐，请基于这些项目分析并创建一个包含所有项目模型的 photo_sdr_to_hdr.py, cli 参数至少包含 --input/-i, --output/-o, --models（需区分不同的项目）, --workers/--threads, --devide auto|cuda|cpu， --suffix, --recursive/-r, --dry-run, --log, --help 等，需要支持文件/文件夹批量处理。请先给出具体的执行方案再进行具体开发设计。

🛠️ 工程应用阶段
libultrahdr

📦 https://github.com/google/libultrahdr
Ultra HDR 格式的官方 C++ 参考编解码库，是 Android 系统的原生支持组件，格式合规性与稳定性最高。

hdr-conversion

📦 https://github.com/Jackchou00/hdr-conversion
Python 实现的 HDR 格式转换工具，支持 UltraHDR、Adaptive Gainmap 及 PQ/HLG 等多种格式的解析、写入与互转，pip 即可安装。

sdr2hdr

📦 https://github.com/guofei9987/sdr2hdr
Rust 编写的轻量级工具，通过嵌入 ICC 配置文件让普通 PNG/JPEG 在 HDR 显示器上呈现更亮效果，cargo install 即可使用。

gainmap-js

📦 https://github.com/MONOGRID/gainmap-js
Adobe Gain Map 技术的 TypeScript 实现，专为 three.js 3D 库设计，适合 Web 前端场景的 HDR 内容编码与解码。

pyultrahdr

📦 https://github.com/hanfeisun/pyultrahdr

一个约 230 行的纯 Python 脚本，无需编译 libultrahdr 即可将任意 SDR 图像转换为字节兼容的 Ultra HDR JPEG，非常适合快速集成到 Python 流水线或批量任务中。


🧪 研究/接近工程化
SDR-to-HDR-Reconstruction-System

📦 https://github.com/zhuiber/SDR-to-HDR-Reconstruction-System
基于 U-Net 的 SDR 转 HDR 深度学习项目，提供预训练的 ONNX 模型可直接推理，工程集成门槛较低。

GMNet

📦 https://github.com/qtlark/GMNet
ICLR 2025 论文官方实现，提出双分支网络学习 SDR 图像的 Gain Map，与 Ultra HDR 等格式兼容性方向明确。

LumiPic

📦 https://github.com/oumad/LumiPic
基于扩散模型 LoRA 的单图像 SDR 转浮点 HDR EXR 方案，支持 ARRI LogC3 编码，适合专业后期工作流。

📦 其他工具与资源
HDR Image Transcoder

📦 https://github.com/Yuuichu/hdr-image-transcoder
多格式 HDR 图像转码工具，支持 Gainmap AVIF、AVIF/JXL/HEIF 色彩元数据等格式的互转。

gainforge

📦 https://github.com/awxkee/gainforge
Rust 实现的 HDR 色调映射库，专注于 HDR 到 SDR 的转换，适合需要反向处理流程的场景。

Awesome-Gain-Maps

📦 https://github.com/ronv/awesome-gain-maps（注：该仓库地址为社区常用入口，如遇迁移可通过 GitHub 搜索 "awesome gain maps" 定位最新地址）
Gain Map HDR 领域的资源汇总列表，涵盖代码、论文与示例图像，适合系统性了解该技术生态。