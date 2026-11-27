# T4 多图层模型部署实验

实验目录为服务器 `36.134.45.38` 的 `/data/image2ppt-layer-experiments/20261105-r1`，镜像为 `image2ppt-qwen-layered:20261105-r1`。当前提供 Docker/CLI 推理入口。

## Qwen-Image-Layered GGUF

- 基础模型固定为 `Qwen/Qwen-Image-Layered@8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf`。
- GGUF 固定为 `unsloth/Qwen-Image-Layered-GGUF@4bb0a560b7a8fb190b10498992ea39af5a80ff04`，文件为 `qwen-image-layered-Q3_K_M.gguf`，约 9.01 GB。
- 基础配套文件和 GGUF 共约 25.87 GB，文件大小、Git blob 摘要或 LFS SHA-256 记录在服务器 `asset-manifest.json`。
- 环境使用 Torch 2.5.1+cu124、Diffusers 0.37.0、Transformers 4.57.3。完整依赖由镜像内 `/opt/layer-experiments-environment.lock` 保存。
- 文本编码在 CPU 上以 FP32 执行，结果保存为 safetensors。页面描述为当前对话中人工撰写的图像说明，位于 `scripts/caption-page03.txt`，需要与被处理页面对应。
- 扩散模型在物理 GPU 2 上运行，GPU UUID 为 `GPU-44e3bad2-be08-97b6-5769-f50c01fb40e4`。GGUF 权重保持量化存储，主模型和 RGBA-VAE 的默认计算精度均为 FP32。
- T4 上 FP16 试运行出现非有限数值。运行器保留逐步数值检查，并适配了 Diffusers 0.37 对 `time_text_embed.addition_t_embedding.weight` 的 GGUF 反量化处理。该适配记录在输出 manifest 中。
- 每个试运行容器限制为 8 CPU、72 GiB 主机内存、256 PID。运行时模型只读挂载，产物目录可写，网络模式为 none。

`640` 为模型的面积分辨率档，当前第 3 页输出为 `864×480`。图层输出是 RGBA PNG，文字保留为像素。`manifest.json` 保存步数、精度、显存峰值、原图哈希及合成误差。

## 复跑第 3 页

在服务器实验目录运行以下命令。每次使用新的容器名和输出目录，保留已完成版本。

```python
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from bootstrap_qwen import run_container

source = next(Path('/data/image2ppt-vision/data/pilot/deck-src').glob('03_*.png'))
run_container(
    'i2p-qwen-page03-10step-new',
    ['infer',
     '--output', '/workspace/outputs/page03-10step-new',
     '--embeddings', '/workspace/outputs/embeddings-page03-r1',
     '--input', '/input/' + source.name,
     '--steps', '10', '--layers', '3'],
    gpu=True,
)
```

运行前查看所选 GPU 的占用情况。`run_container` 将标准输出和错误保存到 `evidence/<容器名>.log`，容器退出状态也会保留。新页面需要先通过 `run_qwen.py encode` 为对应描述生成新的缓存，再使用 `infer`。

## OmniPSD 与 PixelART

OmniPSD 核查版本为 `showlab/OmniPSD@1bfeae9cea6d01ad73b4538a7ff9c2eb10007f09`。当前官方文件树中没有模型权重文件，GitHub Release 数为 0，公开推理入口是 `inference/infer_psd_flux.py`。图像分解所需四个任务专家 LoRA 及匹配 RGBA-VAE 的公开下载引用仍待补充。[官方 issue](https://github.com/showlab/OmniPSD/issues/7)

PixelART 已核对论文、Hugging Face 模型目录和 GitHub 仓库搜索，当前没有确认到对应的公开推理权重。论文披露训练语料和内部布局检测器的公开资产限制，并要求外部布局框。[论文原文](https://arxiv.org/html/2609.22789v1)

这两项当前停在权重资产阶段，详情保存在 `evidence/asset-preflight.json`，模型 GPU 推理测试状态为 false。
