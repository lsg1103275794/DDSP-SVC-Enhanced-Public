# DDSP-SVC-Enhanced - 具备专业音频增强功能的下一代歌声转换系统

[**English**](./README.md) | **中文**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/lsg1103275794/DDSP-SVC-Enhanced-Public?style=social)](https://github.com/lsg1103275794/DDSP-SVC-Enhanced-Public)

> 🚀 **核心定位**：基于 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 的专业级增强分支，集成先进 DSP 技术实现极致音高稳定性。
>
> 🎨 **全新功能**：**LFO 动态表现力系统** & **内置录音室 FX 效果链** - 让 AI 合成声线拥有自然的情感起伏。
>
> 💡 **致谢与致敬**：本项目的核心 DDSP-SVC 功能完全来自 [yxlllc](https://github.com/yxlllc/DDSP-SVC) 的原始项目。本增强版在此基础上添加了音频增强算法、Web 界面及多项性能优化，旨在提供更专业的歌声合成体验。

---

## 📖 目录
- [🚀 仓库亮点 (VS 原版)](#-仓库亮点-vs-原版)
- [✨ 核心特性](#-核心特性)
- [📦 安装指南](#-安装指南)
- [🔧 快速开始](#-快速开始)
- [📂 数据准备](#-数据准备)
- [🌐 Web 界面](#-web-界面)
- [🔬 技术架构](#-技术架构)
- [🗺️ 路线图](#-路线图)
- [🤝 参与贡献](#-参与贡献)

---

## 🚀 仓库亮点 (VS 原版)

本分支专为追求**录音室级品质**和**现代工作流**的用户打造。

| 特性 | 原版 DDSP-SVC | **DDSP-SVC-Enhanced** | 提升效果 |
|:---|:---:|:---:|:---|
| **音高稳定性** | 基础提取 | **IIR + 中值滤波** | 彻底消除长音中的音高抖动 |
| **八度修复** | 手动调整 | **自动修正** | 消除 95% 以上的八度跳变错误 |
| **表现力增强** | 静态音高 | **LFO 揉弦与颤音** | 为合成人声注入情感与生命力 |
| **后期处理** | 需外部插件 | **内置 FX 效果链** | 无需离开程序即可获得专业听感 |
| **用户界面** | 命令行/旧版 GUI | **现代 Vue.js 3 Web UI** | 流程化、美观且直观的操作体验 |
| **预处理** | 基础功能 | **集成 MSST + UVR** | 开箱即用的高质量人声分离 |

---

## ✨ 核心特性

### 💎 专业音频增强 (AudioNoise 核心)
- 🎙️ **高级 F0 平滑**: 采用 IIR 巴特沃斯滤波器稳定音高，对气息音或弱声部分尤为有效。
- 🎹 **智能八度修正**: 实时检测并修复突发的八度跳变（如 440Hz 与 880Hz 间的错误切换）。
- 🌈 **动态 LFO 调制**: 可调节频率与深度的自然揉弦 (Vibrato) 与颤音 (Tremolo) 效果。
- 🎛️ **录音室级效果链**: 内置高保真合唱 (Chorus)、混响 (Reverb)、法兰 (Flanger) 和移相 (Phaser)。

### ⚡ 性能与效率优化
- ⚙️ **优化 DSP 内核**: 使用环形缓冲区和 Biquad 滤波器转置结构，降低 20-30% CPU 占用。
- 🚄 **低延迟设计**: 针对实时转换和监听进行了深度优化，支持毫秒级响应。
- 🎵 **高保真输出**: 由 NSF-HiFiGAN 驱动，完美支持 44.1kHz/48kHz 采样率。

---

## 📦 安装指南

### 系统要求
- **操作系统**: Windows 10/11, Linux (推荐 Ubuntu 20.04+)
- **显卡**: NVIDIA RTX 30/40 系列 (推荐 8GB+ 显存)
- **Python**: 3.8 - 3.11

### 1. 克隆项目与环境配置
```bash
# 克隆仓库
git clone https://github.com/lsg1103275794/DDSP-SVC-Enhanced-Public.git
cd DDSP-SVC-Enhanced

# 创建虚拟环境（强烈推荐，以隔离依赖环境）
python -m venv venv
# 激活环境 (Windows)
venv\Scripts\activate
# 激活环境 (Linux/macOS)
source venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
# 安装带 CUDA 支持的 PyTorch (以 CUDA 11.8 为例)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔧 快速开始

### 第一步：下载预训练模型
请将以下文件放入 `pretrain/` 目录：
- **ContentVec (推荐)**: `pretrain/contentvec/checkpoint_best_legacy_500.pt`
- **声码器 (Vocoder)**: `pretrain/nsf_hifigan/` (从 [OpenVPI](https://github.com/openvpi/vocoders/releases) 下载并解压)
- **音高提取器 (Pitch Extractor)**: `pretrain/rmvpe/model.pt`

### 第二步：单条指令推理
```bash
# 使用全套增强功能：平滑音高 + 八度修正 + 颤音 + 混响
python main_reflow.py -i input.wav -m model.pt -o output.wav \
  -f0smooth -octavefix -vibrato -fx natural -reverb -revmix 0.25
```

---

## 📂 数据准备

### 目录结构规范

#### 单说话人 (Single Speaker)
```text
data/
├── train/audio/    # 约 1000+ 个 2秒以上的 .wav 文件
│   ├── song1.wav
│   └── ...
└── val/audio/      # 约 10 个验证文件
    └── test1.wav
```

#### 多说话人 (Multi Speaker)
```text
data/
├── train/audio/
│   ├── spk1/       # 说话人 1
│   │   ├── a.wav
│   │   └── ...
│   └── spk2/       # 说话人 2
│       └── b.wav
└── val/audio/
    ├── spk1/
    └── spk2/
```

### 💡 训练与预处理小贴士
- **预处理**: 使用 `python preprocess.py -c configs/reflow.yaml`。多说话人模型请在配置文件中设置 `n_spk` 数量。
- **训练**: 运行 `python train_reflow.py -c configs/reflow.yaml`。程序会自动恢复中断的进度。
- **监控**: 使用 `tensorboard --logdir=exp` 实时查看训练曲线。

---

## 🌐 Web 界面

通过我们现代化的 Web 界面体验 **DDSP-SVC-Enhanced** 的全部功能。

1. **启动后端**: `python -m uvicorn api.main:app --port 8000`
2. **启动前端**: `cd web && npm install && npm run dev`
3. **打开浏览器**: 访问 `http://localhost:5173`

> 🎨 **界面特性**: 实时音高可视化、交互式效果器滑块、批量处理支持以及深色模式。

---

## 🔬 技术架构

增强流水线遵循高性能 DSP 架构：

1. **输入**: 原始音频 (Mono, 44.1kHz)
2. **特征提取**: ContentVec (Units) + RMVPE (F0)
3. **增强层**:
    - **八度修复器** -> **F0 平滑器 (IIR)** -> **LFO 调制器**
4. **合成**: DDSP 谐波 + 噪声模型
5. **效果链**: Biquad 滤波器 -> 合唱 -> 混响
6. **输出**: 增强后的歌声

---

## 🗺️ 路线图
- [ ] **v1.1**: 支持实时 VST 插件
- [ ] **v1.2**: 集成更先进的音高提取器 (如 FCPE)
- [ ] **v1.3**: 为 Windows 用户提供一键安装包
- [ ] **v2.0**: 支持基于扩散 (Diffusion) 的增强层

---

<div align="center">

**[GitHub 仓库](https://github.com/lsg1103275794/DDSP-SVC-Enhanced-Public)** • **[项目文档](docs/)** • **[报告问题](https://github.com/lsg1103275794/DDSP-SVC-Enhanced-Public/issues)**

为您而作 ❤️ DDSP-SVC 开发者社区

</div>
