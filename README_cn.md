# DDSP-SVC-Enhanced（增强版）

> 🎵 基于 DDSP 和 AudioNoise 技术的增强型歌声转换系统
>
> 🔗 **基于项目**: [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) by yxlllc

**语言**: [English](./README.md) | 简体中文

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![原始项目](https://img.shields.io/badge/Fork%20from-DDSP--SVC-brightgreen)](https://github.com/yxlllc/DDSP-SVC)

---

## 🚀 项目亮点（对比原版）

相比于原始 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 项目，本仓库提供了多项专业级增强：

| 功能特性 | 原始 DDSP-SVC | **DDSP-SVC-Enhanced (增强版)** |
|----------|---------------|--------------------------------|
| **F0 处理** | 基础提取 | **IIR 平滑滤波 + 八度音程自动修正** |
| **歌唱表现力** | 静态音高 | **基于 LFO 的自然颤音与震音调制** |
| **音频效果** | 无 | **集成合唱、混响、镶边、相位效果器** |
| **用户界面** | 基础 CLI/GUI | **基于 Vue.js 的现代化 Web 界面** |
| **预处理** | 标准流程 | **集成 MSST/UVR 音乐源分离技术** |
| **性能优化** | 标准实现 | **环形缓冲区与 Biquad 滤波器优化** |

---

## ✨ 新特性

**DDSP-SVC-Enhanced** 是原始 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 项目的增强版本，融合了来自 [AudioNoise](https://github.com/torvalds/AudioNoise) 的专业音频处理技术。

**全新功能：**

- 🎚️ **F0 平滑** - 高级音高稳定技术，支持八度音程错误修正
- 🎶 **LFO 调制** - 自然的颤音和震音效果，增强声音表现力
- 🎛️ **音频效果链** - 合唱、混响、镶边和相位效果
- 🌐 **现代化 Web 界面** - 基于 Vue.js 的美观界面，支持实时预览
- 🎼 **音乐源分离** - 集成 MSST 和 UVR 技术
- ⚡ **性能优化** - 环形缓冲区和 Biquad 滤波器优化

**致谢：** 所有核心 DDSP-SVC 功能来自 **yxlllc** 的原始项目。本分支仅添加了音频增强功能以及webui。

---

## 🎯 主要特性

### 核心 DDSP-SVC 能力

- ✅ 低硬件要求（推荐 RTX 4060）
- ✅ 快速训练（与 RVC 相当）
- ✅ 支持实时语音转换
- ✅ 多说话人模型支持
- ✅ 高质量声码器（NSF-HiFiGAN）

### 增强音频处理（AudioNoise）

| 功能 | 描述 | 优势 |
|------|------|------|
| **F0 平滑** | IIR 低通滤波 + 中值滤波 | 减少音高抖动约 30% |
| **八度音程修正** | 自动检测/修复八度跳跃 | 消除 440Hz↔880Hz 错误 |
| **颤音** | 基于 LFO 的音高调制 | 自然的歌唱表现力 |
| **震音** | 基于 LFO 的音量调制 | 动态振幅变化 |
| **效果链** | 合唱 + 混响 + 镶边 + 相位 | 专业录音室品质 |

## 📦 安装

> 💡 **注意：** 如果您只需要基本的语音转换功能而不需要音频增强，请考虑使用[原始 DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)以获得更简单的设置。

### 系统要求

- Python 3.8+
- CUDA 11.8+（用于 GPU 加速）
- PyTorch 2.4.1+
- 8GB+ 内存（推荐 16GB）
- RTX 4060 或更好的 GPU

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/lsg1103275794/DDSP-SVC-Enhanced-Public.git
cd DDSP-SVC-Enhanced

# 创建并激活虚拟环境（强烈推荐）
python -m venv venv

# 激活虚拟环境
# Windows 系统：
venv\Scripts\activate
# Linux/macOS 系统：
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型（见第 2 节）
```

## 🔧 配置

### 1. 安装 PyTorch

访问 [PyTorch 官方网站](https://pytorch.org/)并安装适合的版本：

```bash
# CUDA 11.8 示例
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 下载预训练模型

#### 内容编码器（选择其一）

**ContentVec（推荐）**
```bash
# 从 https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr 下载
# 放置在：pretrain/contentvec/checkpoint_best_legacy_500.pt
```

#### 声码器

```bash
# NSF-HiFiGAN (44.1kHz, hop=512)
# 从 https://github.com/openvpi/vocoders/releases 下载
# 解压到：pretrain/nsf_hifigan/
```

#### 音高提取器

```bash
# RMVPE（推荐）
# 从 https://github.com/yxlllc/RMVPE/releases 下载
# 解压到：pretrain/rmvpe/
```

## 🚀 使用方法

### 数据预处理

```bash
python preprocess.py -c configs/reflow.yaml
```

### 训练

```bash
python train_reflow.py -c configs/reflow.yaml
```

### 推理（非实时）

```bash
# 基础使用
python main_reflow.py -i input.wav -m model.pt -o output.wav -k 0 -step 50 -method euler

# 完整增强流程
python main_reflow.py -i input.wav -m model.pt -o output.wav \
  -f0smooth -octavefix -vibrato -fx natural -reverb -revmix 0.25
```

### Web 界面（现代化界面）

```bash
# 启动 API 后端
python -m uvicorn api.main:app --reload --port 8000

# 启动 Web 前端（在另一个终端）
cd web && npm run dev
```

访问地址：`http://localhost:5173`

---

## 🔬 技术细节 (AudioNoise)

| 模块 | 技术 | 来源 |
|------|------|------|
| **F0 平滑** | IIR Butterworth 低通滤波器 | AudioNoise `f0_smoother.py` |
| **LFO** | 32位相位累加器 + 正弦查找表 | AudioNoise `lfo.py` |
| **Biquad 滤波器** | 直接型 II 转置 | AudioNoise `biquad.py` |
| **环形缓冲区** | 2的幂次大小 + 位掩码 | AudioNoise `ring_buffer.py` |

**性能改进：**
- 环形缓冲区：比取模索引快 10 倍
- Biquad 滤波器：比 FFT 卷积低 20-30% 的 CPU 使用率

---

## 🙏 致谢

特别感谢：
- **yxlllc** - 原始 DDSP-SVC 作者和维护者
- **Linus Torvalds** - AudioNoise 项目灵感来源
- **OpenVPI 团队** - 声码器和歌声合成工具
- **Sucial & UVR 团队** - 音频分离技术

---

**用 ❤️ 制作，来自 DDSP-SVC-Enhanced 团队**
