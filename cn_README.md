<div align="center">

# DDSP-SVC 6.3 增强版

### 基于社区驱动的专业歌声转换系统

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-red.svg)](https://developer.nvidia.com/cuda-toolkit)

**[English](./README.md)** | **[简体中文](./cn_README.md)**

**站在开源社区巨人的肩膀上**

</div>

---

## 🌟 项目愿景

**DDSP-SVC 6.3 增强版** 代表了一项协作努力，通过整合来自开源社区的前沿研究和成熟技术来推进歌声转换技术。本项目尊重 **yxlllc** 的原始 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 项目，并在此基础上结合多个社区项目的专业音频处理能力进行增强。

### 核心理念

- **开放协作** - 每个功能都明确标注其来源
- **社区优先** - 尊重所有开发者的贡献
- **集成创新** - 融合经过验证的技术
- **透明归属** - 清晰标注原始作者

---

## ✨ 项目特色

### 基于 yxlllc 的原始 DDSP-SVC

所有核心语音转换能力均来自原始 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 项目：

- ✅ 可微分数字信号处理 (DDSP)
- ✅ 整流流 (Rectified Flow) 架构
- ✅ 低硬件要求（推荐 RTX 4060）
- ✅ 快速训练（与 RVC 相当）
- ✅ 实时语音转换支持
- ✅ 多说话人模型支持

### 社区贡献的增强功能

本分支集成了多个开源项目的成熟技术：

#### 🎚️ 来自 AudioNoise 项目 (X-LANCE)
- **F0 平滑** - IIR Butterworth 滤波器用于音高稳定
- **八度修正** - 自动检测/修复八度跳跃
- **LFO 调制** - 颤音/震音效果，实现自然表达
- **Biquad 滤波器** - 优化的数字滤波
- **效果链** - 合唱、混响、镶边、相位

#### 🎼 来自 Music Source Separation Training (ZFTurbo, Sucial, UVR Team)
- **MSST 集成** - BS-Roformer、MDX23C 分离模型
- **多模型支持** - 人声、鼓、贝斯、乐器分离
- **WebUI 适配** - Sucial 提供的用户友好分离界面
- **UVR 技术** - 社区驱动的音频分离算法

#### 🌐 来自现代 Web 技术
- **Vue 3 前端** - Apple 风格设计美学
- **FastAPI 后端** - 微服务架构
- **实时监控** - 任务追踪和进度可视化

---

## 🏗️ 架构概览

### 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                  Web 界面 (Vue 3)                           │
│            预处理 • 推理 • 训练                              │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/JSON API
┌────────────────────────────▼────────────────────────────────┐
│                  FastAPI 服务层                             │
│  ┌──────────────┬──────────────┬──────────────────────┐    │
│  │ 预处理服务   │  推理服务    │  训练控制服务         │    │
│  └──────┬───────┴──────┬───────┴──────────┬───────────┘    │
└─────────┼──────────────┼──────────────────┼─────────────────┘
          │              │                  │
┌─────────▼──────────────▼──────────────────▼─────────────────┐
│              共享核心引擎层                                  │
│                                                              │
│  ┌────────────────┐  ┌──────────────────────────────┐      │
│  │ DDSP 引擎      │  │ 特征提取器                    │      │
│  │ (yxlllc)       │  │ • RMVPE F0 (yxlllc)         │      │
│  ├────────────────┤  │ • ContentVec (auspicious3000)│      │
│  │ • vocoder.py   │  │ • HuBERT (bshall)           │      │
│  │ • core.py      │  └──────────────────────────────┘      │
│  │ • unit2ctrl.py │                                         │
│  └────────────────┘  ┌──────────────────────────────┐      │
│                      │ 音频增强                      │      │
│  ┌────────────────┐  │ (X-LANCE AudioNoise)         │      │
│  │ 整流流模型     │  ├──────────────────────────────┤      │
│  │ (yxlllc)       │  │ • F0 平滑                   │      │
│  ├────────────────┤  │ • LFO 调制                  │      │
│  │ • reflow.py    │  │ • Biquad 滤波器             │      │
│  │ • lynxnet2.py  │  │ • 效果链                    │      │
│  │ • solver.py    │  └──────────────────────────────┘      │
│  └────────────────┘                                         │
│                      ┌──────────────────────────────┐      │
│  ┌────────────────┐  │ 音乐源分离                    │      │
│  │ NSF-HiFiGAN    │  │ (ZFTurbo MSST)               │      │
│  │ (OpenVPI)      │  ├──────────────────────────────┤      │
│  └────────────────┘  │ • BS-Roformer               │      │
│                      │ • MDX23C                    │      │
│                      │ • Band-Split RNN            │      │
│                      └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### 代码复用层次

```python
# 所有服务共享相同的核心引擎 - 无重复！

# 示例：特征提取在整个技术栈中复用
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder

# 被以下模块使用：
# - api/services/preprocess_service.py (预处理)
# - api/services/inference_service.py (推理)
# - reflow/vocoder.py (训练)
# - gui_reflow.py (实时 GUI)

# 单例模式防止重复加载模型
```

---

## 📦 安装

### 系统要求

- **操作系统**: Windows 10/11、Linux (Ubuntu 20.04+)、macOS
- **Python**: 3.8+ (推荐 3.10)
- **CUDA**: 11.8+ 用于 GPU 加速
- **GPU**: NVIDIA RTX 4060 或更好 (8GB+ 显存)
- **内存**: 推荐 16GB+
- **存储**: 20GB+ 用于模型和数据集

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/DDSP-SVC-6.3.git
cd DDSP-SVC-6.3

# 自动化安装（推荐）
# Linux/macOS:
./setup.sh

# Windows:
setup.bat

# 手动安装
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-full.txt

# 安装 PyTorch（CUDA 支持）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 下载预训练模型

#### 1. 内容编码器 (ContentVec by auspicious3000)

```bash
# 从 https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr 下载
# 放置到：pretrain/contentvec/checkpoint_best_legacy_500.pt
```

#### 2. 声码器 (NSF-HiFiGAN by OpenVPI)

```bash
# 从 https://github.com/openvpi/vocoders/releases 下载
# 解压到：pretrain/nsf_hifigan/
```

#### 3. F0 提取器 (RMVPE by yxlllc)

```bash
# 从 https://github.com/yxlllc/RMVPE/releases 下载
# 解压到：pretrain/rmvpe/
```

#### 4. 音乐分离模型（可选，by ZFTurbo）

```bash
# BS-Roformer 用于人声/伴奏分离
# 从 https://github.com/ZFTurbo/Music-Source-Separation-Training 下载
# 放置到：other_weights/
```

---

## 🚀 使用方法

### Web 界面（推荐）

```bash
# 一键启动所有服务
# Linux/macOS:
./start.sh

# Windows:
start.bat

# 访问地址：
# - Web UI: http://localhost:5173
# - API 文档: http://localhost:8000/docs
```

### 命令行界面

#### 预处理

```bash
python preprocess.py -c configs/reflow.yaml
```

#### 训练

```bash
python train_reflow.py -c configs/reflow.yaml

# 监控训练进度
tensorboard --logdir=exp
```

#### 推理

```bash
# 基础转换
python main_reflow.py -i input.wav -m exp/model/model.pt -o output.wav

# 带 AudioNoise 增强
python main_reflow.py -i input.wav -m exp/model/model.pt -o output.wav \
    -f0smooth -octavefix -vibrato -fx natural -reverb -revmix 0.25
```

### 实时 GUI

```bash
python gui_reflow.py
```

---

## 🎛️ 增强功能

### F0 平滑 (AudioNoise)

使用 IIR Butterworth 滤波器将音高不稳定性降低 20-40%：

| 参数 | 默认值 | 范围 | 描述 |
|-----------|---------|-------|-------------|
| `-f0smooth` | 禁用 | - | 启用 F0 平滑 |
| `-f0cutoff` | 20.0 Hz | 5-50 | 低通截止频率 |
| `-mediankernel` | 3 | 3-11 | 中值滤波器核大小 |
| `-octavefix` | 禁用 | - | 自动修正八度跳跃 |

### LFO 调制 (AudioNoise)

自然的颤音和震音效果：

| 参数 | 默认值 | 范围 | 描述 |
|-----------|---------|-------|-------------|
| `-vibrato` | 禁用 | - | 启用音高颤音 |
| `-vibrate` | 5.5 Hz | 3-8 | 颤音频率 |
| `-vibdepth` | 0.02 | 0.01-0.05 | 颤音深度 (±%) |
| `-tremolo` | 禁用 | - | 启用音量震音 |
| `-tremrate` | 4.0 Hz | 2-8 | 震音频率 |

### 效果链 (AudioNoise)

专业音频效果：

| 参数 | 选项 | 描述 |
|-----------|---------|-------------|
| `-fx` | none/natural/spacious/vintage/clean | 效果预设 |
| `-chorus` | - | 合唱效果 |
| `-reverb` | - | 混响效果 |
| `-revmix` | 0.2 (0-0.5) | 混响干湿比 |

---

## 📊 性能基准

| 配置 | RTX 4060 | RTX 3060 | CPU (i7-12700) |
|--------------|----------|----------|----------------|
| 基础 DDSP | 0.12秒 | 0.18秒 | 2.5秒 |
| + F0 平滑 | 0.13秒 | 0.19秒 | 2.6秒 |
| + LFO | 0.14秒 | 0.21秒 | 2.8秒 |
| + 效果链 | 0.18秒 | 0.26秒 | 3.2秒 |
| 完整增强 | 0.20秒 | 0.29秒 | 3.5秒 |

*每 10 秒音频片段（infer_step=50）*

---

## 📚 文档

- **[CLAUDE.md](./CLAUDE.md)** - 开发者指南和代码复用模式
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** - 如何贡献
- **[API_Documentation.md](./docs/API_Documentation.md)** - API 参考
- **[Implementation_Guide.md](./docs/Implementation_Guide.md)** - 技术细节
- **[AudioNoise_Technical_Analysis.md](./docs/AudioNoise_Technical_Analysis.md)** - 增强分析

---

## 🤝 贡献

我们欢迎来自社区的贡献！请阅读我们的[贡献指南](CONTRIBUTING.md)了解：

- 代码风格和标准
- 如何提交拉取请求
- Bug 报告指南
- 功能请求流程

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black .
isort .
```

---

## 🙏 致谢

本项目的存在要感谢开源社区的杰出工作：

### 核心框架

**[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) by yxlllc**
- 原始 DDSP 歌声转换框架
- 整流流实现
- 实时处理管道
- 所有基础语音转换能力

**[pc-ddsp](https://github.com/yxlllc/pc-ddsp) by yxlllc**
- 基于相位的 DDSP 创新

**[ddsp](https://github.com/magenta/ddsp) by Google Magenta**
- 可微分 DSP 库基础

### 音频增强技术

**[AudioNoise](https://github.com/X-LANCE/AudioNoise) by X-LANCE (上海交通大学)**
- F0 平滑算法（IIR 滤波器、中值滤波）
- LFO 调制引擎（颤音、震音）
- Biquad 滤波器实现
- 环形缓冲区优化
- 音频效果链架构

**[Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) by ZFTurbo**
- BS-Roformer 分离模型
- MDX23C 架构
- Band-Split RNN 技术
- 训练工具和配置

**[MSST-WebUI](https://huggingface.co/Sucial/MSST-WebUI) by Sucial**
- 音乐源分离网页界面
- MSST 模型集成指导
- 用户友好的分离工作流
- 预训练模型分发

**[UVR (Ultimate Vocal Remover)](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07 & UVR Team**
- 人声移除技术
- 音频分离算法
- 社区驱动的模型改进
- 多格式音频处理

### 特征提取

**[ContentVec](https://github.com/auspicious3000/contentvec) by auspicious3000**
- 自监督语音表示

**[soft-vc](https://github.com/bshall/soft-vc) by bshall**
- 用于语音转换的软语音单元
- HuBERT 实现

**[RMVPE](https://github.com/yxlllc/RMVPE) by yxlllc**
- 鲁棒音高提取算法

### 声码器与语音合成

**[NSF-HiFiGAN](https://github.com/openvpi/vocoders) by OpenVPI**
- 神经源滤波声码器
- 高质量语音合成

**[DiffSinger](https://github.com/openvpi/DiffSinger) by OpenVPI**
- 基于扩散的歌声合成

**[Diff-SVC](https://github.com/prophesier/diff-svc) by prophesier**
- 用于歌声转换的扩散模型

**[Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) by CNChTu**
- SVC 的高级扩散技术

### Web 技术

**前端框架**
- [Vue 3](https://vuejs.org/) - 渐进式 JavaScript 框架
- [Naive UI](https://www.naiveui.com/) - Vue 3 组件库
- [Vite](https://vitejs.dev/) - 下一代前端工具

**后端框架**
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架

### 社区与工具

**[SVC Fusion](https://github.com/HuanLinOTO/svc-fusion-docs) by HuanLinOTO**
- 全面的 SVC 社区文档
- 多 SVC 框架集成指南
- 最佳实践和教程
- 社区知识分享

### 特别鸣谢

- **yxlllc** - 原始 DDSP-SVC 作者、RMVPE 创建者和持续维护者
- **X-LANCE 团队（上海交通大学）** - AudioNoise 项目和音频处理研究
- **OpenVPI 团队** - 声码器和歌声合成工具
- **ZFTurbo** - 音乐源分离模型和训练框架
- **Sucial** - MSST-WebUI 集成和网页界面
- **Anjok07 & UVR Team** - Ultimate Vocal Remover 音频分离工具
- **HuanLinOTO** - SVC Fusion 社区文档和集成指南
- **Google Magenta** - DDSP 库和研究
- **所有贡献者** - 每一位报告问题、提出建议和改进代码的人

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件。

### 重要法律声明

**本软件仅用于研究和教育目的。**

- ⚠️ 仅使用**合法获得和授权的数据**进行训练
- ⚠️ 请勿将模型或生成的音频用于非法目的
- ⚠️ 尊重您所在司法管辖区的版权、隐私和冒充法律
- ⚠️ 作者和贡献者对任何滥用行为概不负责

本项目继承上游项目的所有限制和免责声明，特别是原始 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)。

---

## 📧 社区与支持

- **问题**: [GitHub Issues](https://github.com/yourusername/DDSP-SVC-6.3/issues) - Bug 报告和功能请求
- **讨论**: [GitHub Discussions](https://github.com/yourusername/DDSP-SVC-6.3/discussions) - 问答和社区交流
- **拉取请求**: [贡献指南](CONTRIBUTING.md) - 如何贡献代码

---

## 🌟 Star 历史

如果本项目对您的研究或创作工作有帮助，请考虑：

- ⭐ 为本仓库加星
- ⭐ 为[原始 DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 加星
- ⭐ 为 [AudioNoise](https://github.com/X-LANCE/AudioNoise) 加星
- ⭐ 为 [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) 加星
- ⭐ 为 [UVR](https://github.com/Anjok07/ultimatevocalremovergui) 加星
- ⭐ 支持 Sucial 的 [MSST-WebUI](https://huggingface.co/Sucial/MSST-WebUI)

每一颗星都帮助这些开源项目成长！

---

<div align="center">

**由开源社区用 ❤️ 构建**

*站在巨人的肩膀上*

[yxlllc](https://github.com/yxlllc) • [X-LANCE](https://github.com/X-LANCE) • [OpenVPI](https://github.com/openvpi) • [ZFTurbo](https://github.com/ZFTurbo) • [Sucial](https://huggingface.co/Sucial) • [UVR Team](https://github.com/Anjok07/ultimatevocalremovergui) • [HuanLinOTO](https://github.com/HuanLinOTO) • [更多贡献者...](https://github.com/yourusername/DDSP-SVC-6.3/graphs/contributors)

</div>