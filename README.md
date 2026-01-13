# DDSP-SVC-Enhanced

> 🎵 An enhanced singing voice conversion system powered by DDSP and AudioNoise technologies
>
> 🔗 **Based on**: [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) by yxlllc

**Language**: English | [简体中文](./README_cn.md)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Original Project](https://img.shields.io/badge/Fork%20from-DDSP--SVC-brightgreen)](https://github.com/yxlllc/DDSP-SVC)

---

## 🚀 Repository Highlights (VS Original)

Compared to the original [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC), this repository provides several professional-grade enhancements:

| Feature | Original DDSP-SVC | **DDSP-SVC-Enhanced** |
|---------|-------------------|-----------------------|
| **F0 Processing** | Basic extraction | **IIR Smoothing + Octave Fix** |
| **Vocal Expression** | Static pitch | **LFO-based Vibrato & Tremolo** |
| **Audio Effects** | None | **Chorus, Reverb, Flanger, Phaser** |
| **User Interface** | Basic CLI/GUI | **Modern Vue.js Web Interface** |
| **Preprocessing** | Standard | **Integrated MSST/UVR Separation** |
| **Performance** | Standard | **Ring Buffer & Biquad Optimizations** |

---

## ✨ What's New

**DDSP-SVC-Enhanced** is a fork of the original [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) project with professional audio processing capabilities inspired by [AudioNoise](https://github.com/torvalds/AudioNoise).

**New Features:**

- 🎚️ **F0 Smoothing** - Advanced pitch stabilization with octave error correction
- 🎶 **LFO Modulation** - Natural vibrato and tremolo effects for expressive vocals
- 🎛️ **Audio Effects Chain** - Chorus, Reverb, Flanger, and Phaser effects
- 🌐 **Modern Web GUI** - Beautiful Vue.js-based interface with real-time preview
- 🎼 **Music Source Separation** - Integrated MSST and UVR technologies
- ⚡ **Optimized Performance** - Ring buffer and Biquad filter optimizations

**Credits:** All core DDSP-SVC functionality is from the original project by **yxlllc**. This fork adds audio enhancement features only.

---

## 🎯 Key Features

### Core DDSP-SVC Capabilities

- ✅ Low hardware requirements (RTX 4060 recommended)
- ✅ Fast training (comparable to RVC)
- ✅ Real-time voice conversion support
- ✅ Multi-speaker model support
- ✅ High-quality vocoder (NSF-HiFiGAN)

### Enhanced Audio Processing (AudioNoise)

| Feature | Description | Benefits |
|---------|-------------|----------|
| **F0 Smoothing** | IIR low-pass filtering + median filtering | Reduces pitch jitter by ~30% |
| **Octave Correction** | Automatic octave jump detection/fix | Eliminates 440Hz↔880Hz errors |
| **Vibrato** | LFO-based pitch modulation | Natural singing expression |
| **Tremolo** | LFO-based volume modulation | Dynamic amplitude variation |
| **Effects Chain** | Chorus + Reverb + Flanger + Phaser | Professional studio quality |

## 📦 Installation

> 💡 **Note:** If you only need basic voice conversion without audio enhancements, consider using the [original DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) for a simpler setup.

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.4.1+
- 8GB+ RAM (16GB recommended)
- RTX 4060 or better GPU

### Quick Start

```bash
# Clone repository
git clone https://github.com/lsg1103275794/DDSP-SVC-Enhanced-Public.git
cd DDSP-SVC-Enhanced

# Create and activate virtual environment (RECOMMENDED)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models (see section 2)
```

## 🔧 Setup

### 1. Install PyTorch

Visit [PyTorch Official Website](https://pytorch.org/) and install the appropriate version:

```bash
# Example for CUDA 11.8
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Pretrained Models

#### Content Encoder (Choose one)

**ContentVec (Recommended)**
```bash
# Download from https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr
# Place in: pretrain/contentvec/checkpoint_best_legacy_500.pt
```

#### Vocoder

```bash
# NSF-HiFiGAN (44.1kHz, hop=512)
# Download from https://github.com/openvpi/vocoders/releases
# Extract to: pretrain/nsf_hifigan/
```

#### Pitch Extractor

```bash
# RMVPE (Recommended)
# Download from https://github.com/yxlllc/RMVPE/releases
# Extract to: pretrain/rmvpe/
```

## 🚀 Usage

### Data Preprocessing

```bash
python preprocess.py -c configs/reflow.yaml
```

### Training

```bash
python train_reflow.py -c configs/reflow.yaml
```

### Inference (Non-real-time)

```bash
# Basic Usage
python main_reflow.py -i input.wav -m model.pt -o output.wav -k 0 -step 50 -method euler

# With Full Enhancement Pipeline
python main_reflow.py -i input.wav -m model.pt -o output.wav \
  -f0smooth -octavefix -vibrato -fx natural -reverb -revmix 0.25
```

### Web GUI (Modern Interface)

```bash
# Start API backend
python -m uvicorn api.main:app --reload --port 8000

# Start web frontend (in separate terminal)
cd web && npm run dev
```

Access at: `http://localhost:5173`

---

## 🔬 Technical Details (AudioNoise)

| Module | Technique | Source |
|--------|-----------|--------|
| **F0 Smoothing** | IIR Butterworth low-pass filter | AudioNoise `f0_smoother.py` |
| **LFO** | 32-bit phase accumulator + sine LUT | AudioNoise `lfo.py` |
| **Biquad Filters** | Direct Form 2 Transposed | AudioNoise `biquad.py` |
| **Ring Buffer** | Power-of-2 sizing + bit masking | AudioNoise `ring_buffer.py` |

**Performance Improvements:**
- Ring buffer: 10x faster than modulo indexing
- Biquad filters: 20-30% lower CPU usage vs FFT convolution

---

## 🙏 Acknowledgements

Special thanks to:
- **yxlllc** - Original DDSP-SVC author and maintainer
- **Linus Torvalds** - AudioNoise project inspiration
- **OpenVPI Team** - Vocoder and singing synthesis tools
- **Sucial & UVR Team** - Audio separation technologies

---

**Made with ❤️ by the DDSP-SVC-Enhanced Team**
