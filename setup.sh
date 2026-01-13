#!/bin/bash

# ===================================
# DDSP-SVC-Enhanced Setup Script
# ===================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════╗"
echo "║  DDSP-SVC-Enhanced Setup             ║"
echo "╚══════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Skipping...${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install PyTorch
echo "🔥 Installing PyTorch..."
echo "Please select your CUDA version:"
echo "  1) CUDA 11.8"
echo "  2) CUDA 12.1"
echo "  3) CPU only"
read -p "Enter choice (1-3): " CUDA_CHOICE

case $CUDA_CHOICE in
    1)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac
echo -e "${GREEN}✅ PyTorch installed${NC}"
echo ""

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✅ Core dependencies installed${NC}"
echo ""

# Install API dependencies
echo "🌐 Installing API dependencies..."
pip install -r requirements-api.txt
echo -e "${GREEN}✅ API dependencies installed${NC}"
echo ""

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
if command -v npm &> /dev/null; then
    cd web
    npm install
    cd ..
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  npm not found. Skipping frontend setup.${NC}"
    echo "   Please install Node.js and run: cd web && npm install"
fi
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/train/audio
mkdir -p data/val/audio
mkdir -p storage/uploads
mkdir -p storage/processed
mkdir -p storage/outputs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Download pretrained models reminder
echo -e "${YELLOW}"
echo "⚠️  IMPORTANT: Download pretrained models"
echo ""
echo "1. ContentVec encoder:"
echo "   https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr"
echo "   → Place in: pretrain/contentvec/"
echo ""
echo "2. NSF-HiFiGAN vocoder:"
echo "   https://github.com/openvpi/vocoders/releases"
echo "   → Place in: pretrain/nsf_hifigan/"
echo ""
echo "3. RMVPE pitch extractor:"
echo "   https://github.com/yxlllc/RMVPE/releases"
echo "   → Place in: pretrain/rmvpe/"
echo -e "${NC}"

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════╗"
echo "║  Setup Complete! 🎉                  ║"
echo "╚══════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Next steps:"
echo "  1. Download pretrained models (see above)"
echo "  2. Prepare your dataset in data/train/audio/"
echo "  3. Run: ./start.sh"
echo ""
