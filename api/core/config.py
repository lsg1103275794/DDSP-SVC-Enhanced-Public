import os
import torch

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 存储路径配置
UPLOAD_DIR = os.path.join(BASE_DIR, "storage", "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "storage", "processed")
DATASETS_DIR = os.path.join(BASE_DIR, "storage", "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "storage", "outputs")
EXP_DIR = os.path.join(BASE_DIR, "exp")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# MSST 相关路径 (Music Source Separation Training)
MSST_ROOT = os.path.join(BASE_DIR, "Music_Source_Separation_Training")
MSST_WEIGHTS_DIR = os.path.join(BASE_DIR, "other_weights")

# 创建必要目录
for d in [UPLOAD_DIR, PROCESSED_DIR, DATASETS_DIR, OUTPUT_DIR, EXP_DIR, CONFIG_DIR, DATA_DIR, MSST_WEIGHTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "None"

# API 配置
API_V1_STR = "/api/v1"
PROJECT_NAME = "DDSP-SVC Apple Style API"
