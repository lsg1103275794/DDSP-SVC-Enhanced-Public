import os
import sys
import torch
import yaml
import time
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Any, List
from api.core.config import MSST_ROOT, MSST_WEIGHTS_DIR, PROCESSED_DIR, DEVICE

# 将 MSST 根目录添加到系统路径以支持内部导入
if MSST_ROOT not in sys.path:
    sys.path.append(MSST_ROOT)

try:
    from Music_Source_Separation_Training.utils import (
        get_model_from_config,
        demix_track,
        demix_track_demucs,
    )

    HAS_MSST = True
except ImportError:
    HAS_MSST = False


class MSSTService:
    def __init__(self):
        self.current_model = None
        self.current_config = None
        self.current_model_type = None
        self.device = DEVICE

    def _load_model(self, model_type: str, config_path: str, checkpoint_path: str):
        """加载或切换模型单例"""
        if not HAS_MSST:
            raise ImportError(
                "MSST modules not found. Check Music_Source_Separation_Training directory."
            )

        if self.current_model_type != model_type or self.current_model is None:
            print(f"Loading MSST model: {model_type} with config {config_path}")
            model, config = get_model_from_config(model_type, config_path)

            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                if model_type == "htdemucs" and "state" in state_dict:
                    state_dict = state_dict["state"]
                model.load_state_dict(state_dict)

            model = model.to(self.device)
            model.eval()

            self.current_model = model
            self.current_config = config
            self.current_model_type = model_type

        return self.current_model, self.current_config

    async def separate(
        self,
        input_path: str,
        model_type: str,
        config_name: str,
        checkpoint_name: str,
        stems: int = 2,
        quality_params: Dict[str, Any] = None,
    ) -> Dict[str, str]:
        """执行人声分离"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 构造路径
        config_path = os.path.join(MSST_ROOT, "configs", config_name)
        checkpoint_path = os.path.join(MSST_WEIGHTS_DIR, checkpoint_name)

        # 加载模型
        model, config = self._load_model(model_type, config_path, checkpoint_path)

        # 准备输出目录
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_subdir = os.path.join(PROCESSED_DIR, "separated", base_name)
        os.makedirs(output_subdir, exist_ok=True)

        vocal_path = os.path.join(output_subdir, "vocal.wav")
        instr_path = os.path.join(output_subdir, "instrument.wav")

        # 加载音频 (参考 MSST inference.py)
        mix, sr = librosa.load(input_path, sr=44100, mono=False)
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=-1)
        else:
            mix = mix.T  # to [L, C]

        mixture = torch.tensor(mix.T, dtype=torch.float32)  # to [C, L]

        # 执行分离
        with torch.no_grad():
            if model_type == "htdemucs":
                res = demix_track_demucs(config, model, mixture, self.device)
            else:
                # Mock progress for demix_track
                class MockProgress:
                    def __call__(self, *args, **kwargs):
                        pass

                    def update(self, *args, **kwargs):
                        pass

                res = demix_track(
                    config, model, mixture, self.device, MockProgress(), "Separating..."
                )

        # 保存结果
        instruments = config.training.instruments
        target = (
            config.training.target_instrument
            if config.training.target_instrument
            else instruments[0]
        )

        # 保存人声
        sf.write(vocal_path, res[target].T, sr, subtype="FLOAT")

        # 计算伴奏 (Original - Vocal)
        # 重新加载以对齐长度
        vocal_audio, _ = librosa.load(vocal_path, sr=sr, mono=False)
        # 对齐原始音频长度
        if vocal_audio.shape[-1] < mix.T.shape[-1]:
            mix_trimmed = mix.T[:, : vocal_audio.shape[-1]]
        else:
            vocal_audio = vocal_audio[:, : mix.T.shape[-1]]
            mix_trimmed = mix.T

        instr_audio = mix_trimmed - vocal_audio
        sf.write(instr_path, instr_audio.T, sr, subtype="FLOAT")

        return {"vocal": vocal_path, "instrument": instr_path}


msst_service = MSSTService()
