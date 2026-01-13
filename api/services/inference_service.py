import os
import torch
import librosa
import numpy as np
import soundfile as sf
import torchaudio
from typing import List, Dict, Any, Optional
from api.core.config import DEVICE, EXP_DIR, UPLOAD_DIR, OUTPUT_DIR, PROCESSED_DIR
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.enhancements import F0EnhancedProcessor, F0ProcessorConfig
from ddsp.lfo import apply_modulation, ModulationConfig
from ddsp.effects_chain import AudioEffectsChain, EffectsChainConfig, create_effects_chain
from reflow.vocoder import load_model_vocoder
from slicer import Slicer

class InferenceService:
    def __init__(self):
        self.device = DEVICE
        self.current_model = None
        self.current_args = None
        self.current_ckpt = None
        
        # Extractor cache
        self.extractors = {}

    def _get_extractors(self, args):
        """获取或初始化提取器"""
        key = (args.data.f0_extractor, args.data.sampling_rate, args.data.block_size, args.data.encoder)
        if key not in self.extractors:
            print(f"Initializing extractors for: {key}")
            f0_extractor = F0_Extractor(
                args.data.f0_extractor,
                args.data.sampling_rate,
                args.data.block_size,
                args.data.f0_min,
                args.data.f0_max
            )
            volume_extractor = Volume_Extractor(args.data.block_size)
            units_encoder = Units_Encoder(
                args.data.encoder,
                args.data.encoder_ckpt,
                args.data.encoder_sample_rate,
                args.data.encoder_hop_size,
                device=self.device
            )
            self.extractors[key] = (f0_extractor, volume_extractor, units_encoder)
        return self.extractors[key]

    def load_model(self, checkpoint_path: str):
        """加载模型单例"""
        if self.current_ckpt == checkpoint_path and self.current_model is not None:
            return self.current_model, self.current_args

        print(f"Loading DDSP-SVC model from {checkpoint_path}")
        model, args = load_model_vocoder(checkpoint_path, device=self.device)
        self.current_model = model
        self.current_args = args
        self.current_ckpt = checkpoint_path
        return model, args

    async def infer(self,
                    input_id: str,
                    checkpoint_name: str,
                    spk_id: int = 1,
                    pitch_shift: float = 0,
                    f0_extractor_type: str = 'rmvpe',
                    # AudioNoise 增强参数
                    f0_smooth: bool = False,
                    f0_smooth_cutoff: float = 20.0,
                    median_kernel: int = 3,
                    octave_fix: bool = False,
                    vibrato: bool = False,
                    vibrato_rate: float = 5.5,
                    vibrato_depth: float = 0.02,
                    vibrato_delay: float = 0.2,
                    tremolo: bool = False,
                    tremolo_rate: float = 4.0,
                    tremolo_depth: float = 0.1,
                    effects_preset: str = "none",
                    chorus: bool = False,
                    reverb: bool = False,
                    reverb_mix: float = 0.2) -> str:
        """执行音频转换推理"""
        # 1. 寻找输入文件
        input_path = self._find_input_file(input_id)
        checkpoint_path = os.path.join(EXP_DIR, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            # 尝试在子目录找
            found = False
            for root, dirs, files in os.walk(EXP_DIR):
                if checkpoint_name in files:
                    checkpoint_path = os.path.join(root, checkpoint_name)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")

        # 2. 加载模型
        model, args = self.load_model(checkpoint_path)
        f0_extractor, volume_extractor, units_encoder = self._get_extractors(args)

        # 3. 加载音频
        audio, sr = librosa.load(input_path, sr=args.data.sampling_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # 4. 切分处理 (为了性能和稳定性)
        slicer = Slicer(sr=args.data.sampling_rate, threshold=-40, min_length=5000)
        chunks = slicer.slice(audio)
        
        output_audio_segments = []
        
        for k, v in chunks.items():
            tag = v["split_time"].split(",")
            start, end = int(tag[0]), int(tag[1])
            if start == end: continue
            
            segment = audio[start:end]
            if v["slice"]:
                # 静音片段直接填充
                output_audio_segments.append(np.zeros_like(segment))
                continue
                
            # 5. 提取特征
            # F0
            f0 = f0_extractor.extract(segment, uv_interp=True)
            f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(0).unsqueeze(-1)
            f0 = f0 * (2 ** (pitch_shift / 12))

            # F0 平滑处理 (AudioNoise 技术)
            if f0_smooth or octave_fix:
                f0_config = F0ProcessorConfig(
                    enable_smoothing=f0_smooth,
                    cutoff_freq=f0_smooth_cutoff,
                    median_kernel=median_kernel,
                    enable_octave_fix=octave_fix,
                    octave_threshold=1.8
                )
                f0_processor = F0EnhancedProcessor(
                    sample_rate=args.data.sampling_rate,
                    hop_size=args.data.block_size,
                    config=f0_config
                ).to(self.device)
                f0 = f0_processor(f0)

            # Volume
            volume = volume_extractor.extract(segment)
            volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(0).unsqueeze(-1)

            # LFO 调制 (AudioNoise 技术)
            if vibrato or tremolo:
                mod_config = ModulationConfig(
                    enable_vibrato=vibrato,
                    vibrato_rate=vibrato_rate,
                    vibrato_depth=vibrato_depth,
                    vibrato_delay=vibrato_delay,
                    vibrato_fade_in=0.3,
                    enable_tremolo=tremolo,
                    tremolo_rate=tremolo_rate,
                    tremolo_depth=tremolo_depth
                )
                f0, volume = apply_modulation(f0, volume, args.data.sampling_rate, args.data.block_size, mod_config)

            # 移除最后一个维度用于模型输入
            f0 = f0.squeeze(-1) if f0.dim() > 2 else f0
            volume = volume.squeeze(-1) if volume.dim() > 2 else volume
            
            # Units
            audio_t = torch.from_numpy(segment).float().to(self.device).unsqueeze(0)
            units = units_encoder.encode(audio_t, args.data.sampling_rate, args.data.block_size)
            
            # Spk ID
            spk_id_t = torch.LongTensor([spk_id]).to(self.device)
            
            # 6. 推理
            with torch.no_grad():
                out_audio = model(units, f0, volume, spk_id_t)
                out_audio = out_audio.squeeze().cpu().numpy()
                output_audio_segments.append(out_audio)

        # 7. 合并并保存
        res_audio = np.concatenate(output_audio_segments)

        # 音频效果链处理 (AudioNoise 技术)
        if effects_preset != "none" or chorus or reverb:
            if effects_preset != "none":
                effects_chain, fx_config = create_effects_chain(
                    args.data.sampling_rate,
                    preset=effects_preset
                )
            else:
                fx_config = EffectsChainConfig(
                    enable_chorus=chorus,
                    chorus_rate=1.5,
                    chorus_depth=0.4,
                    chorus_mix=0.25,
                    enable_reverb=reverb,
                    reverb_decay=0.5,
                    reverb_damping=0.3,
                    reverb_mix=reverb_mix
                )
                effects_chain = AudioEffectsChain(args.data.sampling_rate, fx_config)

            result_tensor = torch.from_numpy(res_audio).float().unsqueeze(0).to(self.device)
            result_tensor = effects_chain(result_tensor, fx_config)
            res_audio = result_tensor.squeeze().cpu().numpy()

        output_filename = f"out_{input_id}_{os.path.basename(checkpoint_path)}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        sf.write(output_path, res_audio, args.data.sampling_rate)
        
        return output_path

    def _find_input_file(self, input_id: str) -> str:
        # 优先级：直接匹配路径 > ID 解析
        # 1. 检查是否是 "dir:file" 格式
        if ":" in input_id:
            dir_part, file_part = input_id.split(":", 1)
            # 检查分离目录
            sep_path = os.path.join(PROCESSED_DIR, "separated", dir_part, file_part)
            if os.path.exists(sep_path):
                return sep_path
            # 检查切片目录
            slice_path = os.path.join(PROCESSED_DIR, "sliced", dir_part, file_part)
            if os.path.exists(slice_path):
                return slice_path

        # 2. 检查上传目录
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(input_id):
                return os.path.join(UPLOAD_DIR, f)
        
        # 3. 兼容旧逻辑：检查分离目录 (返回 vocal.wav)
        sep_dir = os.path.join(PROCESSED_DIR, "separated")
        if os.path.exists(sep_dir):
            for d in os.listdir(sep_dir):
                if d == input_id:
                    vocal_path = os.path.join(sep_dir, d, "vocal.wav")
                    if os.path.exists(vocal_path):
                        return vocal_path
        
        raise FileNotFoundError(f"Input file for {input_id} not found")

inference_service = InferenceService()
