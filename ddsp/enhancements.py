"""
DDSP-SVC 增强处理模块
整合 AudioNoise 技术到推理流程

功能:
- F0EnhancedProcessor: 增强的 F0 处理器（包含平滑和八度错误修正）
- AudioPostProcessor: 音频后处理器（EQ、效果链）
- EnhancedInference: 增强推理管理器

使用方式:
    # 基本用法 - F0 增强
    from ddsp.enhancements import F0EnhancedProcessor
    f0_processor = F0EnhancedProcessor(sample_rate=44100, hop_size=512)
    f0_smooth = f0_processor.process(f0_raw)

    # 音频后处理
    from ddsp.enhancements import AudioPostProcessor
    post_processor = AudioPostProcessor(sample_rate=44100)
    audio_enhanced = post_processor.process(audio)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# 导入我们实现的模块
from .f0_smoother import (
    AdaptiveF0Smoother,
    MedianF0Smoother,
    CombinedF0Smoother,
    remove_octave_errors
)
from .biquad import (
    BiquadFilter,
    ParametricEQ,
    FormantShifter
)


@dataclass
class F0ProcessorConfig:
    """F0 处理器配置"""
    # 平滑参数
    enable_smoothing: bool = True
    cutoff_freq: float = 20.0  # 低通截止频率 (Hz)
    median_kernel: int = 3     # 中值滤波核大小

    # 八度修正参数
    enable_octave_fix: bool = True
    octave_threshold: float = 1.8

    # 置信度自适应
    use_confidence: bool = False
    confidence_threshold: float = 0.8


@dataclass
class PostProcessorConfig:
    """后处理器配置"""
    # EQ 参数
    enable_eq: bool = False
    low_freq: float = 100.0
    low_gain: float = 0.0      # dB
    high_freq: float = 8000.0
    high_gain: float = 0.0     # dB

    # 共振峰参数
    enable_formant_shift: bool = False
    formant_shift_semitones: float = 0.0

    # 软削波参数
    enable_limiter: bool = False
    limiter_threshold: float = 0.95


class F0EnhancedProcessor(nn.Module):
    """
    增强的 F0 处理器

    整合多种 F0 优化技术:
    - 中值滤波去除孤立异常值
    - IIR 低通滤波平滑轨迹
    - 八度错误自动检测和修正
    - 置信度自适应平滑

    Args:
        sample_rate: 音频采样率 (Hz)
        hop_size: 帧移 (samples)
        config: F0ProcessorConfig 配置对象

    Example:
        >>> processor = F0EnhancedProcessor(44100, 512)
        >>> f0_smooth = processor.process(f0_raw)  # f0_raw: (B, T) 或 (B, T, 1)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512,
                 config: Optional[F0ProcessorConfig] = None):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.config = config or F0ProcessorConfig()

        # 初始化平滑器
        if self.config.enable_smoothing:
            self.smoother = CombinedF0Smoother(
                sample_rate=sample_rate,
                hop_size=hop_size,
                median_kernel=self.config.median_kernel,
                cutoff_freq=self.config.cutoff_freq
            )
        else:
            self.smoother = None

    def process(self,
                f0: torch.Tensor,
                confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        处理 F0 轨迹

        Args:
            f0: F0 序列 (B, T) 或 (B, T, 1)
            confidence: 置信度序列（可选）

        Returns:
            处理后的 F0 轨迹
        """
        # 处理输入形状
        squeeze_last = False
        if f0.dim() == 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
            if confidence is not None and confidence.dim() == 3:
                confidence = confidence.squeeze(-1)
            squeeze_last = True

        f0_processed = f0.clone()

        # 1. 八度错误修正
        if self.config.enable_octave_fix:
            f0_processed = remove_octave_errors(
                f0_processed,
                threshold_ratio=self.config.octave_threshold
            )

        # 2. 平滑处理
        if self.config.enable_smoothing and self.smoother is not None:
            if self.config.use_confidence and confidence is not None:
                # 使用自适应平滑器，支持置信度
                f0_processed = self.smoother.iir_smoother(
                    self.smoother.median_smoother(f0_processed),
                    confidence
                )
            else:
                f0_processed = self.smoother(f0_processed)

        # 恢复形状
        if squeeze_last:
            f0_processed = f0_processed.unsqueeze(-1)

        return f0_processed

    def forward(self,
                f0: torch.Tensor,
                confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """nn.Module 前向传播接口"""
        return self.process(f0, confidence)


class AudioPostProcessor(nn.Module):
    """
    音频后处理器

    提供多种音频增强功能:
    - 参数均衡器 (EQ)
    - 共振峰移位
    - 软削波限幅

    Args:
        sample_rate: 采样率 (Hz)
        config: PostProcessorConfig 配置对象

    Example:
        >>> processor = AudioPostProcessor(44100)
        >>> audio_enhanced = processor.process(audio)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 config: Optional[PostProcessorConfig] = None):
        super().__init__()

        self.sample_rate = sample_rate
        self.config = config or PostProcessorConfig()

        # 初始化处理模块
        if self.config.enable_eq:
            self.eq = ParametricEQ(sample_rate, num_bands=3)
        else:
            self.eq = None

        if self.config.enable_formant_shift:
            self.formant_shifter = FormantShifter(sample_rate)
        else:
            self.formant_shifter = None

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """
        处理音频

        Args:
            audio: 音频信号 (B, T) 或 (T,)

        Returns:
            处理后的音频
        """
        # 处理输入形状
        squeeze_batch = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_batch = True

        audio_processed = audio

        # 1. EQ 处理
        if self.config.enable_eq and self.eq is not None:
            audio_processed = self.eq(
                audio_processed,
                low_freq=self.config.low_freq,
                low_gain=self.config.low_gain,
                high_freq=self.config.high_freq,
                high_gain=self.config.high_gain
            )

        # 2. 共振峰移位
        if self.config.enable_formant_shift and self.formant_shifter is not None:
            if abs(self.config.formant_shift_semitones) > 0.1:
                audio_processed = self.formant_shifter(
                    audio_processed,
                    shift_semitones=self.config.formant_shift_semitones
                )

        # 3. 软削波限幅
        if self.config.enable_limiter:
            audio_processed = self._soft_limit(
                audio_processed,
                threshold=self.config.limiter_threshold
            )

        # 恢复形状
        if squeeze_batch:
            audio_processed = audio_processed.squeeze(0)

        return audio_processed

    def _soft_limit(self,
                    audio: torch.Tensor,
                    threshold: float = 0.95) -> torch.Tensor:
        """
        软削波限幅

        使用多项式近似 tanh，避免硬削波失真
        """
        # 归一化到 [-1, 1]
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        # 应用软削波
        x = audio / threshold
        x2 = x * x
        x4 = x2 * x2
        limited = x * (1.0 - 0.19 * x2 + 0.0162 * x4)
        limited = limited * threshold

        # 恢复原始幅度（略微降低以保护）
        if peak > 0:
            limited = limited * min(peak, 0.99)

        return limited

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """nn.Module 前向传播接口"""
        return self.process(audio)


class EnhancedInference:
    """
    增强推理管理器

    提供完整的增强推理流程管理:
    - F0 预处理
    - 音频后处理
    - 配置管理

    Example:
        >>> enhancer = EnhancedInference(sample_rate=44100, hop_size=512)
        >>> f0_enhanced = enhancer.enhance_f0(f0_raw)
        >>> audio_enhanced = enhancer.enhance_audio(audio)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512,
                 f0_config: Optional[F0ProcessorConfig] = None,
                 post_config: Optional[PostProcessorConfig] = None,
                 device: str = 'cpu'):

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.device = device

        # 初始化处理器
        self.f0_processor = F0EnhancedProcessor(
            sample_rate, hop_size, f0_config
        ).to(device)

        self.audio_processor = AudioPostProcessor(
            sample_rate, post_config
        ).to(device)

    def enhance_f0(self,
                   f0: torch.Tensor,
                   confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """增强 F0 轨迹"""
        f0 = f0.to(self.device)
        if confidence is not None:
            confidence = confidence.to(self.device)
        return self.f0_processor(f0, confidence)

    def enhance_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """增强音频"""
        audio = audio.to(self.device)
        return self.audio_processor(audio)

    def get_config_dict(self) -> Dict[str, Any]:
        """获取当前配置字典"""
        return {
            'f0_processor': {
                'enable_smoothing': self.f0_processor.config.enable_smoothing,
                'cutoff_freq': self.f0_processor.config.cutoff_freq,
                'median_kernel': self.f0_processor.config.median_kernel,
                'enable_octave_fix': self.f0_processor.config.enable_octave_fix,
                'octave_threshold': self.f0_processor.config.octave_threshold,
            },
            'audio_processor': {
                'enable_eq': self.audio_processor.config.enable_eq,
                'low_freq': self.audio_processor.config.low_freq,
                'low_gain': self.audio_processor.config.low_gain,
                'high_freq': self.audio_processor.config.high_freq,
                'high_gain': self.audio_processor.config.high_gain,
                'enable_limiter': self.audio_processor.config.enable_limiter,
            }
        }


def create_default_enhancer(sample_rate: int = 44100,
                            hop_size: int = 512,
                            device: str = 'cpu') -> EnhancedInference:
    """
    创建默认配置的增强器

    默认启用:
    - F0 平滑（中值滤波 + IIR 低通）
    - 八度错误修正
    - 禁用 EQ 和共振峰移位（可手动启用）
    """
    f0_config = F0ProcessorConfig(
        enable_smoothing=True,
        cutoff_freq=20.0,
        median_kernel=3,
        enable_octave_fix=True,
        octave_threshold=1.8
    )

    post_config = PostProcessorConfig(
        enable_eq=False,
        enable_formant_shift=False,
        enable_limiter=False
    )

    return EnhancedInference(
        sample_rate=sample_rate,
        hop_size=hop_size,
        f0_config=f0_config,
        post_config=post_config,
        device=device
    )


# ============ 测试函数 ============

def _test_f0_processor():
    """测试 F0 处理器"""
    print("=" * 50)
    print("F0EnhancedProcessor 测试")
    print("=" * 50)

    processor = F0EnhancedProcessor(44100, 512)

    # 创建带噪声和八度错误的 F0
    f0 = torch.ones(1, 100) * 440
    f0 += torch.randn(1, 100) * 20  # 噪声
    f0[0, 30] = 880  # 八度向上错误
    f0[0, 60] = 220  # 八度向下错误

    f0_enhanced = processor(f0)

    # 检查八度错误是否被修正
    print(f"原始 F0[30]: {f0[0, 30]:.1f} Hz (八度错误)")
    print(f"处理后 F0[30]: {f0_enhanced[0, 30]:.1f} Hz")
    print(f"原始 F0[60]: {f0[0, 60]:.1f} Hz (八度错误)")
    print(f"处理后 F0[60]: {f0_enhanced[0, 60]:.1f} Hz")

    # 检查平滑效果
    noise_before = (f0 - 440).abs().mean()
    noise_after = (f0_enhanced - 440).abs().mean()
    print(f"平滑前平均偏差: {noise_before:.2f} Hz")
    print(f"平滑后平均偏差: {noise_after:.2f} Hz")

    print("✓ F0 处理器测试通过")


def _test_audio_processor():
    """测试音频后处理器"""
    print("\n" + "=" * 50)
    print("AudioPostProcessor 测试")
    print("=" * 50)

    # 启用 EQ 的配置
    config = PostProcessorConfig(
        enable_eq=True,
        low_gain=3.0,
        high_gain=-2.0,
        enable_limiter=True
    )

    processor = AudioPostProcessor(44100, config)

    # 创建测试音频
    t = torch.linspace(0, 1, 44100)
    audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
    audio = audio * 1.2  # 超过 1.0 的幅度

    audio_processed = processor(audio)

    print(f"原始峰值: {audio.abs().max():.4f}")
    print(f"处理后峰值: {audio_processed.abs().max():.4f}")

    assert audio_processed.abs().max() < 1.0, "限幅器应该将峰值限制在 1.0 以下"
    print("✓ 音频后处理器测试通过")


def _test_enhanced_inference():
    """测试完整增强推理流程"""
    print("\n" + "=" * 50)
    print("EnhancedInference 测试")
    print("=" * 50)

    enhancer = create_default_enhancer(44100, 512)

    # 测试 F0 增强
    f0 = torch.ones(1, 100, 1) * 440
    f0[0, 50, 0] = 880  # 八度错误

    f0_enhanced = enhancer.enhance_f0(f0)

    print(f"F0 输入形状: {f0.shape}")
    print(f"F0 输出形状: {f0_enhanced.shape}")

    # 测试音频增强
    audio = torch.randn(44100)
    audio_enhanced = enhancer.enhance_audio(audio)

    print(f"音频输入形状: {audio.shape}")
    print(f"音频输出形状: {audio_enhanced.shape}")

    # 打印配置
    config = enhancer.get_config_dict()
    print(f"当前配置: {config}")

    print("✓ 增强推理测试通过")


def _test_shape_compatibility():
    """测试形状兼容性"""
    print("\n" + "=" * 50)
    print("形状兼容性测试")
    print("=" * 50)

    processor = F0EnhancedProcessor(44100, 512)

    # 测试 (B, T) 形状
    f0_2d = torch.randn(2, 100) + 440
    out_2d = processor(f0_2d)
    assert out_2d.shape == (2, 100), f"2D: {out_2d.shape}"

    # 测试 (B, T, 1) 形状
    f0_3d = torch.randn(2, 100, 1) + 440
    out_3d = processor(f0_3d)
    assert out_3d.shape == (2, 100, 1), f"3D: {out_3d.shape}"

    print("✓ 形状兼容性测试通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  enhancements.py 模块测试")
    print("=" * 60)

    _test_f0_processor()
    _test_audio_processor()
    _test_enhanced_inference()
    _test_shape_compatibility()

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
