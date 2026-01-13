"""
LFO (Low Frequency Oscillator) 参数调制模块
基于 AudioNoise 项目 (github.com/torvalds/AudioNoise) 的 LFO 系统

主要功能:
- LFO: 低频振荡器（支持多种波形）
- VibratoModulator: 颤音调制器（调制 F0）
- TremoloModulator: 震音调制器（调制音量）
- FilterModulator: 滤波器调制器（调制截止频率）
- CombinedModulator: 组合调制器

应用场景:
- 为合成声音增加自然的抖动和表现力
- 模拟真实歌唱中的颤音
- 创建动态的音色变化

设计参考:
- AudioNoise lfo.h: 32位相位累加器
- 经典合成器 LFO 设计
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class LFOWaveform(Enum):
    """LFO 波形类型"""
    SINE = "sine"           # 正弦波 - 最自然的调制
    TRIANGLE = "triangle"   # 三角波 - 线性变化
    SAW_UP = "saw_up"       # 上升锯齿波
    SAW_DOWN = "saw_down"   # 下降锯齿波
    SQUARE = "square"       # 方波 - 突变效果
    RANDOM = "random"       # 随机/噪声 - 不规则调制


@dataclass
class LFOConfig:
    """LFO 配置参数"""
    rate: float = 5.0           # 频率 (Hz)，典型范围 0.1-20Hz
    depth: float = 0.5          # 调制深度，范围 0-1
    waveform: str = "sine"      # 波形类型
    phase_offset: float = 0.0   # 初始相位 (0-1)
    delay: float = 0.0          # 延迟时间 (秒)，淡入效果
    fade_in: float = 0.0        # 淡入时间 (秒)


class LFO(nn.Module):
    """
    低频振荡器 (Low Frequency Oscillator)

    使用相位累加器实现高精度 LFO，支持多种波形

    Args:
        sample_rate: 采样率（帧率）
        config: LFO 配置

    Example:
        >>> lfo = LFO(sample_rate=86.13)  # 44100/512 帧率
        >>> modulation = lfo(num_frames=100, rate=5.0, depth=0.02)
    """

    def __init__(self,
                 sample_rate: float = 86.13,  # 默认帧率 44100/512
                 config: Optional[LFOConfig] = None):
        super().__init__()

        self.sample_rate = sample_rate
        self.config = config or LFOConfig()

        # 相位状态（用于流式处理）
        self.register_buffer('phase', torch.zeros(1))

        # 随机波形的状态
        self.register_buffer('random_state', torch.zeros(1))
        self.register_buffer('random_target', torch.zeros(1))

    def reset(self, batch_size: int = 1, device: str = 'cpu'):
        """重置 LFO 状态"""
        self.phase = torch.zeros(batch_size, device=device)
        self.random_state = torch.zeros(batch_size, device=device)
        self.random_target = torch.randn(batch_size, device=device)

    def _generate_waveform(self,
                           phase: torch.Tensor,
                           waveform: str) -> torch.Tensor:
        """
        根据相位生成波形

        Args:
            phase: 相位值 [0, 1)
            waveform: 波形类型

        Returns:
            波形值，范围 [-1, 1]
        """
        if waveform == "sine":
            return torch.sin(2 * np.pi * phase)

        elif waveform == "triangle":
            # 三角波：0->1->0->-1->0
            return 4 * torch.abs(phase - 0.5) - 1

        elif waveform == "saw_up":
            # 上升锯齿：-1 -> 1
            return 2 * phase - 1

        elif waveform == "saw_down":
            # 下降锯齿：1 -> -1
            return 1 - 2 * phase

        elif waveform == "square":
            # 方波
            return torch.where(phase < 0.5,
                             torch.ones_like(phase),
                             -torch.ones_like(phase))

        elif waveform == "random":
            # 平滑随机（使用插值）
            # 这里简化实现，每周期更新目标值
            return self.random_state

        else:
            # 默认正弦波
            return torch.sin(2 * np.pi * phase)

    def forward(self,
                num_frames: int,
                rate: Optional[float] = None,
                depth: Optional[float] = None,
                waveform: Optional[str] = None,
                device: str = 'cpu') -> torch.Tensor:
        """
        生成 LFO 调制信号

        Args:
            num_frames: 帧数
            rate: LFO 频率 (Hz)，覆盖配置值
            depth: 调制深度，覆盖配置值
            waveform: 波形类型，覆盖配置值
            device: 设备

        Returns:
            调制信号 (num_frames,)，范围 [-depth, depth]
        """
        rate = rate if rate is not None else self.config.rate
        depth = depth if depth is not None else self.config.depth
        waveform = waveform if waveform is not None else self.config.waveform

        # 计算相位增量
        phase_increment = rate / self.sample_rate

        # 生成时间序列的相位
        t = torch.arange(num_frames, dtype=torch.float32, device=device)
        phase = (self.config.phase_offset + t * phase_increment) % 1.0

        # 生成波形
        wave = self._generate_waveform(phase, waveform)

        # 应用深度
        modulation = wave * depth

        # 应用延迟和淡入
        if self.config.delay > 0 or self.config.fade_in > 0:
            delay_frames = int(self.config.delay * self.sample_rate)
            fade_frames = int(self.config.fade_in * self.sample_rate)

            envelope = torch.ones(num_frames, device=device)

            # 延迟期间为 0
            if delay_frames > 0:
                envelope[:min(delay_frames, num_frames)] = 0

            # 淡入
            if fade_frames > 0 and delay_frames < num_frames:
                fade_start = delay_frames
                fade_end = min(delay_frames + fade_frames, num_frames)
                fade_range = fade_end - fade_start
                if fade_range > 0:
                    envelope[fade_start:fade_end] = torch.linspace(
                        0, 1, fade_range, device=device
                    )

            modulation = modulation * envelope

        return modulation

    def forward_batch(self,
                      num_frames: int,
                      batch_size: int,
                      rate: Optional[torch.Tensor] = None,
                      depth: Optional[torch.Tensor] = None,
                      device: str = 'cpu') -> torch.Tensor:
        """
        批量生成 LFO 调制信号

        Args:
            num_frames: 帧数
            batch_size: 批大小
            rate: LFO 频率 (B,) 或标量
            depth: 调制深度 (B,) 或标量
            device: 设备

        Returns:
            调制信号 (B, num_frames)
        """
        if rate is None:
            rate = torch.full((batch_size,), self.config.rate, device=device)
        elif not isinstance(rate, torch.Tensor):
            rate = torch.full((batch_size,), rate, device=device)

        if depth is None:
            depth = torch.full((batch_size,), self.config.depth, device=device)
        elif not isinstance(depth, torch.Tensor):
            depth = torch.full((batch_size,), depth, device=device)

        # 计算相位增量 (B,)
        phase_increment = rate / self.sample_rate

        # 生成时间序列 (T,)
        t = torch.arange(num_frames, dtype=torch.float32, device=device)

        # 计算相位 (B, T)
        phase = (self.config.phase_offset +
                 t.unsqueeze(0) * phase_increment.unsqueeze(1)) % 1.0

        # 生成波形
        wave = self._generate_waveform(phase, self.config.waveform)

        # 应用深度 (B, T)
        modulation = wave * depth.unsqueeze(1)

        return modulation


class VibratoModulator(nn.Module):
    """
    颤音调制器

    对 F0 进行周期性调制，模拟自然歌唱中的颤音

    典型参数:
    - 频率: 4-7 Hz（自然颤音范围）
    - 深度: 1-3%（即 ±12-36 音分）

    Args:
        sample_rate: 音频采样率
        hop_size: 帧移

    Example:
        >>> vibrato = VibratoModulator(44100, 512)
        >>> f0_modulated = vibrato(f0, rate=5.5, depth=0.02)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_rate = sample_rate / hop_size

        # 创建 LFO
        self.lfo = LFO(sample_rate=self.frame_rate)

        # 默认颤音参数
        self.default_rate = 5.5      # Hz
        self.default_depth = 0.02    # 2% = ±24 音分
        self.default_delay = 0.2     # 200ms 延迟
        self.default_fade_in = 0.3   # 300ms 淡入

    def forward(self,
                f0: torch.Tensor,
                rate: float = None,
                depth: float = None,
                delay: float = None,
                fade_in: float = None,
                waveform: str = "sine") -> torch.Tensor:
        """
        对 F0 应用颤音调制

        Args:
            f0: F0 序列 (B, T) 或 (B, T, 1)
            rate: 颤音频率 (Hz)
            depth: 调制深度（比例，如 0.02 表示 ±2%）
            delay: 延迟时间 (秒)
            fade_in: 淡入时间 (秒)
            waveform: 波形类型

        Returns:
            调制后的 F0，形状同输入
        """
        rate = rate if rate is not None else self.default_rate
        depth = depth if depth is not None else self.default_depth
        delay = delay if delay is not None else self.default_delay
        fade_in = fade_in if fade_in is not None else self.default_fade_in

        # 处理输入形状
        squeeze_last = False
        if f0.dim() == 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
            squeeze_last = True

        B, T = f0.shape
        device = f0.device

        # 更新 LFO 配置
        self.lfo.config.delay = delay
        self.lfo.config.fade_in = fade_in

        # 生成调制信号
        modulation = self.lfo(T, rate=rate, depth=depth,
                             waveform=waveform, device=device)

        # 应用调制: f0_new = f0 * (1 + modulation)
        f0_modulated = f0 * (1 + modulation.unsqueeze(0))

        if squeeze_last:
            f0_modulated = f0_modulated.unsqueeze(-1)

        return f0_modulated

    def forward_adaptive(self,
                         f0: torch.Tensor,
                         rate: torch.Tensor = None,
                         depth: torch.Tensor = None) -> torch.Tensor:
        """
        自适应颤音调制（每帧可变参数）

        Args:
            f0: F0 序列 (B, T)
            rate: 颤音频率序列 (T,) 或标量
            depth: 调制深度序列 (T,) 或标量

        Returns:
            调制后的 F0
        """
        B, T = f0.shape
        device = f0.device

        if rate is None:
            rate = torch.full((T,), self.default_rate, device=device)
        elif rate.dim() == 0:
            rate = rate.expand(T)

        if depth is None:
            depth = torch.full((T,), self.default_depth, device=device)
        elif depth.dim() == 0:
            depth = depth.expand(T)

        # 累积相位
        phase_increment = rate / self.frame_rate
        phase = torch.cumsum(phase_increment, dim=0) % 1.0

        # 生成调制
        modulation = torch.sin(2 * np.pi * phase) * depth

        # 应用调制
        f0_modulated = f0 * (1 + modulation.unsqueeze(0))

        return f0_modulated


class TremoloModulator(nn.Module):
    """
    震音调制器

    对音量进行周期性调制，产生抖动效果

    典型参数:
    - 频率: 3-8 Hz
    - 深度: 5-30%

    Args:
        sample_rate: 音频采样率
        hop_size: 帧移
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_rate = sample_rate / hop_size

        self.lfo = LFO(sample_rate=self.frame_rate)

        # 默认震音参数
        self.default_rate = 4.0      # Hz
        self.default_depth = 0.1     # 10%

    def forward(self,
                volume: torch.Tensor,
                rate: float = None,
                depth: float = None,
                waveform: str = "sine") -> torch.Tensor:
        """
        对音量应用震音调制

        Args:
            volume: 音量序列 (B, T) 或 (B, T, 1)
            rate: 震音频率 (Hz)
            depth: 调制深度
            waveform: 波形类型

        Returns:
            调制后的音量
        """
        rate = rate if rate is not None else self.default_rate
        depth = depth if depth is not None else self.default_depth

        squeeze_last = False
        if volume.dim() == 3 and volume.shape[-1] == 1:
            volume = volume.squeeze(-1)
            squeeze_last = True

        B, T = volume.shape
        device = volume.device

        # 生成调制信号（震音通常使用单极性调制）
        modulation = self.lfo(T, rate=rate, depth=depth,
                             waveform=waveform, device=device)

        # 震音调制: volume_new = volume * (1 - depth + depth * (mod + 1) / 2)
        # 这样确保调制范围是 [1-depth, 1]
        mod_factor = 1 - depth * (1 - (modulation / depth + 1) / 2)
        volume_modulated = volume * mod_factor.unsqueeze(0)

        if squeeze_last:
            volume_modulated = volume_modulated.unsqueeze(-1)

        return volume_modulated


class FilterModulator(nn.Module):
    """
    滤波器调制器

    调制滤波器截止频率，产生 "wah-wah" 或扫频效果

    Args:
        sample_rate: 音频采样率
        hop_size: 帧移
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_rate = sample_rate / hop_size

        self.lfo = LFO(sample_rate=self.frame_rate)

        # 默认参数
        self.default_rate = 0.5      # Hz (慢速扫频)
        self.default_center = 1000   # Hz (中心频率)
        self.default_range = 500     # Hz (调制范围)

    def forward(self,
                num_frames: int,
                rate: float = None,
                center_freq: float = None,
                freq_range: float = None,
                waveform: str = "triangle",
                device: str = 'cpu') -> torch.Tensor:
        """
        生成滤波器截止频率序列

        Args:
            num_frames: 帧数
            rate: 调制频率 (Hz)
            center_freq: 中心频率 (Hz)
            freq_range: 调制范围 (Hz)
            waveform: 波形类型
            device: 设备

        Returns:
            截止频率序列 (num_frames,)
        """
        rate = rate if rate is not None else self.default_rate
        center_freq = center_freq if center_freq is not None else self.default_center
        freq_range = freq_range if freq_range is not None else self.default_range

        # 生成调制信号
        modulation = self.lfo(num_frames, rate=rate, depth=1.0,
                             waveform=waveform, device=device)

        # 计算截止频率
        cutoff = center_freq + modulation * freq_range

        # 确保频率在有效范围内
        cutoff = torch.clamp(cutoff, 20, self.sample_rate / 2 - 100)

        return cutoff


class CombinedModulator(nn.Module):
    """
    组合调制器

    整合颤音、震音等多种调制效果

    Args:
        sample_rate: 音频采样率
        hop_size: 帧移

    Example:
        >>> modulator = CombinedModulator(44100, 512)
        >>> f0_mod, vol_mod = modulator(f0, volume,
        ...                              vibrato_rate=5.5, vibrato_depth=0.02,
        ...                              tremolo_rate=4.0, tremolo_depth=0.1)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size

        self.vibrato = VibratoModulator(sample_rate, hop_size)
        self.tremolo = TremoloModulator(sample_rate, hop_size)
        self.filter_mod = FilterModulator(sample_rate, hop_size)

    def forward(self,
                f0: torch.Tensor,
                volume: torch.Tensor,
                vibrato_rate: float = 5.5,
                vibrato_depth: float = 0.02,
                vibrato_delay: float = 0.2,
                vibrato_fade_in: float = 0.3,
                tremolo_rate: float = 0.0,
                tremolo_depth: float = 0.0,
                enable_vibrato: bool = True,
                enable_tremolo: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用组合调制

        Args:
            f0: F0 序列
            volume: 音量序列
            vibrato_*: 颤音参数
            tremolo_*: 震音参数
            enable_vibrato: 是否启用颤音
            enable_tremolo: 是否启用震音

        Returns:
            (f0_modulated, volume_modulated)
        """
        f0_out = f0
        volume_out = volume

        if enable_vibrato and vibrato_depth > 0:
            f0_out = self.vibrato(
                f0_out,
                rate=vibrato_rate,
                depth=vibrato_depth,
                delay=vibrato_delay,
                fade_in=vibrato_fade_in
            )

        if enable_tremolo and tremolo_depth > 0:
            volume_out = self.tremolo(
                volume_out,
                rate=tremolo_rate,
                depth=tremolo_depth
            )

        return f0_out, volume_out


@dataclass
class ModulationConfig:
    """调制配置（用于推理集成）"""
    # 颤音参数
    enable_vibrato: bool = False
    vibrato_rate: float = 5.5       # Hz
    vibrato_depth: float = 0.02     # 2%
    vibrato_delay: float = 0.2      # 秒
    vibrato_fade_in: float = 0.3    # 秒

    # 震音参数
    enable_tremolo: bool = False
    tremolo_rate: float = 4.0       # Hz
    tremolo_depth: float = 0.1      # 10%


def apply_modulation(f0: torch.Tensor,
                     volume: torch.Tensor,
                     sample_rate: int,
                     hop_size: int,
                     config: ModulationConfig = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用调制效果（便捷函数）

    Args:
        f0: F0 序列 (B, T, 1)
        volume: 音量序列 (B, T, 1)
        sample_rate: 采样率
        hop_size: 帧移
        config: 调制配置

    Returns:
        (f0_modulated, volume_modulated)
    """
    if config is None:
        config = ModulationConfig()

    if not config.enable_vibrato and not config.enable_tremolo:
        return f0, volume

    modulator = CombinedModulator(sample_rate, hop_size)

    return modulator(
        f0, volume,
        vibrato_rate=config.vibrato_rate,
        vibrato_depth=config.vibrato_depth,
        vibrato_delay=config.vibrato_delay,
        vibrato_fade_in=config.vibrato_fade_in,
        tremolo_rate=config.tremolo_rate,
        tremolo_depth=config.tremolo_depth,
        enable_vibrato=config.enable_vibrato,
        enable_tremolo=config.enable_tremolo
    )


# ============ 测试函数 ============

def _test_lfo_basic():
    """测试基础 LFO"""
    print("=" * 50)
    print("LFO 基础测试")
    print("=" * 50)

    lfo = LFO(sample_rate=86.13)

    # 测试各种波形
    waveforms = ["sine", "triangle", "saw_up", "saw_down", "square"]

    for wf in waveforms:
        mod = lfo(100, rate=5.0, depth=0.5, waveform=wf)
        print(f"{wf:12s}: min={mod.min():.3f}, max={mod.max():.3f}, shape={mod.shape}")

        # 验证范围
        assert mod.min() >= -0.51 and mod.max() <= 0.51, f"{wf} 范围错误"

    print("✓ LFO 基础测试通过")


def _test_vibrato():
    """测试颤音调制"""
    print("\n" + "=" * 50)
    print("颤音调制测试")
    print("=" * 50)

    vibrato = VibratoModulator(44100, 512)

    # 创建恒定 F0
    f0 = torch.ones(1, 100, 1) * 440

    # 应用颤音
    f0_mod = vibrato(f0, rate=5.5, depth=0.02, delay=0.0, fade_in=0.0)

    # 检查调制范围
    f0_mod_squeezed = f0_mod.squeeze()
    min_f0 = f0_mod_squeezed.min().item()
    max_f0 = f0_mod_squeezed.max().item()

    print(f"原始 F0: 440 Hz")
    print(f"调制后范围: {min_f0:.1f} - {max_f0:.1f} Hz")
    print(f"调制深度: ±{(max_f0 - 440) / 440 * 100:.1f}%")

    # 验证调制深度约为 ±2%
    assert 430 < min_f0 < 440, "最小 F0 应该略低于 440"
    assert 440 < max_f0 < 450, "最大 F0 应该略高于 440"

    print("✓ 颤音调制测试通过")


def _test_tremolo():
    """测试震音调制"""
    print("\n" + "=" * 50)
    print("震音调制测试")
    print("=" * 50)

    tremolo = TremoloModulator(44100, 512)

    # 创建恒定音量
    volume = torch.ones(1, 100, 1)

    # 应用震音
    vol_mod = tremolo(volume, rate=4.0, depth=0.2)

    vol_squeezed = vol_mod.squeeze()
    min_vol = vol_squeezed.min().item()
    max_vol = vol_squeezed.max().item()

    print(f"原始音量: 1.0")
    print(f"调制后范围: {min_vol:.3f} - {max_vol:.3f}")

    # 震音应该使音量在 [1-depth, 1] 范围内变化
    assert 0.7 < min_vol < 0.85, f"最小音量应约为 0.8, 实际 {min_vol}"
    assert 0.95 < max_vol <= 1.01, f"最大音量应约为 1.0, 实际 {max_vol}"

    print("✓ 震音调制测试通过")


def _test_combined():
    """测试组合调制"""
    print("\n" + "=" * 50)
    print("组合调制测试")
    print("=" * 50)

    modulator = CombinedModulator(44100, 512)

    f0 = torch.ones(1, 200, 1) * 440
    volume = torch.ones(1, 200, 1)

    f0_mod, vol_mod = modulator(
        f0, volume,
        vibrato_rate=5.5, vibrato_depth=0.02,
        vibrato_delay=0.0, vibrato_fade_in=0.0,
        tremolo_rate=4.0, tremolo_depth=0.1,
        enable_vibrato=True,
        enable_tremolo=True
    )

    print(f"F0 输入形状: {f0.shape}")
    print(f"F0 输出形状: {f0_mod.shape}")
    print(f"Volume 输入形状: {volume.shape}")
    print(f"Volume 输出形状: {vol_mod.shape}")

    assert f0_mod.shape == f0.shape
    assert vol_mod.shape == volume.shape

    print("✓ 组合调制测试通过")


def _test_delay_and_fade():
    """测试延迟和淡入"""
    print("\n" + "=" * 50)
    print("延迟和淡入测试")
    print("=" * 50)

    vibrato = VibratoModulator(44100, 512)

    f0 = torch.ones(1, 200, 1) * 440

    # 应用带延迟和淡入的颤音
    # 帧率约 86.13，0.5秒约 43 帧
    f0_mod = vibrato(f0, rate=5.5, depth=0.05, delay=0.5, fade_in=0.5)

    f0_diff = (f0_mod - f0).abs().squeeze()

    # 检查延迟期间无调制
    delay_region = f0_diff[:40].mean()
    # 检查完全调制区域
    full_region = f0_diff[150:].mean()

    print(f"延迟区域平均偏差: {delay_region:.4f}")
    print(f"完全调制区域平均偏差: {full_region:.4f}")

    assert delay_region < 0.1, "延迟期间应该几乎无调制"
    assert full_region > 1.0, "完全调制区域应该有明显调制"

    print("✓ 延迟和淡入测试通过")


def _test_convenience_function():
    """测试便捷函数"""
    print("\n" + "=" * 50)
    print("便捷函数测试")
    print("=" * 50)

    f0 = torch.ones(1, 100, 1) * 440
    volume = torch.ones(1, 100, 1)

    config = ModulationConfig(
        enable_vibrato=True,
        vibrato_rate=5.5,
        vibrato_depth=0.02,
        vibrato_delay=0.0,
        vibrato_fade_in=0.0
    )

    f0_mod, vol_mod = apply_modulation(f0, volume, 44100, 512, config)

    print(f"配置: {config}")
    print(f"F0 变化范围: {f0_mod.min():.1f} - {f0_mod.max():.1f} Hz")

    print("✓ 便捷函数测试通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  lfo.py 模块测试")
    print("=" * 60)

    _test_lfo_basic()
    _test_vibrato()
    _test_tremolo()
    _test_combined()
    _test_delay_and_fade()
    _test_convenience_function()

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
