"""
Biquad IIR 滤波器模块
基于 AudioNoise 项目 (github.com/torvalds/AudioNoise) 的实现

主要功能:
- BiquadFilter: 单个 Biquad 滤波器（支持 LPF/HPF/BPF/Notch/AllPass/Peaking/LowShelf/HighShelf）
- BiquadFilterChain: 多级滤波器链（用于 Phaser 等效果）
- ParametricEQ: 参数均衡器

设计参考:
- Audio EQ Cookbook by Robert Bristow-Johnson
- AudioNoise biquad.h

应用场景:
- 音色雕塑
- 共振峰调整
- Phaser/Flanger 效果
- 实时均衡器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal, List, Tuple
from dataclasses import dataclass


@dataclass
class BiquadCoeffs:
    """Biquad 滤波器系数"""
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def to_tensor(self, device='cpu') -> torch.Tensor:
        """转换为张量 [b0, b1, b2, a1, a2]"""
        return torch.tensor([self.b0, self.b1, self.b2, self.a1, self.a2],
                           dtype=torch.float32, device=device)


# 滤波器类型
FilterType = Literal['lpf', 'hpf', 'bpf', 'notch', 'allpass',
                     'peaking', 'lowshelf', 'highshelf']


def compute_biquad_coeffs(filter_type: FilterType,
                          sample_rate: float,
                          freq: float,
                          Q: float = 0.707,
                          gain_db: float = 0.0) -> BiquadCoeffs:
    """
    计算 Biquad 滤波器系数

    参考: Audio EQ Cookbook by Robert Bristow-Johnson

    Args:
        filter_type: 滤波器类型
        sample_rate: 采样率 (Hz)
        freq: 截止/中心频率 (Hz)
        Q: 品质因子（默认 0.707 = Butterworth）
        gain_db: 增益 (dB)，仅用于 peaking/shelf 类型

    Returns:
        BiquadCoeffs: 归一化后的滤波器系数
    """
    # 限制频率范围，防止数值问题
    freq = np.clip(freq, 20, sample_rate / 2 - 100)
    Q = np.clip(Q, 0.1, 30)

    # 计算中间变量
    w0 = 2 * np.pi * freq / sample_rate
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2 * Q)

    # 用于 shelf 和 peaking
    A = 10 ** (gain_db / 40)  # sqrt(10^(dB/20))

    if filter_type == 'lpf':
        # 低通滤波器
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == 'hpf':
        # 高通滤波器
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == 'bpf':
        # 带通滤波器（恒定峰值增益）
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == 'notch':
        # 陷波滤波器
        b0 = 1
        b1 = -2 * cos_w0
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == 'allpass':
        # 全通滤波器
        b0 = 1 - alpha
        b1 = -2 * cos_w0
        b2 = 1 + alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == 'peaking':
        # 峰值均衡器
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

    elif filter_type == 'lowshelf':
        # 低频搁架
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    elif filter_type == 'highshelf':
        # 高频搁架
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    else:
        raise ValueError(f"未知滤波器类型: {filter_type}")

    # 归一化
    return BiquadCoeffs(
        b0=b0 / a0,
        b1=b1 / a0,
        b2=b2 / a0,
        a1=a1 / a0,
        a2=a2 / a0
    )


class BiquadFilter(nn.Module):
    """
    Biquad IIR 滤波器

    使用 Direct Form 2 Transposed 实现，数值稳定性好

    Args:
        filter_type: 滤波器类型
        sample_rate: 采样率 (Hz)

    Example:
        >>> filt = BiquadFilter('lpf', 44100)
        >>> y = filt(x, freq=1000, Q=0.707)
    """

    def __init__(self,
                 filter_type: FilterType = 'lpf',
                 sample_rate: int = 44100):
        super().__init__()
        self.filter_type = filter_type
        self.sample_rate = sample_rate

        # 滤波器状态
        self.register_buffer('w1', torch.zeros(1))
        self.register_buffer('w2', torch.zeros(1))

    def reset_state(self, batch_size: int = 1, device: str = 'cpu'):
        """重置滤波器状态"""
        self.w1 = torch.zeros(batch_size, device=device)
        self.w2 = torch.zeros(batch_size, device=device)

    def forward(self,
                x: torch.Tensor,
                freq: float,
                Q: float = 0.707,
                gain_db: float = 0.0) -> torch.Tensor:
        """
        应用滤波器

        Args:
            x: 输入信号 (B, T) 或 (T,)
            freq: 截止/中心频率 (Hz)
            Q: 品质因子
            gain_db: 增益 (dB)，仅用于 peaking/shelf

        Returns:
            y: 滤波后的信号
        """
        # 处理输入形状
        squeeze_batch = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, T = x.shape
        device = x.device

        # 计算滤波器系数
        coeffs = compute_biquad_coeffs(
            self.filter_type, self.sample_rate, freq, Q, gain_db
        )

        # 确保状态张量形状正确
        if self.w1.shape[0] != B or self.w1.device != device:
            self.reset_state(B, device)

        # Direct Form 2 Transposed 实现
        y = torch.zeros_like(x)
        w1, w2 = self.w1.clone(), self.w2.clone()

        b0, b1, b2 = coeffs.b0, coeffs.b1, coeffs.b2
        a1, a2 = coeffs.a1, coeffs.a2

        for t in range(T):
            x_t = x[:, t]
            y_t = b0 * x_t + w1
            w1 = b1 * x_t - a1 * y_t + w2
            w2 = b2 * x_t - a2 * y_t
            y[:, t] = y_t

        # 保存状态
        self.w1 = w1
        self.w2 = w2

        if squeeze_batch:
            y = y.squeeze(0)

        return y

    def forward_batch(self,
                      x: torch.Tensor,
                      freqs: torch.Tensor,
                      Qs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        时变参数批量处理（每帧使用不同的频率/Q）

        Args:
            x: 输入信号 (B, T)
            freqs: 频率序列 (T,) 或标量
            Qs: Q 值序列 (T,)，可选

        Returns:
            y: 滤波后的信号
        """
        B, T = x.shape
        device = x.device

        if Qs is None:
            Qs = torch.full((T,), 0.707, device=device)

        if freqs.dim() == 0:
            freqs = freqs.expand(T)

        self.reset_state(B, device)
        y = torch.zeros_like(x)
        w1, w2 = self.w1, self.w2

        for t in range(T):
            # 计算当前帧的系数
            coeffs = compute_biquad_coeffs(
                self.filter_type, self.sample_rate,
                freqs[t].item(), Qs[t].item()
            )

            x_t = x[:, t]
            y_t = coeffs.b0 * x_t + w1
            w1 = coeffs.b1 * x_t - coeffs.a1 * y_t + w2
            w2 = coeffs.b2 * x_t - coeffs.a2 * y_t
            y[:, t] = y_t

        return y


class BiquadFilterChain(nn.Module):
    """
    多级 Biquad 滤波器链

    可用于 Phaser 效果或复杂频响塑形

    Args:
        num_stages: 级联数量
        filter_type: 滤波器类型
        sample_rate: 采样率
    """

    def __init__(self,
                 num_stages: int = 4,
                 filter_type: FilterType = 'allpass',
                 sample_rate: int = 44100):
        super().__init__()
        self.num_stages = num_stages
        self.filter_type = filter_type
        self.sample_rate = sample_rate

        # 创建滤波器链
        self.filters = nn.ModuleList([
            BiquadFilter(filter_type, sample_rate)
            for _ in range(num_stages)
        ])

    def reset_all_states(self, batch_size: int = 1, device: str = 'cpu'):
        """重置所有滤波器状态"""
        for filt in self.filters:
            filt.reset_state(batch_size, device)

    def forward(self,
                x: torch.Tensor,
                freq: float,
                Q: float = 0.707) -> torch.Tensor:
        """
        应用滤波器链

        Args:
            x: 输入信号 (B, T)
            freq: 所有级使用相同的频率
            Q: 所有级使用相同的 Q

        Returns:
            y: 滤波后的信号
        """
        y = x
        for filt in self.filters:
            y = filt(y, freq, Q)
        return y

    def forward_with_freqs(self,
                           x: torch.Tensor,
                           freqs: List[float],
                           Qs: Optional[List[float]] = None) -> torch.Tensor:
        """
        每级使用不同的频率/Q

        Args:
            x: 输入信号
            freqs: 每级的频率列表
            Qs: 每级的 Q 列表
        """
        if Qs is None:
            Qs = [0.707] * self.num_stages

        assert len(freqs) == self.num_stages
        assert len(Qs) == self.num_stages

        y = x
        for i, filt in enumerate(self.filters):
            y = filt(y, freqs[i], Qs[i])
        return y


class ParametricEQ(nn.Module):
    """
    参数均衡器

    包含低频搁架、多个峰值频段、高频搁架

    Args:
        sample_rate: 采样率
        num_bands: 峰值频段数量
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 num_bands: int = 3):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_bands = num_bands

        # 低频搁架
        self.low_shelf = BiquadFilter('lowshelf', sample_rate)
        # 峰值频段
        self.bands = nn.ModuleList([
            BiquadFilter('peaking', sample_rate)
            for _ in range(num_bands)
        ])
        # 高频搁架
        self.high_shelf = BiquadFilter('highshelf', sample_rate)

    def forward(self,
                x: torch.Tensor,
                low_freq: float = 100,
                low_gain: float = 0,
                band_freqs: Optional[List[float]] = None,
                band_gains: Optional[List[float]] = None,
                band_Qs: Optional[List[float]] = None,
                high_freq: float = 8000,
                high_gain: float = 0) -> torch.Tensor:
        """
        应用均衡器

        Args:
            x: 输入信号
            low_freq: 低频搁架频率
            low_gain: 低频增益 (dB)
            band_freqs: 各频段中心频率
            band_gains: 各频段增益 (dB)
            band_Qs: 各频段 Q 值
            high_freq: 高频搁架频率
            high_gain: 高频增益 (dB)

        Returns:
            y: 均衡后的信号
        """
        if band_freqs is None:
            band_freqs = [300, 1000, 3000][:self.num_bands]
        if band_gains is None:
            band_gains = [0] * self.num_bands
        if band_Qs is None:
            band_Qs = [1.0] * self.num_bands

        y = x

        # 低频搁架
        if abs(low_gain) > 0.1:
            y = self.low_shelf(y, low_freq, Q=0.707, gain_db=low_gain)

        # 峰值频段
        for i, band in enumerate(self.bands):
            if abs(band_gains[i]) > 0.1:
                y = band(y, band_freqs[i], band_Qs[i], band_gains[i])

        # 高频搁架
        if abs(high_gain) > 0.1:
            y = self.high_shelf(y, high_freq, Q=0.707, gain_db=high_gain)

        return y


class FormantShifter(nn.Module):
    """
    共振峰移位器

    使用多个峰值滤波器模拟共振峰移位

    Args:
        sample_rate: 采样率
        num_formants: 共振峰数量
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 num_formants: int = 3):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_formants = num_formants

        # 典型元音共振峰频率 (Hz)
        # F1: 200-800, F2: 800-2500, F3: 2500-3500
        self.default_formants = [500, 1500, 2500][:num_formants]

        self.filters = nn.ModuleList([
            BiquadFilter('peaking', sample_rate)
            for _ in range(num_formants)
        ])

    def forward(self,
                x: torch.Tensor,
                shift_semitones: float = 0,
                formant_freqs: Optional[List[float]] = None) -> torch.Tensor:
        """
        应用共振峰移位

        Args:
            x: 输入信号
            shift_semitones: 移位量（半音）
            formant_freqs: 原始共振峰频率

        Returns:
            y: 处理后的信号
        """
        if formant_freqs is None:
            formant_freqs = self.default_formants

        # 计算移位后的频率
        shift_ratio = 2 ** (shift_semitones / 12)
        shifted_freqs = [f * shift_ratio for f in formant_freqs]

        y = x
        for i, filt in enumerate(self.filters):
            # 先衰减原始共振峰
            # 再增强新位置
            # 简化实现：直接在新位置增强
            y = filt(y, shifted_freqs[i], Q=2.0, gain_db=3.0)

        return y


# ============ 测试函数 ============

def _test_lpf():
    """测试低通滤波器"""
    print("=" * 50)
    print("LPF 测试")
    print("=" * 50)

    filt = BiquadFilter('lpf', 44100)

    # 创建测试信号：440Hz + 4400Hz
    t = torch.linspace(0, 1, 44100)
    signal = torch.sin(2 * np.pi * 440 * t) + torch.sin(2 * np.pi * 4400 * t)
    signal = signal.unsqueeze(0)

    # 1000Hz 低通
    filtered = filt(signal, freq=1000)

    # 高频成分应该被衰减
    # 计算能量比
    energy_before = (signal ** 2).mean()
    energy_after = (filtered ** 2).mean()

    print(f"滤波前能量: {energy_before:.4f}")
    print(f"滤波后能量: {energy_after:.4f}")
    print(f"能量比: {(energy_after/energy_before):.2%}")

    # 低通后能量应该减少（高频被滤除）
    assert energy_after < energy_before, "低通滤波应减少能量"
    print("✓ 测试通过")


def _test_hpf():
    """测试高通滤波器"""
    print("\n" + "=" * 50)
    print("HPF 测试")
    print("=" * 50)

    filt = BiquadFilter('hpf', 44100)

    # 低频 + 高频信号
    t = torch.linspace(0, 1, 44100)
    signal = torch.sin(2 * np.pi * 100 * t) + torch.sin(2 * np.pi * 2000 * t)
    signal = signal.unsqueeze(0)

    # 500Hz 高通
    filtered = filt(signal, freq=500)

    # 检查低频被衰减
    energy_ratio = (filtered ** 2).mean() / (signal ** 2).mean()
    print(f"能量比: {energy_ratio:.2%}")

    assert energy_ratio < 1.0, "高通滤波应减少低频能量"
    print("✓ 测试通过")


def _test_allpass_chain():
    """测试全通滤波器链"""
    print("\n" + "=" * 50)
    print("AllPass Chain 测试")
    print("=" * 50)

    chain = BiquadFilterChain(num_stages=4, filter_type='allpass')

    t = torch.linspace(0, 1, 44100)
    signal = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)

    filtered = chain(signal, freq=1000, Q=1.0)

    # 全通滤波器应该保持幅度
    energy_before = (signal ** 2).mean()
    energy_after = (filtered ** 2).mean()
    energy_ratio = energy_after / energy_before

    print(f"能量比: {energy_ratio:.4f}")

    # 全通滤波器能量应该接近 1
    assert 0.9 < energy_ratio < 1.1, "全通滤波器应保持能量"
    print("✓ 测试通过")


def _test_parametric_eq():
    """测试参数均衡器"""
    print("\n" + "=" * 50)
    print("ParametricEQ 测试")
    print("=" * 50)

    eq = ParametricEQ(44100, num_bands=3)

    t = torch.linspace(0, 1, 44100)
    signal = torch.randn(1, 44100) * 0.5

    # 应用均衡
    equalized = eq(
        signal,
        low_freq=100, low_gain=3,
        band_freqs=[500, 1000, 3000],
        band_gains=[2, -2, 1],
        band_Qs=[1.5, 2.0, 1.0],
        high_freq=8000, high_gain=-3
    )

    assert equalized.shape == signal.shape, "输出形状应与输入相同"
    print(f"输入形状: {signal.shape}")
    print(f"输出形状: {equalized.shape}")
    print("✓ 测试通过")


def _test_shape_compatibility():
    """测试形状兼容性"""
    print("\n" + "=" * 50)
    print("形状兼容性测试")
    print("=" * 50)

    filt = BiquadFilter('lpf', 44100)

    # 1D 输入
    x1d = torch.randn(1000)
    y1d = filt(x1d, 1000)
    assert y1d.shape == x1d.shape, f"1D: {y1d.shape} != {x1d.shape}"

    # 2D 输入
    x2d = torch.randn(2, 1000)
    filt.reset_state(2, x2d.device)
    y2d = filt(x2d, 1000)
    assert y2d.shape == x2d.shape, f"2D: {y2d.shape} != {x2d.shape}"

    print("✓ 形状兼容性测试通过")


def _test_coefficients():
    """测试滤波器系数计算"""
    print("\n" + "=" * 50)
    print("系数计算测试")
    print("=" * 50)

    # 测试各种滤波器类型
    filter_types = ['lpf', 'hpf', 'bpf', 'notch', 'allpass', 'peaking', 'lowshelf', 'highshelf']

    for ft in filter_types:
        try:
            coeffs = compute_biquad_coeffs(ft, 44100, 1000, Q=1.0, gain_db=3.0)
            print(f"{ft:10s}: b0={coeffs.b0:.4f}, b1={coeffs.b1:.4f}, b2={coeffs.b2:.4f}, "
                  f"a1={coeffs.a1:.4f}, a2={coeffs.a2:.4f}")
        except Exception as e:
            print(f"{ft}: 错误 - {e}")
            raise

    print("✓ 系数计算测试通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  biquad.py 模块测试")
    print("=" * 60)

    _test_coefficients()
    _test_lpf()
    _test_hpf()
    _test_allpass_chain()
    _test_parametric_eq()
    _test_shape_compatibility()

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
