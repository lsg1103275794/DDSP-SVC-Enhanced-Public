"""
自适应 F0 平滑器
用于减少音高轨迹的抖动，提升合成音质

主要功能:
- AdaptiveF0Smoother: 基于 IIR 低通滤波的 F0 平滑
- MedianF0Smoother: 中值滤波去除异常值
- ExpMovingAverage: 指数移动平均平滑

设计参考:
- AudioNoise 项目的 Biquad 滤波器思想
- DDSP-SVC 的 MaskedAvgPool1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class AdaptiveF0Smoother(nn.Module):
    """
    自适应 F0 平滑器

    使用 Biquad 低通滤波器平滑 F0 轨迹
    可根据置信度自适应调整平滑强度

    Args:
        sample_rate: 音频采样率 (Hz)
        hop_size: 帧移 (samples)
        cutoff_freq: 低通滤波器截止频率 (Hz)，默认 20Hz
        Q: 品质因子，默认 0.707 (Butterworth)

    Example:
        >>> smoother = AdaptiveF0Smoother(44100, 512, cutoff_freq=20)
        >>> f0_smooth = smoother(f0_frames)  # (B, T, 1)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512,
                 cutoff_freq: float = 20.0,
                 Q: float = 0.707):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_rate = sample_rate / hop_size

        # 计算 Biquad 低通滤波器系数
        # 参考: Audio EQ Cookbook
        w0 = 2 * np.pi * cutoff_freq / self.frame_rate
        w0 = min(w0, np.pi * 0.99)  # 防止数值问题

        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2 * Q)

        # LPF 系数
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        # 归一化
        self.register_buffer('b0', torch.tensor(b0 / a0, dtype=torch.float32))
        self.register_buffer('b1', torch.tensor(b1 / a0, dtype=torch.float32))
        self.register_buffer('b2', torch.tensor(b2 / a0, dtype=torch.float32))
        self.register_buffer('a1', torch.tensor(a1 / a0, dtype=torch.float32))
        self.register_buffer('a2', torch.tensor(a2 / a0, dtype=torch.float32))

        # 置信度阈值
        self.confidence_threshold = 0.8

    def forward(self,
                f0_frames: torch.Tensor,
                confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        平滑 F0 轨迹

        Args:
            f0_frames: F0 序列 (B, T) 或 (B, T, 1)
            confidence: 置信度序列，可选
                        高置信度区域保留原值，低置信度区域平滑

        Returns:
            smoothed_f0: 平滑后的 F0，形状同输入
        """
        # 处理输入形状
        squeeze_last = False
        if f0_frames.dim() == 3 and f0_frames.shape[-1] == 1:
            f0_frames = f0_frames.squeeze(-1)
            if confidence is not None and confidence.dim() == 3:
                confidence = confidence.squeeze(-1)
            squeeze_last = True

        B, T = f0_frames.shape
        device = f0_frames.device

        # 确保系数在正确设备上
        if self.b0.device != device:
            self.b0 = self.b0.to(device)
            self.b1 = self.b1.to(device)
            self.b2 = self.b2.to(device)
            self.a1 = self.a1.to(device)
            self.a2 = self.a2.to(device)

        # 初始化滤波器状态
        w1 = torch.zeros(B, device=device)
        w2 = torch.zeros(B, device=device)

        smoothed = torch.zeros_like(f0_frames)

        # Direct Form 2 IIR 滤波
        for t in range(T):
            x = f0_frames[:, t]

            # Biquad 计算
            w0 = x - self.a1 * w1 - self.a2 * w2
            y = self.b0 * w0 + self.b1 * w1 + self.b2 * w2

            # 更新状态
            w2 = w1
            w1 = w0

            # 根据置信度混合原值和平滑值
            if confidence is not None:
                conf = confidence[:, t]
                y = torch.where(conf > self.confidence_threshold, x, y)

            smoothed[:, t] = y

        if squeeze_last:
            smoothed = smoothed.unsqueeze(-1)

        return smoothed

    def forward_causal(self,
                       f0_frames: torch.Tensor,
                       state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                       ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        因果（实时）平滑，保留状态用于流式处理

        Args:
            f0_frames: F0 序列 (B, T)
            state: (w1, w2) 滤波器状态，首次调用传 None

        Returns:
            (smoothed_f0, new_state)
        """
        B, T = f0_frames.shape
        device = f0_frames.device

        if state is None:
            w1 = torch.zeros(B, device=device)
            w2 = torch.zeros(B, device=device)
        else:
            w1, w2 = state

        smoothed = torch.zeros_like(f0_frames)

        for t in range(T):
            x = f0_frames[:, t]
            w0 = x - self.a1 * w1 - self.a2 * w2
            y = self.b0 * w0 + self.b1 * w1 + self.b2 * w2
            w2 = w1
            w1 = w0
            smoothed[:, t] = y

        return smoothed, (w1, w2)


class MedianF0Smoother(nn.Module):
    """
    中值滤波 F0 平滑器

    用于去除孤立的异常值（如八度错误）

    Args:
        kernel_size: 滤波窗口大小，必须为奇数，默认 3
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        self.kernel_size = kernel_size
        self.pad_size = kernel_size // 2

    def forward(self, f0_frames: torch.Tensor) -> torch.Tensor:
        """
        应用中值滤波

        Args:
            f0_frames: F0 序列 (B, T) 或 (B, T, 1)

        Returns:
            smoothed_f0: 中值滤波后的 F0
        """
        squeeze_last = False
        if f0_frames.dim() == 3 and f0_frames.shape[-1] == 1:
            f0_frames = f0_frames.squeeze(-1)
            squeeze_last = True

        B, T = f0_frames.shape

        # 反射填充
        padded = F.pad(f0_frames, (self.pad_size, self.pad_size), mode='reflect')

        # 构建滑动窗口
        windows = padded.unfold(1, self.kernel_size, 1)  # (B, T, K)

        # 计算中值
        smoothed = windows.median(dim=-1).values

        if squeeze_last:
            smoothed = smoothed.unsqueeze(-1)

        return smoothed


class ExpMovingAverage(nn.Module):
    """
    指数移动平均 F0 平滑器

    EMA: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]

    Args:
        alpha: 平滑系数，范围 (0, 1]
               较小的值 -> 更平滑
               较大的值 -> 更快响应
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        assert 0 < alpha <= 1, "alpha 必须在 (0, 1] 范围内"
        self.alpha = alpha

    def forward(self, f0_frames: torch.Tensor) -> torch.Tensor:
        """
        应用指数移动平均

        Args:
            f0_frames: F0 序列 (B, T) 或 (B, T, 1)

        Returns:
            smoothed_f0: 平滑后的 F0
        """
        squeeze_last = False
        if f0_frames.dim() == 3 and f0_frames.shape[-1] == 1:
            f0_frames = f0_frames.squeeze(-1)
            squeeze_last = True

        B, T = f0_frames.shape
        device = f0_frames.device

        smoothed = torch.zeros_like(f0_frames)
        y_prev = f0_frames[:, 0]

        for t in range(T):
            x = f0_frames[:, t]
            y = self.alpha * x + (1 - self.alpha) * y_prev
            smoothed[:, t] = y
            y_prev = y

        if squeeze_last:
            smoothed = smoothed.unsqueeze(-1)

        return smoothed


class CombinedF0Smoother(nn.Module):
    """
    组合 F0 平滑器

    先用中值滤波去除异常值，再用 IIR 低通平滑

    Args:
        sample_rate: 采样率
        hop_size: 帧移
        median_kernel: 中值滤波核大小
        cutoff_freq: 低通截止频率
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 hop_size: int = 512,
                 median_kernel: int = 3,
                 cutoff_freq: float = 20.0):
        super().__init__()

        self.median_smoother = MedianF0Smoother(median_kernel)
        self.iir_smoother = AdaptiveF0Smoother(sample_rate, hop_size, cutoff_freq)

    def forward(self,
                f0_frames: torch.Tensor,
                confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        两阶段平滑
        """
        # 第一阶段：中值滤波去除异常值
        f0_median = self.median_smoother(f0_frames)

        # 第二阶段：IIR 低通平滑
        f0_smooth = self.iir_smoother(f0_median, confidence)

        return f0_smooth


def remove_octave_errors(f0: torch.Tensor,
                         threshold_ratio: float = 1.8) -> torch.Tensor:
    """
    去除八度错误

    检测并修正突然的八度跳变（频率突然翻倍或减半）

    Args:
        f0: F0 序列 (B, T) 或 (B, T, 1)
        threshold_ratio: 判定为八度错误的频率比阈值

    Returns:
        corrected_f0: 修正后的 F0
    """
    squeeze_last = False
    if f0.dim() == 3 and f0.shape[-1] == 1:
        f0 = f0.squeeze(-1)
        squeeze_last = True

    B, T = f0.shape
    corrected = f0.clone()

    for t in range(1, T - 1):
        prev_f0 = corrected[:, t - 1]
        curr_f0 = corrected[:, t]
        next_f0 = f0[:, t + 1]

        # 检测向上八度跳变
        ratio_up = curr_f0 / (prev_f0 + 1e-7)
        is_octave_up = (ratio_up > threshold_ratio) & (ratio_up < 2.2)

        # 检测向下八度跳变
        ratio_down = prev_f0 / (curr_f0 + 1e-7)
        is_octave_down = (ratio_down > threshold_ratio) & (ratio_down < 2.2)

        # 如果前后帧相近，当前帧异常，则修正
        neighbor_ratio = next_f0 / (prev_f0 + 1e-7)
        is_neighbor_similar = (neighbor_ratio > 0.9) & (neighbor_ratio < 1.1)

        # 修正八度向上错误
        should_fix_up = is_octave_up & is_neighbor_similar
        corrected[:, t] = torch.where(should_fix_up, curr_f0 / 2, corrected[:, t])

        # 修正八度向下错误
        should_fix_down = is_octave_down & is_neighbor_similar
        corrected[:, t] = torch.where(should_fix_down, curr_f0 * 2, corrected[:, t])

    if squeeze_last:
        corrected = corrected.unsqueeze(-1)

    return corrected


# ============ 测试函数 ============

def _test_adaptive_smoother():
    """测试自适应平滑器"""
    print("=" * 50)
    print("AdaptiveF0Smoother 测试")
    print("=" * 50)

    smoother = AdaptiveF0Smoother(44100, 512, cutoff_freq=20)

    # 创建带噪声的 F0
    t = torch.linspace(0, 1, 100)
    f0_clean = 440 * torch.ones(1, 100)  # 440Hz
    noise = torch.randn(1, 100) * 30  # 噪声
    f0_noisy = f0_clean + noise

    # 平滑
    f0_smooth = smoother(f0_noisy)

    # 计算噪声减少量
    noise_before = (f0_noisy - f0_clean).abs().mean()
    noise_after = (f0_smooth - f0_clean).abs().mean()
    reduction = (noise_before - noise_after) / noise_before * 100

    print(f"平滑前噪声: {noise_before:.2f} Hz")
    print(f"平滑后噪声: {noise_after:.2f} Hz")
    print(f"噪声减少: {reduction:.1f}%")

    assert noise_after < noise_before, "平滑应该减少噪声"
    print("✓ 测试通过")


def _test_median_smoother():
    """测试中值滤波"""
    print("\n" + "=" * 50)
    print("MedianF0Smoother 测试")
    print("=" * 50)

    smoother = MedianF0Smoother(kernel_size=3)

    # 创建带异常值的 F0
    f0 = torch.ones(1, 10) * 440
    f0[0, 5] = 880  # 八度错误

    f0_smooth = smoother(f0)

    print(f"原始 F0: {f0[0].tolist()}")
    print(f"平滑 F0: {f0_smooth[0].tolist()}")

    # 异常值应该被修正
    assert abs(f0_smooth[0, 5].item() - 440) < 1, "异常值应被修正"
    print("✓ 测试通过")


def _test_octave_removal():
    """测试八度错误去除"""
    print("\n" + "=" * 50)
    print("八度错误去除测试")
    print("=" * 50)

    # 创建带八度错误的 F0
    f0 = torch.ones(1, 10) * 440
    f0[0, 4] = 880  # 向上八度错误
    f0[0, 7] = 220  # 向下八度错误

    f0_fixed = remove_octave_errors(f0)

    print(f"原始 F0: {f0[0].tolist()}")
    print(f"修正 F0: {f0_fixed[0].tolist()}")

    # 检查修正
    assert abs(f0_fixed[0, 4].item() - 440) < 1, "向上八度应被修正"
    assert abs(f0_fixed[0, 7].item() - 440) < 1, "向下八度应被修正"
    print("✓ 测试通过")


def _test_combined_smoother():
    """测试组合平滑器"""
    print("\n" + "=" * 50)
    print("CombinedF0Smoother 测试")
    print("=" * 50)

    smoother = CombinedF0Smoother(44100, 512)

    # 创建带噪声和异常值的 F0
    f0 = torch.ones(1, 100) * 440
    f0 += torch.randn(1, 100) * 20  # 噪声
    f0[0, 30] = 880  # 异常值
    f0[0, 60] = 220  # 异常值

    f0_smooth = smoother(f0)

    # 检查异常值区域
    assert abs(f0_smooth[0, 30].item() - 440) < 50, "异常值应被平滑"
    assert abs(f0_smooth[0, 60].item() - 440) < 50, "异常值应被平滑"

    print("✓ 测试通过")


def _test_shape_compatibility():
    """测试形状兼容性"""
    print("\n" + "=" * 50)
    print("形状兼容性测试")
    print("=" * 50)

    smoother = AdaptiveF0Smoother(44100, 512)

    # 测试 (B, T) 形状
    f0_2d = torch.randn(2, 100)
    out_2d = smoother(f0_2d)
    assert out_2d.shape == (2, 100), f"2D 输出形状错误: {out_2d.shape}"

    # 测试 (B, T, 1) 形状
    f0_3d = torch.randn(2, 100, 1)
    out_3d = smoother(f0_3d)
    assert out_3d.shape == (2, 100, 1), f"3D 输出形状错误: {out_3d.shape}"

    print("✓ 形状兼容性测试通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  f0_smoother.py 模块测试")
    print("=" * 60)

    _test_adaptive_smoother()
    _test_median_smoother()
    _test_octave_removal()
    _test_combined_smoother()
    _test_shape_compatibility()

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
