"""
环形缓冲区模块
基于 AudioNoise 项目 (github.com/torvalds/AudioNoise) 的优化技术

主要功能:
- RingBuffer: 高性能环形缓冲区（2的幂大小，位运算优化）
- DelayLine: 延迟线（支持分数延迟）
- MultiTapDelay: 多抽头延迟
- FeedbackDelay: 带反馈的延迟效果
- PingPongDelay: 乒乓延迟（立体声）

应用场景:
- 延迟效果 (Delay/Echo)
- 混响预延迟
- Chorus/Flanger 效果
- 实时流式处理

设计参考:
- AudioNoise 的位运算优化技术
- 2的幂大小确保位与运算代替取模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


def next_power_of_2(n: int) -> int:
    """返回大于等于 n 的最小 2 的幂"""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


class RingBuffer(nn.Module):
    """
    高性能环形缓冲区

    使用 2 的幂大小和位运算优化，避免取模运算

    AudioNoise 优化技术:
    - 缓冲区大小必须是 2 的幂
    - 使用 & (size-1) 代替 % size
    - 位运算比取模快约 10x

    Args:
        max_delay_samples: 最大延迟样本数
        num_channels: 通道数（默认 1）

    Example:
        >>> rb = RingBuffer(max_delay_samples=44100)
        >>> rb.write(input_sample)
        >>> output = rb.read(delay=1000)  # 读取 1000 样本前的数据
    """

    def __init__(self,
                 max_delay_samples: int,
                 num_channels: int = 1):
        super().__init__()

        # 确保大小是 2 的幂
        self.buffer_size = next_power_of_2(max_delay_samples + 1)
        self.mask = self.buffer_size - 1  # 用于位与运算
        self.num_channels = num_channels

        # 缓冲区
        self.register_buffer(
            'buffer',
            torch.zeros(num_channels, self.buffer_size)
        )

        # 写指针
        self.register_buffer(
            'write_index',
            torch.zeros(1, dtype=torch.long)
        )

    def reset(self, device: str = 'cpu'):
        """重置缓冲区"""
        self.buffer = torch.zeros(
            self.num_channels, self.buffer_size, device=device
        )
        self.write_index = torch.zeros(1, dtype=torch.long, device=device)

    def write(self, samples: torch.Tensor):
        """
        写入样本

        Args:
            samples: 输入样本 (num_channels,) 或 (num_channels, num_samples)
        """
        if samples.dim() == 1:
            samples = samples.unsqueeze(-1)

        num_samples = samples.shape[-1]

        for i in range(num_samples):
            idx = (self.write_index.item() + i) & self.mask
            self.buffer[:, idx] = samples[:, i]

        self.write_index = (self.write_index + num_samples) & self.mask

    def read(self, delay: int) -> torch.Tensor:
        """
        读取延迟后的样本

        Args:
            delay: 延迟样本数

        Returns:
            延迟后的样本 (num_channels,)
        """
        read_idx = (self.write_index.item() - delay - 1) & self.mask
        return self.buffer[:, read_idx]

    def read_interpolated(self, delay: float) -> torch.Tensor:
        """
        读取分数延迟的样本（线性插值）

        Args:
            delay: 延迟样本数（可以是小数）

        Returns:
            插值后的样本
        """
        delay_int = int(delay)
        delay_frac = delay - delay_int

        # 读取两个相邻样本
        idx1 = (self.write_index.item() - delay_int - 1) & self.mask
        idx2 = (self.write_index.item() - delay_int - 2) & self.mask

        sample1 = self.buffer[:, idx1]
        sample2 = self.buffer[:, idx2]

        # 线性插值
        return sample1 * (1 - delay_frac) + sample2 * delay_frac

    def read_block(self, delay: int, num_samples: int) -> torch.Tensor:
        """
        批量读取延迟后的样本块

        Args:
            delay: 延迟样本数
            num_samples: 读取的样本数

        Returns:
            样本块 (num_channels, num_samples)
        """
        output = torch.zeros(
            self.num_channels, num_samples,
            device=self.buffer.device
        )

        for i in range(num_samples):
            read_idx = (self.write_index.item() - delay - num_samples + i) & self.mask
            output[:, i] = self.buffer[:, read_idx]

        return output


class DelayLine(nn.Module):
    """
    延迟线

    基于环形缓冲区的延迟效果实现

    Args:
        max_delay_ms: 最大延迟时间（毫秒）
        sample_rate: 采样率

    Example:
        >>> delay = DelayLine(max_delay_ms=1000, sample_rate=44100)
        >>> output = delay(input_audio, delay_ms=500, feedback=0.3)
    """

    def __init__(self,
                 max_delay_ms: float = 1000.0,
                 sample_rate: int = 44100):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)

        self.ring_buffer = RingBuffer(self.max_delay_samples)

    def reset(self, device: str = 'cpu'):
        """重置延迟线"""
        self.ring_buffer.reset(device)

    def forward(self,
                x: torch.Tensor,
                delay_ms: float = 500.0,
                feedback: float = 0.0,
                mix: float = 0.5) -> torch.Tensor:
        """
        应用延迟效果

        Args:
            x: 输入信号 (T,) 或 (B, T)
            delay_ms: 延迟时间（毫秒）
            feedback: 反馈量（0-1）
            mix: 干湿混合比（0=全干，1=全湿）

        Returns:
            延迟后的信号
        """
        squeeze_batch = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, T = x.shape
        device = x.device

        # 计算延迟样本数
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        delay_samples = min(delay_samples, self.max_delay_samples)

        # 初始化输出
        output = torch.zeros_like(x)

        # 确保缓冲区在正确设备上
        if self.ring_buffer.buffer.device != device:
            self.ring_buffer.reset(device)

        # 逐样本处理（单通道简化版）
        for b in range(B):
            self.ring_buffer.reset(device)

            for t in range(T):
                # 读取延迟样本
                delayed = self.ring_buffer.read(delay_samples)

                # 计算输出（干湿混合）
                dry = x[b, t]
                wet = delayed[0]
                output[b, t] = dry * (1 - mix) + wet * mix

                # 写入缓冲区（包含反馈）
                self.ring_buffer.write(
                    (dry + wet * feedback).unsqueeze(0)
                )

        if squeeze_batch:
            output = output.squeeze(0)

        return output


class MultiTapDelay(nn.Module):
    """
    多抽头延迟

    同时输出多个不同延迟时间的信号，用于创建丰富的延迟效果

    Args:
        max_delay_ms: 最大延迟时间（毫秒）
        sample_rate: 采样率
        num_taps: 抽头数量
    """

    def __init__(self,
                 max_delay_ms: float = 1000.0,
                 sample_rate: int = 44100,
                 num_taps: int = 4):
        super().__init__()

        self.sample_rate = sample_rate
        self.num_taps = num_taps
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)

        self.ring_buffer = RingBuffer(self.max_delay_samples)

        # 默认抽头延迟（均匀分布）
        self.default_delays = [
            max_delay_ms * (i + 1) / num_taps
            for i in range(num_taps)
        ]

        # 默认抽头增益（递减）
        self.default_gains = [
            0.8 ** (i + 1)
            for i in range(num_taps)
        ]

    def forward(self,
                x: torch.Tensor,
                delays_ms: Optional[List[float]] = None,
                gains: Optional[List[float]] = None,
                mix: float = 0.5) -> torch.Tensor:
        """
        应用多抽头延迟

        Args:
            x: 输入信号 (T,)
            delays_ms: 各抽头延迟时间列表
            gains: 各抽头增益列表
            mix: 干湿混合比

        Returns:
            处理后的信号
        """
        if delays_ms is None:
            delays_ms = self.default_delays
        if gains is None:
            gains = self.default_gains

        T = x.shape[-1]
        device = x.device

        # 转换为样本数
        delay_samples = [
            min(int(d * self.sample_rate / 1000), self.max_delay_samples)
            for d in delays_ms
        ]

        output = torch.zeros_like(x)
        self.ring_buffer.reset(device)

        for t in range(T):
            # 累加所有抽头
            wet = 0.0
            for i, (delay, gain) in enumerate(zip(delay_samples, gains)):
                delayed = self.ring_buffer.read(delay)
                wet += delayed[0] * gain

            # 干湿混合
            dry = x[t] if x.dim() == 1 else x[0, t]
            output_val = dry * (1 - mix) + wet * mix

            if x.dim() == 1:
                output[t] = output_val
            else:
                output[0, t] = output_val

            # 写入缓冲区
            self.ring_buffer.write(dry.unsqueeze(0).unsqueeze(0))

        return output


class FeedbackDelay(nn.Module):
    """
    带反馈的延迟效果

    支持高通/低通滤波器在反馈路径中，模拟真实的模拟延迟

    Args:
        max_delay_ms: 最大延迟时间
        sample_rate: 采样率
    """

    def __init__(self,
                 max_delay_ms: float = 1000.0,
                 sample_rate: int = 44100):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)

        self.ring_buffer = RingBuffer(self.max_delay_samples)

        # 反馈滤波器状态
        self.register_buffer('filter_state', torch.zeros(1))

    def forward(self,
                x: torch.Tensor,
                delay_ms: float = 500.0,
                feedback: float = 0.5,
                damping: float = 0.3,
                mix: float = 0.5) -> torch.Tensor:
        """
        应用带滤波的反馈延迟

        Args:
            x: 输入信号
            delay_ms: 延迟时间（毫秒）
            feedback: 反馈量（0-0.95）
            damping: 高频衰减量（0-1，模拟磁带延迟）
            mix: 干湿混合比

        Returns:
            处理后的信号
        """
        # 限制反馈防止失控
        feedback = min(feedback, 0.95)

        T = x.shape[-1] if x.dim() > 0 else 1
        device = x.device

        delay_samples = min(
            int(delay_ms * self.sample_rate / 1000),
            self.max_delay_samples
        )

        output = torch.zeros_like(x)
        self.ring_buffer.reset(device)
        filter_state = torch.zeros(1, device=device)

        for t in range(T):
            # 读取延迟样本
            delayed = self.ring_buffer.read(delay_samples)[0]

            # 一阶低通滤波（damping）
            filtered = delayed * (1 - damping) + filter_state * damping
            filter_state = filtered

            # 干湿混合
            dry = x[t] if x.dim() == 1 else x[0, t]
            wet = filtered
            output_val = dry * (1 - mix) + wet * mix

            if x.dim() == 1:
                output[t] = output_val
            else:
                output[0, t] = output_val

            # 写入缓冲区（带反馈）
            self.ring_buffer.write(
                (dry + filtered * feedback).unsqueeze(0).unsqueeze(0)
            )

        return output


class BatchDelayProcessor(nn.Module):
    """
    批量延迟处理器

    针对批量音频处理优化，使用向量化操作

    Args:
        max_delay_ms: 最大延迟时间
        sample_rate: 采样率
    """

    def __init__(self,
                 max_delay_ms: float = 1000.0,
                 sample_rate: int = 44100):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.buffer_size = next_power_of_2(self.max_delay_samples + 1)

    def forward(self,
                x: torch.Tensor,
                delay_ms: float = 500.0,
                feedback: float = 0.3,
                mix: float = 0.5) -> torch.Tensor:
        """
        批量延迟处理（向量化实现）

        Args:
            x: 输入信号 (B, T) 或 (T,)
            delay_ms: 延迟时间
            feedback: 反馈量
            mix: 干湿混合比

        Returns:
            处理后的信号
        """
        squeeze_batch = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, T = x.shape
        device = x.device

        delay_samples = min(
            int(delay_ms * self.sample_rate / 1000),
            self.max_delay_samples
        )

        # 使用 F.pad 实现简单延迟
        if feedback == 0:
            # 无反馈：简单的时移
            delayed = F.pad(x, (delay_samples, 0))[:, :T]
            output = x * (1 - mix) + delayed * mix
        else:
            # 有反馈：需要迭代处理
            output = torch.zeros_like(x)
            buffer = torch.zeros(B, self.buffer_size, device=device)
            write_idx = 0

            for t in range(T):
                # 读取
                read_idx = (write_idx - delay_samples) % self.buffer_size
                delayed = buffer[:, read_idx]

                # 混合
                output[:, t] = x[:, t] * (1 - mix) + delayed * mix

                # 写入（带反馈）
                buffer[:, write_idx] = x[:, t] + delayed * feedback
                write_idx = (write_idx + 1) % self.buffer_size

        if squeeze_batch:
            output = output.squeeze(0)

        return output


@dataclass
class DelayConfig:
    """延迟效果配置"""
    enable: bool = False
    delay_ms: float = 500.0
    feedback: float = 0.3
    damping: float = 0.2
    mix: float = 0.3


def apply_delay(audio: torch.Tensor,
                sample_rate: int,
                config: DelayConfig = None) -> torch.Tensor:
    """
    应用延迟效果（便捷函数）

    Args:
        audio: 输入音频 (B, T) 或 (T,)
        sample_rate: 采样率
        config: 延迟配置

    Returns:
        处理后的音频
    """
    if config is None:
        config = DelayConfig()

    if not config.enable:
        return audio

    processor = BatchDelayProcessor(
        max_delay_ms=config.delay_ms * 2,
        sample_rate=sample_rate
    )

    return processor(
        audio,
        delay_ms=config.delay_ms,
        feedback=config.feedback,
        mix=config.mix
    )


# ============ 测试函数 ============

def _test_power_of_2():
    """测试 2 的幂计算"""
    print("=" * 50)
    print("2的幂计算测试")
    print("=" * 50)

    test_cases = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 1000]

    for n in test_cases:
        p = next_power_of_2(n)
        print(f"{n:5d} -> {p:5d} (2^{int(np.log2(p))})")
        assert p >= n and (p & (p - 1)) == 0, f"不是2的幂: {p}"

    print("✓ 2的幂计算测试通过")


def _test_ring_buffer():
    """测试环形缓冲区"""
    print("\n" + "=" * 50)
    print("环形缓冲区测试")
    print("=" * 50)

    rb = RingBuffer(max_delay_samples=100)

    print(f"缓冲区大小: {rb.buffer_size} (mask: {rb.mask})")

    # 写入测试数据
    for i in range(10):
        rb.write(torch.tensor([[float(i)]]))

    # 读取验证
    for delay in [1, 5, 9]:
        val = rb.read(delay)
        expected = 9 - delay
        print(f"延迟 {delay}: 读取 {val.item():.0f}, 预期 {expected}")
        assert abs(val.item() - expected) < 0.001

    print("✓ 环形缓冲区测试通过")


def _test_delay_line():
    """测试延迟线"""
    print("\n" + "=" * 50)
    print("延迟线测试")
    print("=" * 50)

    delay = DelayLine(max_delay_ms=100, sample_rate=1000)

    # 创建脉冲信号
    x = torch.zeros(200)
    x[0] = 1.0

    # 应用 50ms 延迟
    y = delay(x, delay_ms=50, feedback=0.0, mix=1.0)

    # 找到脉冲位置
    peak_idx = torch.argmax(y).item()
    print(f"脉冲位置: 输入 0, 输出 {peak_idx}")
    print(f"预期延迟: 50 样本")

    # 允许 ±1 样本误差
    assert 49 <= peak_idx <= 51, f"延迟不正确: {peak_idx}"

    print("✓ 延迟线测试通过")


def _test_feedback_delay():
    """测试反馈延迟"""
    print("\n" + "=" * 50)
    print("反馈延迟测试")
    print("=" * 50)

    delay = FeedbackDelay(max_delay_ms=100, sample_rate=1000)

    # 创建脉冲信号
    x = torch.zeros(500)
    x[0] = 1.0

    # 应用带反馈的延迟
    y = delay(x, delay_ms=50, feedback=0.5, damping=0.2, mix=1.0)

    # 检查是否有多次回声
    peaks = (y > 0.1).sum().item()
    print(f"检测到 {peaks} 个显著峰值（反馈回声）")

    assert peaks > 1, "应该有多次反馈回声"

    print("✓ 反馈延迟测试通过")


def _test_batch_delay():
    """测试批量延迟处理"""
    print("\n" + "=" * 50)
    print("批量延迟处理测试")
    print("=" * 50)

    processor = BatchDelayProcessor(max_delay_ms=100, sample_rate=44100)

    # 批量输入
    x = torch.randn(4, 44100)

    y = processor(x, delay_ms=50, feedback=0.3, mix=0.5)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    assert y.shape == x.shape, "输出形状应与输入相同"

    print("✓ 批量延迟处理测试通过")


def _test_multi_tap():
    """测试多抽头延迟"""
    print("\n" + "=" * 50)
    print("多抽头延迟测试")
    print("=" * 50)

    delay = MultiTapDelay(max_delay_ms=200, sample_rate=1000, num_taps=4)

    x = torch.zeros(500)
    x[0] = 1.0

    y = delay(x, mix=1.0)

    # 检查多个抽头
    peaks = (y.abs() > 0.05).sum().item()
    print(f"检测到 {peaks} 个抽头峰值")

    assert peaks >= 4, "应该有至少4个抽头峰值"

    print("✓ 多抽头延迟测试通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  ring_buffer.py 模块测试")
    print("=" * 60)

    _test_power_of_2()
    _test_ring_buffer()
    _test_delay_line()
    _test_feedback_delay()
    _test_batch_delay()
    _test_multi_tap()

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
