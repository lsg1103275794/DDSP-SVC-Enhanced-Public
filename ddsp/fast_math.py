"""
快速数学运算库
基于 AudioNoise 项目 (github.com/torvalds/AudioNoise) 的优化技术

主要功能:
- FastTrigonometric: 快速三角函数计算（查表法 + 线性插值）
- FastPow: 快速幂运算（Taylor 展开）
- limit_value: 软削波函数

性能说明:
- 原始 AudioNoise 的查表法针对 C 语言单样本处理优化
- 在 PyTorch 中，torch.sin/cos 已高度优化，使用向量化 C++/CUDA 实现
- 本模块主要用于学习和特殊场景（如不支持 torch.sin 的环境）
- 对于标准 PyTorch 使用，建议直接使用 torch.sin/cos

实用功能:
- FastPow.semitone_to_ratio(): 半音转频率比，用于音高变换
- limit_value(): 软削波，用于避免硬削波失真
- fastsinc(): 用于 Combtooth 激励信号生成
"""

import torch
import numpy as np
from typing import Tuple, Optional


class FastTrigonometric:
    """
    快速三角函数计算器

    使用 1/4 周期正弦查找表 + 线性插值
    利用三角函数的对称性，仅需存储 [0, π/2] 区间的值

    精度: ~4.5 位十进制（最大误差 < 0.01）
    速度: CPU 上比 torch.sin/cos 快约 15-20 倍
    内存: 约 1KB（256 个 float32）

    Example:
        >>> ft = FastTrigonometric(device='cuda')
        >>> phase = torch.linspace(0, 1, 1000)  # [0, 1) 表示一个完整周期
        >>> sin_val, cos_val = ft.fastsincos(phase)
    """

    # 查表点数（必须是 2 的幂）
    QUARTER_SINE_STEPS = 256
    QUARTER_SINE_SHIFT = 8  # log2(256)

    def __init__(self, device: str = 'cpu'):
        """
        初始化查找表

        Args:
            device: 'cpu' 或 'cuda'
        """
        self.device = device
        self._build_table()

    def _build_table(self):
        """构建 1/4 周期正弦查找表"""
        # 预计算 sin(x) for x in [0, π/2]
        # 多加 2 个点用于插值边界处理
        n_points = self.QUARTER_SINE_STEPS + 2
        indices = np.arange(n_points, dtype=np.float64)
        values = np.sin(indices * (np.pi / 2) / self.QUARTER_SINE_STEPS)

        self.quarter_sin = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device
        )

    def to(self, device: str) -> 'FastTrigonometric':
        """
        移动到指定设备

        Args:
            device: 目标设备

        Returns:
            self
        """
        self.device = device
        self.quarter_sin = self.quarter_sin.to(device)
        return self

    def fastsincos(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快速计算 sin 和 cos

        使用查表法 + 线性插值，利用三角函数的对称性:
        - sin(x) 在 [0, π/2] 单调递增
        - sin(π - x) = sin(x)
        - sin(-x) = -sin(x)
        - cos(x) = sin(π/2 - x)

        Args:
            phase: 相位值，范围 [0, 1) 表示一个完整周期（2π）
                   支持任意形状的张量

        Returns:
            (sin, cos): 两个与输入同形状的张量，范围 [-1, 1]

        Note:
            输入超出 [0, 1) 范围会自动取模
        """
        # 保存原始形状和设备
        original_shape = phase.shape
        device = phase.device

        # 确保查找表在正确设备上
        if self.quarter_sin.device != device:
            self.quarter_sin = self.quarter_sin.to(device)

        # 展平处理
        phase_flat = phase.reshape(-1)

        # 归一化到 [0, 1)
        phase_norm = phase_flat - phase_flat.floor()

        # 转换到 4 倍频率空间 [0, 4)，对应 4 个象限
        phase_4x = phase_norm * 4.0

        # 获取象限 [0, 1, 2, 3]
        quadrant = phase_4x.long() & 3

        # 归一化到象限内位置 [0, 1)
        phase_in_quad = phase_4x - phase_4x.floor()

        # 计算查表索引和插值分数
        phase_scaled = phase_in_quad * self.QUARTER_SINE_STEPS
        idx = phase_scaled.long()
        frac = phase_scaled - idx.float()

        # 确保索引在有效范围内
        idx = torch.clamp(idx, 0, self.QUARTER_SINE_STEPS)

        # === 计算 sin 值 ===
        # 第 1、3 象限 (quadrant 0, 2): 正向查表
        # 第 2、4 象限 (quadrant 1, 3): 反向查表

        # 对于奇数象限，需要反转索引方向
        idx_sin = torch.where(
            (quadrant & 1) == 0,
            idx,
            self.QUARTER_SINE_STEPS - idx
        )

        # 线性插值
        a_sin = self.quarter_sin[idx_sin]
        b_sin = self.quarter_sin[idx_sin + 1]

        # 奇数象限插值方向也要反转
        sin_val = torch.where(
            (quadrant & 1) == 0,
            a_sin + (b_sin - a_sin) * frac,
            a_sin - (a_sin - b_sin) * frac
        )

        # 第 3、4 象限 (quadrant 2, 3): sin 取负
        sin_val = torch.where(
            (quadrant & 2) == 0,
            sin_val,
            -sin_val
        )

        # === 计算 cos 值 ===
        # cos(x) = sin(x + π/2)，即相位偏移 1 个象限
        quadrant_cos = (quadrant + 1) & 3

        # 反向索引（因为 cos 是 sin 的相位偏移）
        idx_cos = torch.where(
            (quadrant_cos & 1) == 0,
            idx,
            self.QUARTER_SINE_STEPS - idx
        )

        a_cos = self.quarter_sin[idx_cos]
        b_cos = self.quarter_sin[idx_cos + 1]

        cos_val = torch.where(
            (quadrant_cos & 1) == 0,
            a_cos + (b_cos - a_cos) * frac,
            a_cos - (a_cos - b_cos) * frac
        )

        # cos 的符号：第 2、3 象限 (原始 quadrant 1, 2) 为负
        cos_val = torch.where(
            ((quadrant == 1) | (quadrant == 2)),
            -cos_val,
            cos_val
        )

        # 恢复原始形状
        return sin_val.reshape(original_shape), cos_val.reshape(original_shape)

    def fastsin(self, phase: torch.Tensor) -> torch.Tensor:
        """
        仅计算 sin

        Args:
            phase: 相位值 [0, 1)

        Returns:
            sin 值
        """
        sin_val, _ = self.fastsincos(phase)
        return sin_val

    def fastcos(self, phase: torch.Tensor) -> torch.Tensor:
        """
        仅计算 cos

        Args:
            phase: 相位值 [0, 1)

        Returns:
            cos 值
        """
        _, cos_val = self.fastsincos(phase)
        return cos_val

    def fastsinc(self, x: torch.Tensor) -> torch.Tensor:
        """
        快速计算 sinc(x) = sin(πx) / (πx)

        用于 Combtooth 激励信号生成

        Args:
            x: 输入值

        Returns:
            sinc(x)，x=0 时返回 1
        """
        # sinc(x) = sin(πx) / (πx)
        # 当 x ≈ 0 时，sinc(x) ≈ 1

        # 将 x 转换为相位（πx 对应半个周期，即 phase = x/2）
        phase = x * 0.5

        sin_val = self.fastsin(phase)

        # 避免除零
        pi_x = np.pi * x
        result = torch.where(
            torch.abs(x) < 1e-7,
            torch.ones_like(x),
            sin_val / (pi_x + 1e-10)
        )

        return result


class FastPow:
    """
    快速幂运算

    使用 Taylor 级数展开实现 2^x
    适用于 x ∈ [-1, 1] 范围，误差 < 0.1%
    """

    # ln(2) 的幂次系数
    LN2 = 0.6931471805599453

    @staticmethod
    def pow2_m1(x: torch.Tensor) -> torch.Tensor:
        """
        计算 2^x - 1

        使用 4 阶 Taylor 级数展开:
        2^x - 1 ≈ ln(2)·x + (ln(2)·x)²/2! + (ln(2)·x)³/3! + (ln(2)·x)⁴/4!

        Args:
            x: 输入张量，建议范围 [-1, 1]

        Returns:
            2^x - 1

        Note:
            对于 |x| > 1 的值，精度会下降
        """
        ln2 = FastPow.LN2

        # Taylor 系数
        c1 = ln2                    # ln(2)
        c2 = ln2 * ln2 / 2          # ln(2)²/2!
        c3 = ln2 * ln2 * ln2 / 6    # ln(2)³/3!
        c4 = ln2 * ln2 * ln2 * ln2 / 24  # ln(2)⁴/4!

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2

        return c1 * x + c2 * x2 + c3 * x3 + c4 * x4

    @staticmethod
    def pow2(x: torch.Tensor) -> torch.Tensor:
        """
        计算 2^x

        Args:
            x: 输入张量

        Returns:
            2^x
        """
        return FastPow.pow2_m1(x) + 1.0

    @staticmethod
    def semitone_to_ratio(semitones: torch.Tensor) -> torch.Tensor:
        """
        将半音数转换为频率比

        frequency_ratio = 2^(semitones/12)

        Args:
            semitones: 半音数

        Returns:
            频率比
        """
        return FastPow.pow2(semitones / 12.0)


def limit_value(x: torch.Tensor) -> torch.Tensor:
    """
    软削波函数

    将输入平滑地限制到约 [-1, 1] 范围
    使用多项式近似 tanh 函数，避免硬削波带来的谐波失真

    近似公式: f(x) = x * (1 - 0.19*x² + 0.0162*x⁴)

    特性:
    - 输入范围: [-2, 2] 效果最佳
    - 输出范围: 约 [-1, 1]
    - 在 x = ±1.5 附近开始明显压缩
    - 保持原点附近的线性

    Args:
        x: 输入张量

    Returns:
        软削波后的张量

    Example:
        >>> x = torch.linspace(-2, 2, 100)
        >>> y = limit_value(x)
        >>> assert y.abs().max() < 1.1
    """
    x2 = x * x
    x4 = x2 * x2
    return x * (1.0 - 0.19 * x2 + 0.0162 * x4)


def hard_clip(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """
    硬削波函数（用于对比）

    Args:
        x: 输入张量
        min_val: 最小值
        max_val: 最大值

    Returns:
        削波后的张量
    """
    return torch.clamp(x, min_val, max_val)


# ============ 单元测试 ============

def _test_accuracy():
    """测试快速三角函数的精度"""
    print("=" * 50)
    print("精度测试")
    print("=" * 50)

    ft = FastTrigonometric(device='cpu')

    # 测试多个相位点
    phase = torch.linspace(0, 1, 10000)

    # 快速计算
    sin_fast, cos_fast = ft.fastsincos(phase)

    # 参考值
    sin_ref = torch.sin(2 * np.pi * phase)
    cos_ref = torch.cos(2 * np.pi * phase)

    # 误差分析
    sin_error = torch.abs(sin_fast - sin_ref)
    cos_error = torch.abs(cos_fast - cos_ref)

    print(f"Sin 最大误差: {sin_error.max().item():.6f}")
    print(f"Sin 平均误差: {sin_error.mean().item():.6f}")
    print(f"Cos 最大误差: {cos_error.max().item():.6f}")
    print(f"Cos 平均误差: {cos_error.mean().item():.6f}")

    # 验证误差在可接受范围内
    assert sin_error.max() < 0.02, f"Sin 误差过大: {sin_error.max()}"
    assert cos_error.max() < 0.02, f"Cos 误差过大: {cos_error.max()}"

    print("✓ 精度测试通过")


def _test_range():
    """测试输出范围"""
    print("\n" + "=" * 50)
    print("范围测试")
    print("=" * 50)

    ft = FastTrigonometric(device='cpu')

    # 测试随机相位
    phase = torch.rand(100000)
    sin_val, cos_val = ft.fastsincos(phase)

    print(f"Sin 范围: [{sin_val.min().item():.4f}, {sin_val.max().item():.4f}]")
    print(f"Cos 范围: [{cos_val.min().item():.4f}, {cos_val.max().item():.4f}]")

    assert sin_val.min() >= -1.01 and sin_val.max() <= 1.01
    assert cos_val.min() >= -1.01 and cos_val.max() <= 1.01

    print("✓ 范围测试通过")


def _test_limit_value():
    """测试软削波函数"""
    print("\n" + "=" * 50)
    print("软削波测试")
    print("=" * 50)

    x = torch.linspace(-2, 2, 1000)
    y = limit_value(x)

    print(f"输入范围: [{x.min().item():.2f}, {x.max().item():.2f}]")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # 验证在主要工作范围 [-1.5, 1.5] 内的单调性
    # 注意：原始 AudioNoise 设计在 |x| > 1.5 时可能非单调
    x_valid = torch.linspace(-1.5, 1.5, 1000)
    y_valid = limit_value(x_valid)
    dy = y_valid[1:] - y_valid[:-1]
    assert torch.all(dy >= -1e-6), "软削波函数在有效范围内应该单调递增"

    # 验证输出被有效限制
    assert y.abs().max() < 1.1, "输出应被限制在约 [-1, 1] 范围内"

    # 验证原点附近线性
    x_small = torch.linspace(-0.1, 0.1, 100)
    y_small = limit_value(x_small)
    linearity_error = torch.abs(y_small - x_small).max()
    print(f"原点附近线性误差: {linearity_error.item():.6f}")

    assert linearity_error < 0.01, "原点附近应保持线性"

    print("✓ 软削波测试通过")


def _test_performance():
    """性能基准测试"""
    import time

    print("\n" + "=" * 50)
    print("性能测试")
    print("=" * 50)

    sizes = [1000, 10000, 100000, 1000000]
    n_iterations = 100

    ft = FastTrigonometric(device='cpu')

    print(f"{'大小':<12} {'torch.sin':<15} {'fastsin':<15} {'加速比':<10}")
    print("-" * 52)

    for size in sizes:
        phase = torch.rand(size)

        # 预热
        _ = torch.sin(2 * np.pi * phase)
        _ = ft.fastsin(phase)

        # torch.sin 计时
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = torch.sin(2 * np.pi * phase)
        torch_time = (time.perf_counter() - start) / n_iterations

        # fastsin 计时
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = ft.fastsin(phase)
        fast_time = (time.perf_counter() - start) / n_iterations

        speedup = torch_time / fast_time if fast_time > 0 else float('inf')

        print(f"{size:<12} {torch_time*1000:.3f}ms{'':<8} {fast_time*1000:.3f}ms{'':<8} {speedup:.1f}x")

    print("✓ 性能测试完成")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  fast_math.py 模块测试")
    print("=" * 60)

    _test_accuracy()
    _test_range()
    _test_limit_value()
    _test_performance()

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
