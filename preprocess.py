import os
import numpy as np
import random
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import Vocoder
from logger.utils import traverse_dir
import concurrent.futures
from typing import Optional, Callable, Dict, List
from datetime import datetime


class AudioPreprocessor:
    """
    音频预处理器 - 可被 API 服务调用的类接口

    Features:
    - 特征提取器单例模式（避免重复加载）
    - Skip 目录机制（失败文件自动归档）
    - 断点续传（跳过已处理文件）
    - 错误日志记录
    - 进度回调支持
    - 数据集验证
    """

    def __init__(self, config_path: str, device: str = None):
        """
        初始化音频预处理器

        Args:
            config_path: 配置文件路径 (configs/reflow.yaml)
            device: 计算设备 ('cuda', 'cpu', 或 None 自动检测)
        """
        self.config_path = config_path
        self.args = utils.load_config(config_path)

        # 设备检测
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 从配置中读取参数
        self.sample_rate = self.args.data.sampling_rate
        self.hop_size = self.args.data.block_size
        self.extensions = self.args.data.extensions

        # 初始化特征提取器（单例模式）
        self._init_extractors()

        print(f"[AudioPreprocessor] Initialized on device: {self.device}")
        print(f"[AudioPreprocessor] Sample rate: {self.sample_rate}, Hop size: {self.hop_size}")

    def _init_extractors(self):
        """初始化所有特征提取器"""
        # F0 提取器
        self.f0_extractor = F0_Extractor(
            self.args.data.f0_extractor,
            self.args.data.sampling_rate,
            self.args.data.block_size,
            self.args.data.f0_min,
            self.args.data.f0_max
        )

        # Volume 提取器
        self.volume_extractor = Volume_Extractor(
            self.args.data.block_size,
            self.args.data.volume_smooth_size
        )

        # Mel 提取器
        self.mel_extractor = None
        try:
            mel_vocoder = Vocoder(
                self.args.vocoder.type,
                self.args.vocoder.ckpt,
                device=self.device
            )
            # 检查参数匹配
            if (mel_vocoder.vocoder_sample_rate == self.sample_rate and
                mel_vocoder.vocoder_hop_size == self.hop_size):
                self.mel_extractor = mel_vocoder
                print("[AudioPreprocessor] Mel extractor loaded successfully")
            else:
                print("[AudioPreprocessor] Vocoder parameters mismatch, mel extraction disabled")
        except Exception as e:
            print(f"[AudioPreprocessor] Failed to load mel extractor: {e}")

        # Units 编码器
        cnhubertsoft_gate = (
            self.args.data.cnhubertsoft_gate
            if self.args.data.encoder == 'cnhubertsoftfish'
            else 10
        )
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=self.device
        )

    def _is_processed(self, path: str, file: str) -> bool:
        """
        检查文件是否已经处理过（用于断点续传）

        Args:
            path: 数据集根目录
            file: 文件名（不含扩展名）

        Returns:
            bool: True 表示已处理，False 表示未处理
        """
        binfile = file + '.npy'
        required_files = [
            os.path.join(path, 'units', binfile),
            os.path.join(path, 'f0', binfile),
            os.path.join(path, 'volume', binfile),
        ]
        return all(os.path.exists(f) for f in required_files)

    def _log_error(self, path: str, file: str, error: str):
        """
        记录错误到日志文件

        Args:
            path: 数据集根目录
            file: 文件名
            error: 错误信息
        """
        error_log_path = os.path.join(path, 'errors.log')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {file}: {error}\n")

    def preprocess_dataset(
        self,
        path: str,
        use_pitch_aug: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        resume: bool = True,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        预处理数据集（核心方法）

        Args:
            path: 数据集根目录（应包含 audio/ 子目录）
            use_pitch_aug: 是否启用音高增强（±5 半音随机偏移）
            progress_callback: 进度回调函数 (current, total, message)
            resume: 是否启用断点续传（跳过已处理文件）
            extensions: 音频文件扩展名列表（默认从配置读取）

        Returns:
            dict: 处理结果统计
                - total: 总文件数
                - processed: 成功处理数
                - skipped: 跳过数（断点续传）
                - failed: 失败数
                - failed_files: 失败文件列表
        """
        # 路径定义
        path_srcdir = os.path.join(path, 'audio')
        path_unitsdir = os.path.join(path, 'units')
        path_f0dir = os.path.join(path, 'f0')
        path_volumedir = os.path.join(path, 'volume')
        path_augvoldir = os.path.join(path, 'aug_vol')
        path_meldir = os.path.join(path, 'mel')
        path_augmeldir = os.path.join(path, 'aug_mel')
        path_skipdir = os.path.join(path, 'skip')

        # 确保 skip 目录存在
        os.makedirs(path_skipdir, exist_ok=True)

        # 获取文件列表
        if extensions is None:
            extensions = self.extensions

        filelist = traverse_dir(
            path_srcdir,
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )

        if not filelist:
            raise ValueError(f"No audio files found in {path_srcdir}")

        print(f"[AudioPreprocessor] Processing {len(filelist)} files in: {path_srcdir}")
        print(f"[AudioPreprocessor] Pitch augmentation: {use_pitch_aug}")
        print(f"[AudioPreprocessor] Resume mode: {resume}")

        # 统计信息
        stats = {
            'total': len(filelist),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'failed_files': []
        }

        # Pitch augmentation 字典
        pitch_aug_dict = {}

        # 处理函数
        def process_file(file: str, index: int):
            binfile = file + '.npy'
            path_srcfile = os.path.join(path_srcdir, file)
            path_unitsfile = os.path.join(path_unitsdir, binfile)
            path_f0file = os.path.join(path_f0dir, binfile)
            path_volumefile = os.path.join(path_volumedir, binfile)
            path_augvolfile = os.path.join(path_augvoldir, binfile)
            path_melfile = os.path.join(path_meldir, binfile)
            path_augmelfile = os.path.join(path_augmeldir, binfile)
            path_skipfile = os.path.join(path_skipdir, file)

            # 断点续传：检查是否已处理
            if resume and self._is_processed(path, file):
                stats['skipped'] += 1
                if progress_callback:
                    progress_callback(index, len(filelist), f"跳过已处理: {file}")
                return True

            try:
                # 加载音频
                audio, _ = librosa.load(path_srcfile, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio_t = torch.from_numpy(audio).float().to(self.device)
                audio_t = audio_t.unsqueeze(0)

                # 提取 volume
                volume = self.volume_extractor.extract(audio)

                # 提取 mel 和 volume augmentation
                keyshift = 0
                if self.mel_extractor is not None:
                    mel_t = self.mel_extractor.extract(audio_t, self.sample_rate)
                    mel = mel_t.squeeze().to('cpu').numpy()

                    max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
                    max_shift = min(1, np.log10(1/max_amp))
                    log10_vol_shift = random.uniform(-1, max_shift)

                    if use_pitch_aug:
                        keyshift = random.uniform(-5, 5)

                    aug_mel_t = self.mel_extractor.extract(
                        audio_t * (10 ** log10_vol_shift),
                        self.sample_rate,
                        keyshift=keyshift
                    )
                    aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
                    aug_vol = self.volume_extractor.extract(audio * (10 ** log10_vol_shift))

                # 提取 units
                units_t = self.units_encoder.encode(audio_t, self.sample_rate, self.hop_size)
                units = units_t.squeeze().to('cpu').numpy()

                # 提取 F0
                f0 = self.f0_extractor.extract(audio, uv_interp=False)

                # 检查 F0 是否有效
                uv = f0 == 0
                if len(f0[~uv]) == 0:
                    raise ValueError("F0 extraction failed: all frames are unvoiced")

                # 插值 F0
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

                # 保存特征
                os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
                np.save(path_unitsfile, units)

                os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
                np.save(path_f0file, f0)

                os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
                np.save(path_volumefile, volume)

                if self.mel_extractor is not None:
                    pitch_aug_dict[file] = keyshift
                    os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                    np.save(path_melfile, mel)
                    os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                    np.save(path_augmelfile, aug_mel)
                    os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                    np.save(path_augvolfile, aug_vol)

                stats['processed'] += 1
                if progress_callback:
                    progress_callback(index, len(filelist), f"完成: {file}")
                return True

            except Exception as e:
                # 错误处理：移动到 skip 目录并记录日志
                error_msg = str(e)
                print(f"\n[Error] Failed to process {file}: {error_msg}")

                # 移动到 skip 目录
                try:
                    if os.path.exists(path_srcfile):
                        shutil.move(path_srcfile, path_skipfile)
                        print(f"[Skip] Moved to: {path_skipfile}")
                except Exception as move_error:
                    print(f"[Error] Failed to move file: {move_error}")

                # 记录错误日志
                self._log_error(path, file, error_msg)

                stats['failed'] += 1
                stats['failed_files'].append({'file': file, 'error': error_msg})

                if progress_callback:
                    progress_callback(index, len(filelist), f"失败: {file} - {error_msg}")
                return False

        # 处理所有文件（使用 tqdm 进度条）
        print("\n[AudioPreprocessor] Starting feature extraction...")
        for i, file in enumerate(tqdm(filelist, desc="Processing", total=len(filelist))):
            process_file(file, i)

        # 保存 pitch augmentation 字典
        if self.mel_extractor is not None and pitch_aug_dict:
            path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
            np.save(path_pitchaugdict, pitch_aug_dict)
            print(f"[AudioPreprocessor] Saved pitch augmentation dict: {path_pitchaugdict}")

        # 打印统计信息
        print(f"\n[AudioPreprocessor] Processing complete!")
        print(f"  Total files: {stats['total']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped (resumed): {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")

        if stats['failed'] > 0:
            print(f"\n[Warning] {stats['failed']} files failed. Check {path}/errors.log for details.")
            print(f"[Warning] Failed files moved to: {path_skipdir}")

        return stats

    def validate_dataset(self, path: str) -> Dict[str, any]:
        """
        预处理前验证数据集

        Args:
            path: 数据集根目录

        Returns:
            dict: 验证结果
                - valid: 是否通过验证
                - total_files: 总文件数
                - valid_files: 合规文件数
                - invalid_files: 不合规文件列表（最多10个）
                - warnings: 警告信息列表
                - estimated_time: 预计处理时间（秒）
        """
        path_srcdir = os.path.join(path, 'audio')

        result = {
            'valid': True,
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': [],
            'warnings': [],
            'estimated_time': 0
        }

        # 检查目录是否存在
        if not os.path.exists(path_srcdir):
            result['valid'] = False
            result['warnings'].append(f"Audio directory not found: {path_srcdir}")
            return result

        # 获取文件列表
        filelist = traverse_dir(
            path_srcdir,
            extensions=self.extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )

        result['total_files'] = len(filelist)

        if len(filelist) == 0:
            result['valid'] = False
            result['warnings'].append("No audio files found")
            return result

        # 检查音频文件（采样 10 个或全部）
        sample_size = min(10, len(filelist))
        sample_files = random.sample(filelist, sample_size)

        print(f"[AudioPreprocessor] Validating {sample_size} sample files...")

        for file in sample_files:
            path_file = os.path.join(path_srcdir, file)
            try:
                audio, sr = librosa.load(path_file, sr=None, duration=1)  # 只读 1 秒

                # 检查采样率
                if sr != self.sample_rate:
                    result['invalid_files'].append({
                        'file': file,
                        'issue': f"Sample rate mismatch: {sr}Hz (expected {self.sample_rate}Hz)"
                    })
                    continue

                # 检查声道数
                if len(audio.shape) > 1 and audio.shape[0] > 1:
                    result['warnings'].append(f"{file}: Multi-channel audio (will be converted to mono)")

                result['valid_files'] += 1

            except Exception as e:
                result['invalid_files'].append({
                    'file': file,
                    'issue': f"Cannot read file: {str(e)}"
                })

        # 根据采样结果推断整体情况
        if len(result['invalid_files']) > 0:
            result['valid'] = False
            # 只返回前 10 个不合规文件
            result['invalid_files'] = result['invalid_files'][:10]
        else:
            result['valid_files'] = len(filelist)

        # 预计处理时间（每个文件约 5 秒）
        result['estimated_time'] = len(filelist) * 5

        return result


# 保持向后兼容：函数式接口
def preprocess(
    path,
    f0_extractor,
    volume_extractor,
    mel_extractor,
    units_encoder,
    sample_rate,
    hop_size,
    device='cuda',
    use_pitch_aug=False,
    extensions=['wav']
):
    """
    原始函数式接口（向后兼容）

    注意：建议使用 AudioPreprocessor 类接口
    """
    path_srcdir = os.path.join(path, 'audio')
    path_unitsdir = os.path.join(path, 'units')
    path_f0dir = os.path.join(path, 'f0')
    path_volumedir = os.path.join(path, 'volume')
    path_augvoldir = os.path.join(path, 'aug_vol')
    path_meldir = os.path.join(path, 'mel')
    path_augmeldir = os.path.join(path, 'aug_mel')
    path_skipdir = os.path.join(path, 'skip')

    filelist = traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True
    )

    pitch_aug_dict = {}

    def process(file):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)

        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        volume = volume_extractor.extract(audio)

        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()

            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0

            aug_mel_t = mel_extractor.extract(
                audio_t * (10 ** log10_vol_shift),
                sample_rate,
                keyshift=keyshift
            )
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift))

        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()

        f0 = f0_extractor.extract(audio, uv_interp=False)

        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_volumefile, volume)
            if mel_extractor is not None:
                pitch_aug_dict[file] = keyshift
                os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                np.save(path_melfile, mel)
                os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                np.save(path_augmelfile, aug_mel)
                os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                np.save(path_augvolfile, aug_vol)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print('This file has been moved to ' + path_skipfile)

    print('Preprocess the audio clips in :', path_srcdir)

    for file in tqdm(filelist, total=len(filelist)):
        process(file)

    if mel_extractor is not None:
        path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
        np.save(path_pitchaugdict, pitch_aug_dict)


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="disable resume mode (reprocess all files)"
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # Parse commands
    cmd = parse_args()

    # 使用新的类接口
    print("\n" + "="*60)
    print("DDSP-SVC Audio Preprocessor")
    print("="*60 + "\n")

    # 初始化预处理器
    preprocessor = AudioPreprocessor(cmd.config, device=cmd.device)

    # 判断是否启用 pitch augmentation
    use_pitch_aug = preprocessor.args.model.use_pitch_aug if hasattr(preprocessor.args.model, 'use_pitch_aug') else False

    # 预处理训练集
    print("\n[1/2] Processing training set...")
    train_stats = preprocessor.preprocess_dataset(
        preprocessor.args.data.train_path,
        use_pitch_aug=use_pitch_aug,
        resume=not cmd.no_resume
    )

    # 预处理验证集
    print("\n[2/2] Processing validation set...")
    val_stats = preprocessor.preprocess_dataset(
        preprocessor.args.data.valid_path,
        use_pitch_aug=False,  # 验证集不使用 pitch aug
        resume=not cmd.no_resume
    )

    # 最终统计
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nTraining set:")
    print(f"  Processed: {train_stats['processed']}/{train_stats['total']}")
    print(f"  Skipped: {train_stats['skipped']}")
    print(f"  Failed: {train_stats['failed']}")
    print(f"\nValidation set:")
    print(f"  Processed: {val_stats['processed']}/{val_stats['total']}")
    print(f"  Skipped: {val_stats['skipped']}")
    print(f"  Failed: {val_stats['failed']}")

    if train_stats['failed'] > 0 or val_stats['failed'] > 0:
        print(f"\n⚠️  Warning: Some files failed to process.")
        print(f"   Check errors.log in data/train and data/val directories.")

    print("\n✅ Done!")
