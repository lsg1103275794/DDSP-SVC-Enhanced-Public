import os
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
import shutil
import random
import sys
import yaml
import time
from typing import List, Dict, Any, Callable
from api.core.config import UPLOAD_DIR, PROCESSED_DIR, DATASETS_DIR, DEVICE, BASE_DIR

# Add SVCFusion path
# ReFlowVaeSVC is inside "SVCFusion 社区", and it uses absolute imports starting with ReFlowVaeSVC
sys.path.append(os.path.join(BASE_DIR, "SVCFusion 社区"))
# Also add ReFlowVaeSVC itself if needed for some modules (though usually parent dir is enough)
sys.path.append(os.path.join(BASE_DIR, "SVCFusion 社区", "ReFlowVaeSVC"))

from api.services.msst_service import msst_service
from slicer import Slicer
from logger import utils
# Use ddsp.vocoder for core extractors to handle dependencies more gracefully (and it's what we have)
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
# Use reflow.vocoder for Mel extraction (Vocoder)
from ReFlowVaeSVC.reflow.vocoder import Vocoder


class PreprocessService:
    def __init__(self):
        self.device = DEVICE
        self.raw_dir = os.path.join(BASE_DIR, "dataset_raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def save_to_raw(
        self,
        file_content: bytes,
        filename: str,
        spk_name: str = None,
        relative_path: str = None,
    ):
        """保存音频到 dataset_raw 目录，遵循用户指定的逻辑"""
        print(
            f" [DEBUG] save_to_raw: filename={filename}, spk_name={spk_name}, relative_path={relative_path}"
        )

        # 规范化路径分隔符
        if relative_path:
            relative_path = relative_path.replace("\\", "/")

        if relative_path and "/" in relative_path:
            # 1. 文件夹上传逻辑: 直接复用文件夹名
            parts = relative_path.split("/")
            actual_spk_name = parts[0]
            if not actual_spk_name or actual_spk_name.strip() == "":
                actual_spk_name = "未命名角色"
            sub_dirs = parts[1:-1]
            target_dir = os.path.join(self.raw_dir, actual_spk_name, *sub_dirs)
            print(
                f" [DEBUG] Folder upload: actual_spk_name={actual_spk_name}, target_dir={target_dir}"
            )
        else:
            # 2. 单文件上传逻辑: 自动创建 01 文件夹
            # 必须有角色名，如果没有则使用 "未命名角色"
            actual_spk_name = (
                spk_name if spk_name and spk_name.strip() else "未命名角色"
            )
            target_dir = os.path.join(self.raw_dir, actual_spk_name, "01")
            print(
                f" [DEBUG] Single file upload: actual_spk_name={actual_spk_name}, target_dir={target_dir}"
            )

        try:
            os.makedirs(target_dir, exist_ok=True)
            # 最终保存路径
            save_path = os.path.abspath(os.path.join(target_dir, filename))

            with open(save_path, "wb") as f:
                f.write(file_content)

            print(f" [Dataset Raw] Successfully saved to: {save_path}")
            # 再次确认文件是否存在
            if os.path.exists(save_path):
                print(
                    f" [DEBUG] Verified file exists at: {save_path}, size={os.path.getsize(save_path)} bytes"
                )
            else:
                print(f" [ERROR] File NOT found immediately after write: {save_path}")

            return save_path
        except Exception as e:
            print(f" [ERROR] Failed to save file: {str(e)}")
            raise e

    def list_files(self, type: str = "upload") -> List[Dict[str, Any]]:
        """获取已上传或处理的文件列表"""
        files = []
        if type == "upload":
            target_dir = self.raw_dir
        elif type == "separated":
            target_dir = PROCESSED_DIR
        elif type == "sliced":
            target_dir = os.path.join(BASE_DIR, "data", "train", "audio")
        else:
            return []

        if not os.path.exists(target_dir):
            return []

        # 递归扫描所有音频文件
        for root, _, filenames in os.walk(target_dir):
            for filename in filenames:
                if filename.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg")):
                    file_path = os.path.join(root, filename)
                    files.append({
                        "id": str(len(files) + 1),
                        "filename": filename,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "type": "audio",
                        "updated_at": os.path.getmtime(file_path)
                    })
        return files

    def list_raw_datasets(self) -> List[Dict[str, Any]]:
        """获取 dataset_raw 目录下的数据集列表 (递归扫描所有音频)"""
        print(f" [DEBUG] Scanning datasets in: {self.raw_dir}")
        datasets = []
        if not os.path.exists(self.raw_dir):
            print(" [DEBUG] raw_dir does not exist")
            return []

        # 遍历 dataset_raw 下的所有一级目录（每个目录代表一个角色/数据集）
        for item in os.listdir(self.raw_dir):
            item_path = os.path.join(self.raw_dir, item)
            if os.path.isdir(item_path):
                # 递归计算该目录下的音频文件数量
                count = 0
                for root, dirs, files in os.walk(item_path):
                    for f in files:
                        if f.lower().endswith(
                            (".wav", ".flac", ".mp3", ".m4a", ".ogg")
                        ):
                            count += 1
                
                # 即使是空文件夹也显示，方便用户看到上传的文件夹结构
                datasets.append(
                    {
                        "name": item,
                        "path": item_path,
                        "file_count": count,
                        "updated_at": os.path.getmtime(item_path),
                    }
                )

        # 检查根目录下是否有散落的文件（如果有）
        try:
            root_audios = [
                f
                for f in os.listdir(self.raw_dir)
                if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg"))
            ]
            if root_audios:
                print(f" [DEBUG] Found root audio files: {len(root_audios)}")
                datasets.append(
                    {
                        "name": "(根目录未分类文件)",
                        "path": self.raw_dir,
                        "file_count": len(root_audios),
                        "updated_at": time.time(),
                    }
                )
        except Exception as e:
            print(f" [ERROR] Failed to scan root audios: {str(e)}")

        print(f" [DEBUG] Total datasets found: {len(datasets)}")
        return datasets

    async def run_full_preprocess(
        self, req: Any, progress_callback: Callable[[int, str], None]
    ):
        """运行完整预处理逻辑 (对齐 SVCFusion 社区版逻辑)"""
        dataset_name = req.dataset_name
        # req 参数可能不全，我们主要依赖 reflow.yaml，但允许 req 覆盖部分
        
        device = req.device if req.device else self.device
        skip_slicing = req.skip_slicing

        # 1. 检查数据集
        src_dir = os.path.join(self.raw_dir, dataset_name)
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"数据集目录 {src_dir} 不存在")

        # 2. 准备训练目录结构
        # SVCFusion logic puts preprocessed data into data/train/audio/dataset_name etc.
        base_data_dir = os.path.join(BASE_DIR, "data")
        train_audio_dir = os.path.join(base_data_dir, "train", "audio")
        val_audio_dir = os.path.join(base_data_dir, "val", "audio")
        
        # 确保基础目录存在
        for d in [train_audio_dir, val_audio_dir]:
            os.makedirs(d, exist_ok=True)

        # 清理旧数据 (针对特定数据集)
        # 注意：SVCFusion 的 preprocess.py 是直接写入，这里我们先清理以保证干净
        for d in [train_audio_dir, val_audio_dir]:
            spk_dir = os.path.join(d, dataset_name)
            if os.path.exists(spk_dir):
                shutil.rmtree(spk_dir)
            os.makedirs(spk_dir, exist_ok=True)

        # 3. 收集并处理音频文件 (切片或直接复制)
        progress_callback(5, "正在扫描和处理音频文件...")
        raw_files = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg")):
                    raw_files.append(os.path.join(root, f))

        if not raw_files:
            raise Exception(f"数据集 {dataset_name} 中未找到有效音频文件")

        all_sliced_files = []
        
        # Slicing / Resampling Logic
        # We process files and put them into data/train/audio/dataset_name
        # Then we select val set and move them to data/val/audio/dataset_name
        
        target_sr = 44100 # default
        
        # 预先加载配置以获取采样率 (为了切片时的重采样)
        config_path = os.path.join(BASE_DIR, "configs", "reflow.yaml")
        try:
            args = utils.load_config(config_path)
            target_sr = args.data.sampling_rate
        except:
            print(" [WARN] Failed to load config early, using default sr=44100")
            target_sr = 44100

        if skip_slicing:
            for f_path in raw_files:
                f_name = os.path.basename(f_path)
                dst_path = os.path.join(train_audio_dir, dataset_name, f_name)
                # Ensure wav extension
                if not f_name.lower().endswith(".wav"):
                     dst_path = os.path.splitext(dst_path)[0] + ".wav"
                
                try:
                    # Load and resample if needed
                    waveform, sr = torchaudio.load(f_path)
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        waveform = resampler(waveform)
                        sr = target_sr
                    
                    # Ensure mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                        
                    torchaudio.save(dst_path, waveform, sr)
                    all_sliced_files.append(dst_path)
                except Exception as e:
                    print(f"Error copying {f_name}: {str(e)}")

        else:
            # 切片逻辑
            slicer = Slicer(sr=target_sr, threshold=-40, min_length=5000)
            for i, f_path in enumerate(raw_files):
                f_name = os.path.basename(f_path)
                progress_callback(
                    10 + int(i / len(raw_files) * 20), f"正在切片: {f_name}"
                )
                try:
                    waveform, sr = torchaudio.load(f_path)
                    # Resample first
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        waveform = resampler(waveform)
                        sr = target_sr
                        
                    # Mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    audio_mono = waveform[0].numpy()
                    
                    chunks = slicer.slice(audio_mono)
                    for k, v in chunks.items():
                        if v["slice"]:
                            continue
                        tag = v["split_time"].split(",")
                        start, end = int(tag[0]), int(tag[1])
                        if start == end:
                            continue

                        chunk_waveform = waveform[:, start:end]
                        out_name = f"{os.path.splitext(f_name)[0]}_{k}.wav"
                        out_path = os.path.join(train_audio_dir, dataset_name, out_name)
                        torchaudio.save(out_path, chunk_waveform, sr)
                        all_sliced_files.append(out_path)
                except Exception as e:
                    print(f"Error slicing {f_name}: {str(e)}")
                    continue

        # 4. 划分验证集
        progress_callback(35, "正在划分验证集...")
        if len(all_sliced_files) > 1:
            val_count = max(1, int(len(all_sliced_files) * 0.02))
            val_files = random.sample(all_sliced_files, val_count)
            for f in val_files:
                dst_path = os.path.join(
                    val_audio_dir, dataset_name, os.path.basename(f)
                )
                shutil.move(f, dst_path)
                # Update all_sliced_files list to reflect moves?
                # Actually for the next step (feature extraction), we need to scan dirs again or track paths.
                # SVCFusion preprocess.py iterates over the directory again.

        # 5. 加载配置并初始化提取器
        progress_callback(40, "正在加载模型配置...")
        # args is already loaded above, but let's reload to be safe or use existing
        if 'args' not in locals():
             args = utils.load_config(config_path)

        # Allow overriding encoder/f0 from request if provided, otherwise use config
        if req.encoder:
            args.data.encoder = req.encoder
        if req.f0_extractor:
            args.data.f0_extractor = req.f0_extractor

        # Initialize Extractors
        f0_extractor = F0_Extractor(
            args.data.f0_extractor,
            args.data.sampling_rate,
            args.data.block_size,
            args.data.f0_min,
            args.data.f0_max,
        )
        volume_extractor = Volume_Extractor(
            args.data.block_size
        )
        
        mel_extractor = None
        use_pitch_aug = False
        if args.model.type in ["RectifiedFlow", "RectifiedFlow_VAE"]:
             try:
                # Use ReFlowVaeSVC.reflow.vocoder.Vocoder
                mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
                if (mel_extractor.vocoder_sample_rate != args.data.sampling_rate or 
                    mel_extractor.vocoder_hop_size != args.data.block_size):
                    mel_extractor = None
                    print("Unmatch vocoder parameters, mel extraction is ignored!")
                elif args.model.use_pitch_aug:
                    use_pitch_aug = True
             except Exception as e:
                 print(f" [ERROR] Failed to init Mel Extractor: {e}")
                 mel_extractor = None

        encoder_hop_size = args.data.encoder_hop_size
        if args.data.encoder == "contentvec768l12tta2x":
             encoder_hop_size = 160
        
        cnhubertsoft_gate = getattr(args.data, "cnhubertsoft_gate", 10)

        try:
            units_encoder = Units_Encoder(
                args.data.encoder,
                args.data.encoder_ckpt,
                args.data.encoder_sample_rate,
                encoder_hop_size,
                cnhubertsoft_gate=cnhubertsoft_gate,
                device=device,
            )
        except Exception as e:
            print(f" [ERROR] Failed to init Units Encoder: {e}")
            raise Exception(f"初始化 Units Encoder 失败: {str(e)}. 可能缺少依赖或模型文件.")

        # 6. 执行特征提取 (Traverse data/train/audio AND data/val/audio)
        progress_callback(50, "正在提取特征 (F0, Units, Volume)...")
        
        pitch_aug_dict = {}

        # Process function
        def process_dataset_folder(path_root, is_train=True):
            # path_root is e.g. data/train or data/val
            # path_srcdir is e.g. data/train/audio/dataset_name
            path_srcdir = os.path.join(path_root, "audio", dataset_name)
            
            # Output dirs relative to path_root
            path_unitsdir = os.path.join(path_root, "units", dataset_name)
            path_f0dir = os.path.join(path_root, "f0", dataset_name)
            path_volumedir = os.path.join(path_root, "volume", dataset_name)
            path_augvoldir = os.path.join(path_root, "aug_vol", dataset_name)
            path_meldir = os.path.join(path_root, "mel", dataset_name)
            path_augmeldir = os.path.join(path_root, "aug_mel", dataset_name)
            path_skipdir = os.path.join(path_root, "skip", dataset_name) # Usually only for train?
            
            if not os.path.exists(path_srcdir):
                return

            # Get file list
            filelist = [f for f in os.listdir(path_srcdir) if f.endswith(".wav")]
            # Note: utils.traverse_dir is better but os.listdir is fine for flat structure we just created

            total_files = len(filelist)
            for i, file in enumerate(filelist):
                if is_train:
                    p = 50 + int(i / total_files * 40)
                else:
                    p = 90 + int(i / total_files * 10)
                progress_callback(p, f"处理 ({'训练集' if is_train else '验证集'}): {file}")

                binfile = file + ".npy"
                path_srcfile = os.path.join(path_srcdir, file)
                
                path_unitsfile = os.path.join(path_unitsdir, binfile)
                path_f0file = os.path.join(path_f0dir, binfile)
                path_volumefile = os.path.join(path_volumedir, binfile)
                path_augvolfile = os.path.join(path_augvoldir, binfile)
                path_melfile = os.path.join(path_meldir, binfile)
                path_augmelfile = os.path.join(path_augmeldir, binfile)
                path_skipfile = os.path.join(path_skipdir, file)

                try:
                    # 加载音频
                    audio, sr = librosa.load(path_srcfile, sr=args.data.sampling_rate)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio)
                    audio_t = torch.from_numpy(audio).float().to(device)
                    audio_t = audio_t.unsqueeze(0) # (1, T)

                    # 5.1 F0 提取
                    f0 = f0_extractor.extract(audio, uv_interp=False)
                    uv = f0 == 0
                    if len(f0[~uv]) > 0:
                        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
                    else:
                         print(f"\n[Error] F0 extraction failed: {path_srcfile}")
                         if is_train:
                             os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
                             shutil.move(path_srcfile, os.path.dirname(path_skipfile))
                             print(f"This file has been moved to {path_skipfile}")
                         continue # 跳过该文件

                    # 5.2 Volume 提取
                    volume = volume_extractor.extract(audio)
                    
                    # 5.3 Mel 提取 & 增强
                    aug_mel = None
                    aug_vol = None
                    keyshift = 0
                    mel = None
                    
                    if mel_extractor is not None:
                        mel_t = mel_extractor.extract(audio_t, args.data.sampling_rate)
                        mel = mel_t.squeeze().to("cpu").numpy()
                        
                        max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
                        max_shift = min(1, np.log10(1 / max_amp))
                        log10_vol_shift = random.uniform(-1, max_shift)
                        
                        if use_pitch_aug:
                            keyshift = random.uniform(-5, 5)
                        else:
                            keyshift = 0
                        
                        aug_mel_t = mel_extractor.extract(
                            audio_t * (10**log10_vol_shift), args.data.sampling_rate, keyshift=keyshift
                        )
                        aug_mel = aug_mel_t.squeeze().to("cpu").numpy()
                        aug_vol = volume_extractor.extract(audio * (10**log10_vol_shift))

                    # 5.4 Units 编码
                    units_t = units_encoder.encode(
                        audio_t, args.data.sampling_rate, args.data.block_size
                    )
                    units = units_t.squeeze().to("cpu").numpy()
                    
                    # 5.5 保存特征
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

                except Exception as e:
                    print(f" [ERROR] Failed to process {file}: {e}")
                    # traceback.print_exc()

        # Process Train
        process_dataset_folder(os.path.join(base_data_dir, "train"), is_train=True)
        # Process Val
        process_dataset_folder(os.path.join(base_data_dir, "val"), is_train=False)

        # Save pitch_aug_dict
        if use_pitch_aug:
             np.save(os.path.join(BASE_DIR, "data", "pitch_aug_dict.npy"), pitch_aug_dict)

        progress_callback(100, "预处理完成!")


preprocess_service = PreprocessService()
