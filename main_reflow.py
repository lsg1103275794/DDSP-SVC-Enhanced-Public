import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
import hashlib
from ast import literal_eval
from slicer import Slicer
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from ddsp.enhancements import F0EnhancedProcessor, F0ProcessorConfig
from ddsp.lfo import apply_modulation, ModulationConfig
from ddsp.effects_chain import AudioEffectsChain, EffectsChainConfig, create_effects_chain
from reflow.vocoder import load_model_vocoder
from tqdm import tqdm

def check_args(ddsp_args, diff_args):
    if ddsp_args.data.sampling_rate != diff_args.data.sampling_rate:
        print("Unmatch data.sampling_rate!")
        return False
    if ddsp_args.data.block_size != diff_args.data.block_size:
        print("Unmatch data.block_size!")
        return False
    if ddsp_args.data.encoder != diff_args.data.encoder:
        print("Unmatch data.encoder!")
        return False
    return True
    
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_ckpt",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-id",
        "--spk_id",
        type=str,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-v",
        "--vocal_register_shift_key",
        type=str,
        required=False,
        default=0,
        help="vocal register changed (number of semitones) , only for pc-type vocoder| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-step",
        "--infer_step",
        type=str,
        required=False,
        default='auto',
        help="sample steps | default: auto",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='auto',
        help="euler or rk4 | default: auto",
    )
    parser.add_argument(
        "-ts",
        "--t_start",
        type=str,
        required=False,
        default=0.0,
        help="t_start | default: auto",
    )
    # F0 平滑参数 (AudioNoise 技术)
    parser.add_argument(
        "-f0smooth",
        "--f0_smooth",
        action='store_true',
        help="enable F0 smoothing (reduces pitch jitter) | default: disabled",
    )
    parser.add_argument(
        "-f0cutoff",
        "--f0_smooth_cutoff",
        type=float,
        required=False,
        default=20.0,
        help="F0 smoothing cutoff frequency (Hz) | default: 20.0",
    )
    parser.add_argument(
        "-octavefix",
        "--octave_fix",
        action='store_true',
        help="enable octave error correction | default: disabled",
    )
    parser.add_argument(
        "-mediankernel",
        "--median_kernel",
        type=int,
        required=False,
        default=3,
        help="median filter kernel size for F0 (must be odd) | default: 3",
    )
    # LFO 调制参数 (AudioNoise 技术)
    parser.add_argument(
        "-vibrato",
        "--vibrato",
        action='store_true',
        help="enable vibrato modulation (natural pitch variation) | default: disabled",
    )
    parser.add_argument(
        "-vibrate",
        "--vibrato_rate",
        type=float,
        required=False,
        default=5.5,
        help="vibrato rate in Hz | default: 5.5",
    )
    parser.add_argument(
        "-vibdepth",
        "--vibrato_depth",
        type=float,
        required=False,
        default=0.02,
        help="vibrato depth (0.02 = 2%% = ±24 cents) | default: 0.02",
    )
    parser.add_argument(
        "-vibdelay",
        "--vibrato_delay",
        type=float,
        required=False,
        default=0.2,
        help="vibrato delay in seconds (onset delay) | default: 0.2",
    )
    parser.add_argument(
        "-tremolo",
        "--tremolo",
        action='store_true',
        help="enable tremolo modulation (volume variation) | default: disabled",
    )
    parser.add_argument(
        "-tremrate",
        "--tremolo_rate",
        type=float,
        required=False,
        default=4.0,
        help="tremolo rate in Hz | default: 4.0",
    )
    parser.add_argument(
        "-tremdepth",
        "--tremolo_depth",
        type=float,
        required=False,
        default=0.1,
        help="tremolo depth (0.1 = 10%%) | default: 0.1",
    )
    # 音频效果链参数 (AudioNoise 技术)
    parser.add_argument(
        "-fx",
        "--effects_preset",
        type=str,
        required=False,
        default="none",
        help="audio effects preset: none, natural, spacious, vintage, clean | default: none",
    )
    parser.add_argument(
        "-chorus",
        "--chorus",
        action='store_true',
        help="enable chorus effect | default: disabled",
    )
    parser.add_argument(
        "-reverb",
        "--reverb",
        action='store_true',
        help="enable reverb effect | default: disabled",
    )
    parser.add_argument(
        "-revmix",
        "--reverb_mix",
        type=float,
        required=False,
        default=0.2,
        help="reverb wet/dry mix (0-1) | default: 0.2",
    )
    return parser.parse_args(args=args, namespace=namespace)

    
def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    #device = 'cpu' 
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load reflow model
    model, vocoder, args = load_model_vocoder(cmd.model_ckpt, device=device)

    # 从配置文件读取增强设置（如果存在），命令行参数优先
    if hasattr(args, 'enhance') and args.enhance is not None:
        enhance_cfg = args.enhance
        # F0 平滑（命令行未指定时使用配置文件）
        if not cmd.f0_smooth and enhance_cfg.get('f0_smooth', False):
            cmd.f0_smooth = True
        if cmd.f0_smooth_cutoff == 20.0 and 'f0_smooth_cutoff' in enhance_cfg:
            cmd.f0_smooth_cutoff = enhance_cfg['f0_smooth_cutoff']
        if cmd.median_kernel == 3 and 'median_kernel' in enhance_cfg:
            cmd.median_kernel = enhance_cfg['median_kernel']
        if not cmd.octave_fix and enhance_cfg.get('octave_fix', False):
            cmd.octave_fix = True
        # LFO 调制
        if not cmd.vibrato and enhance_cfg.get('vibrato', False):
            cmd.vibrato = True
        if cmd.vibrato_rate == 5.5 and 'vibrato_rate' in enhance_cfg:
            cmd.vibrato_rate = enhance_cfg['vibrato_rate']
        if cmd.vibrato_depth == 0.02 and 'vibrato_depth' in enhance_cfg:
            cmd.vibrato_depth = enhance_cfg['vibrato_depth']
        if cmd.vibrato_delay == 0.2 and 'vibrato_delay' in enhance_cfg:
            cmd.vibrato_delay = enhance_cfg['vibrato_delay']
        if not cmd.tremolo and enhance_cfg.get('tremolo', False):
            cmd.tremolo = True
        if cmd.tremolo_rate == 4.0 and 'tremolo_rate' in enhance_cfg:
            cmd.tremolo_rate = enhance_cfg['tremolo_rate']
        if cmd.tremolo_depth == 0.1 and 'tremolo_depth' in enhance_cfg:
            cmd.tremolo_depth = enhance_cfg['tremolo_depth']
        # 音频效果
        if cmd.effects_preset == 'none' and 'effects_preset' in enhance_cfg:
            cmd.effects_preset = enhance_cfg['effects_preset']
        if not cmd.chorus and enhance_cfg.get('chorus', False):
            cmd.chorus = True
        if not cmd.reverb and enhance_cfg.get('reverb', False):
            cmd.reverb = True
        if cmd.reverb_mix == 0.2 and 'reverb_mix' in enhance_cfg:
            cmd.reverb_mix = enhance_cfg['reverb_mix']

    # load input
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
    win_size = args.data.volume_smooth_size * sample_rate / args.data.sampling_rate

    # get MD5 hash from wav file
    md5_hash = ""
    with open(cmd.input, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        print("MD5: " + md5_hash)
    
    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")
    
    is_cache_available = os.path.exists(cache_file_path)
    if is_cache_available:
        # f0 cache load
        print('Loading pitch curves for input audio from cache directory...')
        f0 = np.load(cache_file_path, allow_pickle=False)
    else:
        # extract f0
        print('Pitch extractor type: ' + cmd.pitch_extractor)
        pitch_extractor = F0_Extractor(
                            cmd.pitch_extractor, 
                            sample_rate, 
                            hop_size, 
                            float(cmd.f0_min), 
                            float(cmd.f0_max))
        print('Extracting the pitch curve of the input audio...')
        f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)
        
        # f0 cache save
        os.makedirs(cache_dir_path, exist_ok=True)
        np.save(cache_file_path, f0, allow_pickle=False)
    
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)

    # F0 平滑处理 (AudioNoise 技术)
    if cmd.f0_smooth or cmd.octave_fix:
        print('Applying F0 enhancement...')
        f0_config = F0ProcessorConfig(
            enable_smoothing=cmd.f0_smooth,
            cutoff_freq=cmd.f0_smooth_cutoff,
            median_kernel=cmd.median_kernel,
            enable_octave_fix=cmd.octave_fix,
            octave_threshold=1.8
        )
        f0_processor = F0EnhancedProcessor(
            sample_rate=sample_rate,
            hop_size=int(hop_size),
            config=f0_config
        ).to(device)
        f0 = f0_processor(f0)
        if cmd.f0_smooth:
            print(f'  - F0 smoothing: cutoff={cmd.f0_smooth_cutoff}Hz, median_kernel={cmd.median_kernel}')
        if cmd.octave_fix:
            print('  - Octave error correction: enabled')

    # key change
    f0 = f0 * 2 ** (float(cmd.key) / 12)
    
    # formant change
    formant_shift_key = torch.from_numpy(np.array([[float(cmd.formant_shift_key)]])).float().to(device)
    
    # vocal register change
    if vocoder.vocoder.h.pc_aug:
        vocal_register_factor = 2 ** (float(cmd.vocal_register_shift_key) / 12)
    else:
        print('Vocal register shift is not supported for current vocoder!')
        vocal_register_factor = 1
    
    # extract volume 
    print('Extracting the volume envelope of the input audio...')
    volume_extractor = Volume_Extractor(hop_size, win_size)
    volume = volume_extractor.extract(audio)
    mask = (volume > 10 ** (float(cmd.threhold) / 20)).astype('float')
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

    # LFO 调制 (AudioNoise 技术)
    if cmd.vibrato or cmd.tremolo:
        print('Applying LFO modulation...')
        mod_config = ModulationConfig(
            enable_vibrato=cmd.vibrato,
            vibrato_rate=cmd.vibrato_rate,
            vibrato_depth=cmd.vibrato_depth,
            vibrato_delay=cmd.vibrato_delay,
            vibrato_fade_in=0.3,
            enable_tremolo=cmd.tremolo,
            tremolo_rate=cmd.tremolo_rate,
            tremolo_depth=cmd.tremolo_depth
        )
        f0, volume = apply_modulation(f0, volume, sample_rate, int(hop_size), mod_config)
        if cmd.vibrato:
            print(f'  - Vibrato: rate={cmd.vibrato_rate}Hz, depth={cmd.vibrato_depth*100:.1f}%, delay={cmd.vibrato_delay}s')
        if cmd.tremolo:
            print(f'  - Tremolo: rate={cmd.tremolo_rate}Hz, depth={cmd.tremolo_depth*100:.1f}%')

    # load units encoder
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device = device)
                            
    # speaker id or mix-speaker dictionary
    spk_mix_dict = literal_eval(cmd.spk_mix_dict)
    spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)
    if spk_mix_dict is not None:
        print('Mix-speaker mode')
    else:
        print('Speaker ID: '+ str(int(cmd.spk_id)))
    
    # sampling method    
    if cmd.method == 'auto':
        method = args.infer.method
    else:
        method = cmd.method
        
    # infer step
    if cmd.infer_step == 'auto':
        infer_step = args.infer.infer_step
    else:
        infer_step = int(cmd.infer_step)
    
    # t_start
    if cmd.t_start == 'auto':
        if args.model.t_start is not None:
            t_start = float(args.model.t_start)
        else:
            t_start = 0.0
    else:
        t_start = float(cmd.t_start)
        if args.model.t_start is not None and t_start < args.model.t_start:
            t_start = args.model.t_start
            
    if infer_step > 0:
        print('Sampling method: '+ method)
        print('infer step: '+ str(infer_step))
        print('t_start: '+ str(t_start))
    elif infer_step < 0:
        print('infer step cannot be negative!')
        exit(0)
    
    # forward and save the output
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
            seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]    
            seg_mel = model(
                    seg_units, 
                    seg_f0 / vocal_register_factor, 
                    seg_volume, 
                    spk_id = spk_id, 
                    spk_mix_dict = spk_mix_dict,
                    aug_shift = formant_shift_key,
                    vocoder=vocoder,
                    infer_step=infer_step, 
                    method=method,
                    t_start=t_start)
            seg_output = vocoder.infer(seg_mel, seg_f0)
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
            seg_output = seg_output.squeeze().cpu().numpy()
            
            silent_length = round(start_frame * args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

        # 音频效果链处理 (AudioNoise 技术)
        if cmd.effects_preset != "none" or cmd.chorus or cmd.reverb:
            print('Applying audio effects...')

            # 使用预设或自定义配置
            if cmd.effects_preset != "none":
                effects_chain, fx_config = create_effects_chain(
                    args.data.sampling_rate,
                    preset=cmd.effects_preset
                )
                print(f'  - Effects preset: {cmd.effects_preset}')
            else:
                fx_config = EffectsChainConfig(
                    enable_chorus=cmd.chorus,
                    chorus_rate=1.5,
                    chorus_depth=0.4,
                    chorus_mix=0.25,
                    enable_reverb=cmd.reverb,
                    reverb_decay=0.5,
                    reverb_damping=0.3,
                    reverb_mix=cmd.reverb_mix
                )
                effects_chain = AudioEffectsChain(args.data.sampling_rate, fx_config)
                if cmd.chorus:
                    print('  - Chorus: enabled')
                if cmd.reverb:
                    print(f'  - Reverb: mix={cmd.reverb_mix}')

            # 应用效果
            result_tensor = torch.from_numpy(result).float().unsqueeze(0).to(device)
            result_tensor = effects_chain(result_tensor, fx_config)
            result = result_tensor.squeeze().cpu().numpy()

        sf.write(cmd.output, result, args.data.sampling_rate)
    