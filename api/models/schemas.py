from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SystemStatus(BaseModel):
    status: str
    device: str
    gpu_name: str
    memory: Dict[str, str]


class FileInfo(BaseModel):
    id: str
    filename: str
    path: str
    size: int
    type: str  # 'upload', 'separated', 'sliced', 'output'
    updated_at: float


class SeparateRequest(BaseModel):
    input_id: str
    model: str = "BS-Roformer-Resurrection"
    stems: int = 2
    quality: str = "standard"


class SliceRequest(BaseModel):
    input_id: str
    threshold: float = -40.0
    min_length: int = 5000


class DatasetCreateRequest(BaseModel):
    dataset_name: str
    input_ids: List[str]  # 选中的切片 ID 或 目录名


class PreprocessRequest(BaseModel):
    dataset_name: str
    encoder: str = "contentvec768l12tta2x"
    f0_extractor: str = "fcpe"
    device: str = "cuda"
    slicer_workers: int = 6
    skip_slicing: bool = False
    model_version: str = "DDSP-SVC 6.3"


class AudioToolRequest(BaseModel):
    input_dir: str
    output_dir: str
    max_duration: Optional[int] = 10  # 仅用于切片


class ConvertRequest(BaseModel):
    input_id: str
    model_path: str
    spk_id: int = 1
    pitch_shift: float = 0
    f0_extractor: str = "rmvpe"
    # AudioNoise 增强参数
    f0_smooth: bool = False
    f0_smooth_cutoff: float = 20.0
    median_kernel: int = 3
    octave_fix: bool = False
    vibrato: bool = False
    vibrato_rate: float = 5.5
    vibrato_depth: float = 0.02
    vibrato_delay: float = 0.2
    tremolo: bool = False
    tremolo_rate: float = 4.0
    tremolo_depth: float = 0.1
    effects_preset: str = "none"
    chorus: bool = False
    reverb: bool = False
    reverb_mix: float = 0.2


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    updated_at: float
