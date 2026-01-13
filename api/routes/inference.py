from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.core.config import EXP_DIR, BASE_DIR
from api.models.schemas import ConvertRequest
from api.services.inference_service import inference_service
from api.routes.preprocess import tasks_db, update_task
import os
import uuid
import time

router = APIRouter()

async def run_inference_task(task_id: str, req: ConvertRequest):
    try:
        update_task(task_id, "running", 20, "加载模型与音频中...")
        output_path = await inference_service.infer(
            input_id=req.input_id,
            checkpoint_name=req.model_path,
            spk_id=req.spk_id,
            pitch_shift=req.pitch_shift,
            # AudioNoise 增强参数
            f0_smooth=req.f0_smooth,
            f0_smooth_cutoff=req.f0_smooth_cutoff,
            median_kernel=req.median_kernel,
            octave_fix=req.octave_fix,
            vibrato=req.vibrato,
            vibrato_rate=req.vibrato_rate,
            vibrato_depth=req.vibrato_depth,
            vibrato_delay=req.vibrato_delay,
            tremolo=req.tremolo,
            tremolo_rate=req.tremolo_rate,
            tremolo_depth=req.tremolo_depth,
            effects_preset=req.effects_preset,
            chorus=req.chorus,
            reverb=req.reverb,
            reverb_mix=req.reverb_mix
        )
        update_task(task_id, "completed", 100, f"转换成功: {os.path.basename(output_path)}")
    except Exception as e:
        update_task(task_id, "failed", 0, f"推理失败: {str(e)}")

@router.get("/models")
async def list_models():
    """列出所有 SVC 模型及其 Checkpoints"""
    models = []
    if os.path.exists(EXP_DIR):
        for d in os.listdir(EXP_DIR):
            model_path = os.path.join(EXP_DIR, d)
            if os.path.isdir(model_path):
                checkpoints = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.ckpt'))]
                if checkpoints:
                    models.append({
                        "name": d,
                        "checkpoints": checkpoints,
                        "path": model_path
                    })
    return {"models": models}

@router.post("/convert")
async def convert_audio(req: ConvertRequest, background_tasks: BackgroundTasks):
    """提交音频转换推理任务"""
    task_id = f"conv_{uuid.uuid4().hex[:8]}"
    update_task(task_id, "pending", 0, "初始化推理引擎...")
    
    background_tasks.add_task(run_inference_task, task_id, req)
    
    return {"task_id": task_id}
