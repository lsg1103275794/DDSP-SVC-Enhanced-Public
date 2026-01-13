from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    UploadFile,
    File,
    Query,
    Header,
)
from api.models.schemas import (
    SeparateRequest,
    SliceRequest,
    FileInfo,
    PreprocessRequest,
    AudioToolRequest,
)
from api.services.preprocess_service import preprocess_service
from api.core.config import UPLOAD_DIR
import uuid
import time
import os
import shutil
from typing import Optional
from urllib.parse import unquote

router = APIRouter()

# 任务数据库 (建议后续迁移到全局状态管理)
tasks_db = {}


def update_task(task_id: str, status: str, progress: int, message: str):
    tasks_db[task_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": time.time(),
    }


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    spk_name: Optional[str] = Header(None, alias="X-Speaker-Name"),
    relative_path: Optional[str] = Header(None, alias="X-Relative-Path"),
):
    """上传音频文件并存入 dataset_raw/角色名"""
    # 解码 Header 参数
    spk_name = unquote(spk_name) if spk_name else None
    relative_path = unquote(relative_path) if relative_path else None

    # 规范化 relative_path (处理 Windows 反斜杠)
    if relative_path:
        relative_path = relative_path.replace("\\", "/")

    # 逻辑调整：如果没有角色名，默认为 "未命名角色"
    if not spk_name:
        spk_name = "未命名角色"

    # 判断是否为单文件上传 (没有相对路径，或者相对路径中没有目录层级)
    is_single_file = not relative_path or "/" not in relative_path.replace("\\", "/")

    # 存入 dataset_raw
    content = await file.read()

    # 逻辑变更：仅在单文件上传时进行严格合规性检查
    if is_single_file:
        # 1. 首先进行简单的扩展名检查，避免 torchaudio 在不支持的格式上崩溃
        ext = file.filename.lower().split(".")[-1]
        if ext != "wav":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "AUDIO_INVALID",
                    "message": f"音频格式错误: 当前为 {ext.upper()}，系统仅支持 WAV 格式",
                    "suggestion": "请前往『工具箱』进行『格式转换』处理",
                },
            )

        import io
        import torchaudio

        try:
            # 2. 尝试读取音频属性
            waveform, sr = torchaudio.load(io.BytesIO(content))
            channels = waveform.shape[0]

            errors = []
            if sr != 44100:
                errors.append(f"采样率非 44100Hz (当前: {sr}Hz)")
            if channels != 1:
                errors.append(f"非单声道 (当前: {channels}声道)")

            if errors:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "AUDIO_INVALID",
                        "message": f"音频不符合要求: {', '.join(errors)}",
                        "suggestion": "请前往『工具箱』进行『重采样』处理",
                    },
                )
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"无法解析音频文件，请检查文件是否损坏或格式是否正确: {str(e)}",
            )

    try:
        save_path = preprocess_service.save_to_raw(
            content, file.filename, spk_name, relative_path
        )
        # 返回信息中体现角色名
        actual_spk = (
            os.path.basename(os.path.dirname(os.path.dirname(save_path)))
            if is_single_file
            else relative_path.split("/")[0]
        )

        # 防止 actual_spk 为空的情况 (防御性编码)
        if not actual_spk or actual_spk.strip() == "":
            actual_spk = "未命名角色"

        return {
            "id": actual_spk,
            "filename": file.filename,
            "path": save_path,
            "message": f"成功存入 dataset_raw/{actual_spk}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@router.get("/files", response_model=list[FileInfo])
async def list_preprocess_files(
    type: str = Query("upload", enum=["upload", "separated", "sliced"]),
):
    """获取文件列表"""
    return preprocess_service.list_files(type)


async def run_slice_task(task_id: str, req: SliceRequest):
    try:
        update_task(task_id, "running", 10, "正在读取音频...")
        output_files = await preprocess_service.slice_audio(
            req.input_id, req.threshold, req.min_length
        )
        update_task(
            task_id, "completed", 100, f"切分完成，共 {len(output_files)} 个片段"
        )
    except Exception as e:
        update_task(task_id, "failed", 0, f"错误: {str(e)}")


async def run_separate_task(task_id: str, req: SeparateRequest):
    try:
        update_task(task_id, "running", 20, "正在加载 MSST 模型并分离音频...")
        result = await preprocess_service.separate_audio(
            req.input_id, req.model, req.stems, req.quality
        )
        update_task(
            task_id, "completed", 100, f"分离完成: {os.path.basename(result['vocal'])}"
        )
    except Exception as e:
        update_task(task_id, "failed", 0, f"错误: {str(e)}")


@router.post("/separate")
async def start_separate(req: SeparateRequest, background_tasks: BackgroundTasks):
    task_id = f"sep_{uuid.uuid4().hex[:8]}"
    update_task(task_id, "pending", 0, "准备 MSST 分离...")
    background_tasks.add_task(run_separate_task, task_id, req)
    return {"task_id": task_id}


@router.post("/slice")
async def start_slice(req: SliceRequest, background_tasks: BackgroundTasks):
    task_id = f"slice_{uuid.uuid4().hex[:8]}"
    update_task(task_id, "pending", 0, "准备切分...")
    background_tasks.add_task(run_slice_task, task_id, req)
    return {"task_id": task_id}


async def run_preprocess_task(task_id: str, req: PreprocessRequest):
    try:
        update_task(task_id, "running", 5, "正在初始化预处理环境...")
        await preprocess_service.run_full_preprocess(
            req, lambda p, m: update_task(task_id, "running", p, m)
        )
        update_task(task_id, "completed", 100, "预处理全部完成！")
    except Exception as e:
        update_task(task_id, "failed", 0, f"错误: {str(e)}")


@router.post("/run")
async def start_preprocess(req: PreprocessRequest, background_tasks: BackgroundTasks):
    """一键预处理"""
    task_id = f"pre_{uuid.uuid4().hex[:8]}"
    update_task(task_id, "pending", 0, "准备全流程预处理...")
    background_tasks.add_task(run_preprocess_task, task_id, req)
    return {"task_id": task_id}


@router.get("/datasets")
async def list_raw_datasets():
    """获取待处理的数据集列表"""
    return preprocess_service.list_raw_datasets()


async def run_audio_tool_task(task_id: str, tool_type: str, req: AudioToolRequest):
    try:
        update_task(task_id, "running", 0, f"正在启动 {tool_type} 任务...")
        await preprocess_service.run_audio_tool(
            tool_type, req, lambda p, m: update_task(task_id, "running", p, m)
        )
        update_task(task_id, "completed", 100, f"{tool_type} 任务全部完成！")
    except Exception as e:
        update_task(task_id, "failed", 0, f"错误: {str(e)}")


@router.post("/tool/{tool_type}")
async def start_audio_tool(
    tool_type: str, req: AudioToolRequest, background_tasks: BackgroundTasks
):
    """基础音频工具 (slice, resample, convert)"""
    if tool_type not in ["slice", "resample", "convert"]:
        raise HTTPException(status_code=400, detail="Unsupported tool type")

    task_id = f"tool_{uuid.uuid4().hex[:8]}"
    update_task(task_id, "pending", 0, f"准备 {tool_type} 任务...")
    background_tasks.add_task(run_audio_tool_task, task_id, tool_type, req)
    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **tasks_db[task_id]}
