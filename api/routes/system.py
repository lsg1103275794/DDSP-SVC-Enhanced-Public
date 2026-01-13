from fastapi import APIRouter, UploadFile, File, HTTPException
from api.core.config import DEVICE, GPU_NAME, UPLOAD_DIR
from api.models.schemas import SystemStatus, FileInfo
from typing import List
import os
import uuid
import shutil

router = APIRouter()

@router.get("/status", response_model=SystemStatus)
async def get_status():
    return {
        "status": "online",
        "device": DEVICE,
        "gpu_name": GPU_NAME,
        "memory": {"used": "N/A", "total": "N/A"}
    }

@router.post("/upload", response_model=FileInfo)
async def upload_audio(file: UploadFile = File(...)):
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
    file_id = f"{uuid.uuid4().hex[:8]}"
    ext = os.path.splitext(file.filename)[1].lower()
    
    # 安全性检查：只允许音频格式
    allowed_exts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_exts)}")
    
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    return {
        "id": file_id,
        "filename": filename,
        "path": file_path,
        "size": os.path.getsize(file_path)
    }

@router.get("/files", response_model=List[FileInfo])
async def list_files():
    """获取已上传的音频列表"""
    if not os.path.exists(UPLOAD_DIR):
        return []
    
    files = []
    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(path):
            files.append({
                "id": f.split('_')[0] if '_' in f else "unknown",
                "filename": f,
                "path": path,
                "size": os.path.getsize(path),
                "type": "upload",
                "updated_at": os.path.getmtime(path)
            })
    return files
