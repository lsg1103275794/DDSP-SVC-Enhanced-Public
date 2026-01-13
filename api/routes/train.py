from fastapi import APIRouter, HTTPException
from api.core.config import BASE_DIR
from api.services.train_service import train_service
import os
import yaml

router = APIRouter()

@router.get("/config")
async def get_config():
    """读取默认训练配置"""
    config_path = os.path.join(BASE_DIR, "configs", "reflow.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Config file not found")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

@router.post("/start")
async def start_train(config_path: str = "configs/reflow.yaml"):
    """启动训练任务"""
    result = train_service.start_training(config_path)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@router.post("/stop")
async def stop_train():
    """停止训练任务"""
    result = train_service.stop_training()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@router.get("/status")
async def get_train_status():
    """获取训练状态"""
    return train_service.get_status()

@router.get("/logs")
async def get_train_logs(lines: int = 100):
    """获取训练日志"""
    return {"logs": train_service.get_logs(lines)}
