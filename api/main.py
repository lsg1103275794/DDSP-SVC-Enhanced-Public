from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.core.config import PROJECT_NAME, API_V1_STR, UPLOAD_DIR, PROCESSED_DIR, OUTPUT_DIR
from api.routes import system, preprocess, train, inference
import os
import sys

# 将本地 ffmpeg 加入环境变量，确保 torchaudio 等库可以正常处理非 WAV 格式进行校验
ffmpeg_bin = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ffmpeg", "bin")
if os.path.exists(ffmpeg_bin):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]

app = FastAPI(title=PROJECT_NAME, version="1.0.0")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件访问 (用于音频播放)
app.mount("/api/v1/system/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/api/v1/system/static/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/api/v1/system/static/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# 挂载路由
app.include_router(system.router, prefix=API_V1_STR, tags=["System"])
app.include_router(preprocess.router, prefix=f"{API_V1_STR}/preprocess", tags=["Preprocess"])
app.include_router(train.router, prefix=f"{API_V1_STR}/train", tags=["Train"])
app.include_router(inference.router, prefix=f"{API_V1_STR}/inference", tags=["Inference"])

@app.get("/")
async def root():
    return {"message": f"Welcome to {PROJECT_NAME}", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
