import os
import subprocess
import signal
from typing import Optional, Dict
from api.core.config import BASE_DIR

class TrainService:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log_file = os.path.join(BASE_DIR, "train.log")

    def start_training(self, config_path: str):
        """启动训练子进程"""
        if self.process and self.process.poll() is None:
            return {"status": "error", "message": "Training is already running"}

        # 构造命令行
        cmd = [
            "python", 
            os.path.join(BASE_DIR, "train_reflow.py"),
            "-c", config_path
        ]

        # 启动进程并重定向输出
        log_handle = open(self.log_file, "w")
        self.process = subprocess.Popen(
            cmd, 
            stdout=log_handle, 
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        return {"status": "success", "pid": self.process.pid}

    def stop_training(self):
        """停止训练进程"""
        if not self.process or self.process.poll() is not None:
            return {"status": "error", "message": "No training process running"}

        if os.name == 'nt':
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.process.terminate()
        
        self.process.wait()
        return {"status": "success", "message": "Training stopped"}

    def get_status(self):
        """获取训练状态"""
        if not self.process:
            return {"status": "idle"}
        
        exit_code = self.process.poll()
        if exit_code is None:
            return {"status": "running", "pid": self.process.pid}
        else:
            return {"status": "finished", "exit_code": exit_code}

    def get_logs(self, lines: int = 100) -> str:
        """读取训练日志最后 N 行"""
        if not os.path.exists(self.log_file):
            return "No logs yet."
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                content = f.readlines()
                return "".join(content[-lines:])
        except Exception as e:
            return f"Error reading logs: {str(e)}"

train_service = TrainService()
