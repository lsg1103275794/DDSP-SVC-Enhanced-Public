# DDSP-SVC 现代化 UI - API 接口文档

**版本**: v1.0
**基本路径**: `/api/v1`
**数据格式**: JSON

---

## 0. 项目架构概览 (Project Architecture)

本项目采用模块化、分层设计的微服务化架构，旨在提供高性能、可扩展的音频处理能力。

### 核心层级
- **Web UI 层 (Vue 3)**: 提供 Apple 风格的现代化交互界面，支持任务状态实时追踪。
- **API 路由层 (FastAPI)**: 统一版本化接口 (`/api/v1`)，基于 BackgroundTasks 实现异步任务调度。
- **业务服务层 (Services)**:
    - `PreprocessService`: 负责音频切片、MSST 分离任务的管理。
    - `MSSTService`: 封装音源分离核心逻辑，支持 BS-Roformer 等模型。
    - `InferenceService`: 封装 DDSP-SVC 推理逻辑，支持自动提取 F0、音量及单元编码。
    - `SystemService`: 监控系统资源（CPU/GPU/内存）及文件管理。
- **AI 核心引擎层 (AI Engines)**:
    - **DDSP-SVC Core**: 核心合成引擎，支持多种 F0 提取器（RMVPE, Crepe 等）。
    - **MSST (Music Source Separation Training)**: 音频源分离引擎。
        - **代码路径**: `Music_Source_Separation_Training/` (集成 BS-Roformer 底层实现)
        - **模型路径**: `other_weights/` (存储 `.ckpt` 权重文件)
        - **集成方式**: 通过 `MSSTService` 加载模型并执行 `demix_track` 进行高精度分离。

---

## 系统架构 (System Architecture)

### 目录角色 (Directory Roles)
- **`Music_Source_Separation_Training`**: MSST 引擎核心代码，提供人声分离算法实现。
- **`other_weights`**: 存储 MSST 预训练权重（如 BS-Roformer, Demucs 等）。
- **`storage/`**: 持久化存储，包含 `uploads` (原始上传), `processed` (分离与切分结果), `outputs` (推理生成结果)。
- **`exp/`**: 存储 DDSP-SVC 训练过程中的模型 Checkpoints。

### 核心服务流 (Core Service Flow)
1. **音频分离 (MSST)**: `MSSTService` 加载 `other_weights` 中的权重，利用 `Music_Source_Separation_Training` 提供的 `demix` 逻辑将混音分离为人声 (Vocal) 与伴奏 (Instrument)。
2. **音频切分 (Slicer)**: `PreprocessService` 使用 `slicer.py` 将长音频切分为短片段，适配模型训练与推理。
3. **模型推理 (DDSP-SVC)**: `InferenceService` 加载 `exp/` 下的 SVC 模型，执行 F0 提取与音频转换。

---

## API 路由详情 (API Endpoints)

---

## 1. 公共接口 (Common)

### 1.1 获取系统状态
- **URL**: `/status`
- **方法**: `GET`
- **返回**:
  ```json
  {
    "status": "online",
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "memory": {
      "used": "2.4GB",
      "total": "12.0GB"
    }
  }
  ```

### 1.2 文件上传
- **URL**: `/upload`
- **方法**: `POST`
- **参数**: `file: UploadFile`
- **返回**:
  ```json
  {
    "id": "uuid-123",
    "filename": "vocal.wav",
    "path": "uploads/vocal.wav"
  }
  ```

---

## 2. 音频预处理 (Audio Processing)

### 2.1 提交 MSST 分离任务
- **URL**: `/preprocess/separate`
- **方法**: `POST`
- **参数**:
  ```json
  {
    "input_id": "uuid-123",
    "model": "bs_roformer"
  }
  ```
- **返回**: `{"task_id": "job-456"}`

### 2.2 提交音频切分任务
- **URL**: `/preprocess/slice`
- **方法**: `POST`
- **参数**:
  ```json
  {
    "input_id": "uuid-123",
    "threshold": -40,
    "min_length": 5000
  }
  ```
- **返回**: `{"task_id": "job-789"}`

---

## 3. 模型训练 (Model Training)

### 3.1 获取模型配置
- **URL**: `/train/config`
- **方法**: `GET`
- **返回**: YAML 转换后的 JSON 对象

### 3.2 启动训练
- **URL**: `/train/start`
- **方法**: `POST`
- **参数**: `{ "config_overrides": { ... } }`
- **返回**: `{"status": "starting", "pid": 1234}`

### 3.3 训练实时状态 (WebSocket)
- **URL**: `/ws/train/logs`
- **协议**: `WS`
- **消息**: 实时推送 TensorBoard 指标与控制台输出。

---

## 4. 音频推理 (Audio Inference)

### 4.1 列出可用模型
- **URL**: `/inference/models`
- **方法**: `GET`
- **返回**: `{"models": [{"name": "model_v1", "path": "exp/model_v1/model.pt"}]}`

### 4.2 执行推理转换
- **URL**: `/inference/convert`
- **方法**: `POST`
- **参数**:
  ```json
  {
    "model_name": "model_v1",
    "input_id": "uuid-123",
    "f0_predictor": "crepe",
    "key_shift": 12,
    "p_pitch": 0.5
  }
  ```
- **返回**: `{"output_url": "/api/v1/download/output_uuid.wav"}`

---

## 5. 任务管理 (Task Management)

### 5.1 获取任务进度
- **URL**: `/tasks/{task_id}`
- **方法**: `GET`
- **返回**:
  ```json
  {
    "task_id": "job-456",
    "status": "processing",
    "progress": 65,
    "message": "正在分离人声..."
  }
  ```
