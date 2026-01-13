# DDSP-SVC 现代化 UI 调研报告

**版本**: v1.0
**日期**: 2026-01-13
**调研目标**: 基于 FastAPI + Vue 3 架构，集成模型训练、音频切分及 MSST 音频处理（人声、混响、和声分离）功能的现代化 UI。

---

## 1. 技术栈选型调研

### 1.1 后端：FastAPI
- **优势**: 
  - 异步支持（AsyncIO），适合处理耗时的音频推理和训练任务。
  - 自动生成 OpenAPI (Swagger) 文档，方便前后端对接。
  - Pydantic 数据验证，确保参数传递准确。
- **集成方案**: 
  - 使用 `BackgroundTasks` 处理音频切分和轻量推理。
  - 使用子进程（Subprocess）或分布式任务队列（可选）管理模型训练。

### 1.2 前端：Vue 3 + Vite
- **优势**:
  - 响应式性能优异，Composition API 适合复杂逻辑管理。
  - Vite 提供极速的热更新开发体验。
- **UI 库**: `Element Plus` 或 `Naive UI`（提供丰富的现代化组件，如进度条、文件上传、滑块等）。
- **状态管理**: `Pinia`。

---

## 2. 核心功能模块调研

### 2.1 模型训练 (Training)
- **现状**: 项目已有 `train.py`。
- **集成需求**:
  - 配置可视化编辑（`.yaml` 文件映射为前端表单）。
  - 实时日志流查看（通过 WebSocket 或日志文件轮询）。
  - TensorBoard 嵌入或数据可视化。

### 2.2 音频切分 (Audio Slicing)
- **工具**: 常用 `slicer-gui` 或 `audio-slicer` 算法。
- **集成方案**: 后端封装切分算法，支持调节阈值、最小长度等参数，生成切分后的数据集。

### 2.3 MSST 音频处理 (Music Source Separation)
- **技术背景**: MSST (Music-Source-Separation-Training) 及 UVR5 (Ultimate Vocal Remover) 架构。
- **关键模型**:
  - **人声分离**: `MDX-Net` (Kim_Vocal_2, BS-Roformer)。
  - **混响分离**: `DeReverb` 系列模型 (如 UVR-DeEcho-DeReverb)。
  - **和声分离**: 特定模型或多声道分离 (Vocals/Lead/Harmony)。
- **依赖库**: 推荐集成 `audio-separator` 库，支持多种 ONNX/PyTorch 格式的模型推理。

---

## 3. 架构设计建议

### 3.1 目录结构
```text
DDSP-SVC-6.3/
├── web/                # 前端 Vue 项目
├── api/                # FastAPI 后端逻辑
├── models/             # 存放 SVC 模型及 MSST 模型
├── data/               # 用户上传及处理后的音频
└── docs/               # 文档
```

### 3.2 交互流程
1. **音频预处理**: 用户上传音频 -> 调用 MSST 模型分离人声 -> 调用 DeReverb 去混响 -> 音频切分。
2. **数据集制作**: 自动生成标注文件 -> 启动数据增强（可选）。
3. **模型训练**: 选择配置 -> 启动 `train.py` -> 监控进度。
4. **推理转换**: 选择模型 -> 上传待转换音频 -> 实时调节变调/参数 -> 输出结果。

---

## 4. 结论与风险点
- **结论**: 采用 FastAPI + Vue 3 是目前 AI 应用的最佳平衡点，能兼顾开发效率与用户体验。
- **风险点**: 
  - **显存竞争**: MSST 处理与 SVC 训练/推理可能同时竞争 GPU。
  - **依赖复杂**: MSST 相关模型需要大量 ONNX 运行时支持。
  - **跨平台**: 需要处理 Windows 路径及环境依赖问题。
