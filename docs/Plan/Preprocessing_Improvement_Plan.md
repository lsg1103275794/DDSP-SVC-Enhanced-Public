# 预处理模块改进计划 (Preprocessing Module Improvement Plan)

**创建日期**: 2026-01-13
**状态**: 待实施
**优先级**: 🔥 高优先级 (URGENT)

---

## 📋 问题分析 (Problem Analysis)

### 当前项目问题 (Current Issues)

#### 1. UI 层面问题

| 问题类别 | 具体问题 | 严重程度 |
|---------|---------|---------|
| **用户体验** | 单一表单界面,缺乏步骤引导 | ⭐⭐⭐⭐⭐ |
| **视觉反馈** | 缺少数据集预览和验证 UI | ⭐⭐⭐⭐ |
| **任务管理** | 进度追踪过于简单,缺少详细信息 | ⭐⭐⭐⭐ |
| **错误处理** | 错误信息不直观,用户难以理解 | ⭐⭐⭐⭐⭐ |
| **文件管理** | 文件列表功能单一,缺少预览和筛选 | ⭐⭐⭐ |

**当前 UI 架构** (PreprocessView.vue):
```
数据导入 → 预处理配置 → 开始处理
    ↓           ↓              ↓
上传文件    选择参数      后台任务轮询
```

**问题点**:
- ❌ 用户不知道数据集是否符合要求
- ❌ 没有预处理前的数据验证步骤
- ❌ 任务失败后缺少详细错误信息和重试机制
- ❌ 无法查看预处理进度的详细日志

#### 2. 后端逻辑问题

| 问题类别 | 具体问题 | 影响 |
|---------|---------|-----|
| **代码复用** | `preprocess_service.py` 与 `preprocess.py` 逻辑重复 | 维护困难 |
| **错误处理** | 缺少失败文件的 skip 目录机制 | 数据丢失 |
| **目录结构** | 扁平化处理丢失了原始目录层级 | 角色分类混乱 |
| **断点续传** | 不支持预处理中断后继续 | 浪费计算资源 |
| **验证机制** | 缺少预处理前的数据集完整性检查 | 运行时错误 |

**当前后端架构**:
```python
# api/services/preprocess_service.py
async def run_full_preprocess(self, req, progress_callback):
    # 1. 扫描文件
    # 2. 切片处理
    # 3. 特征提取 (F0, Units, Volume)
    # ❌ 缺少失败处理
    # ❌ 缺少断点续传
    # ❌ 手动实现特征提取逻辑,与 preprocess.py 重复
```

**根本问题**: 没有复用 `preprocess.py` 的成熟逻辑,而是重新实现了一遍。

---

## 🔍 社区实现参考 (Community Reference)

### SVCFusion / DDSP-SVC 6.3 社区最佳实践

#### 1. GUI 设计 (gui_reflow.py)

**特点**:
- 使用 FreeSimpleGUI 构建多标签页界面
- 参数实时验证和提示
- 模型路径自动检测配置文件

```python
# SVCFusion 社区 UI 架构
layout = [
    [音频设备选择],
    [模型加载 + 验证],
    [推理参数配置 (F0提取器, 采样方法, 步数)],
    [实时状态显示]
]
```

**可借鉴点**:
- ✅ 分步骤的配置流程
- ✅ 参数合法性实时验证
- ✅ 清晰的视觉反馈

#### 2. 预处理逻辑 (allin_preprocess.py)

**核心优势**:
```python
def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, ...):
    # ✅ 1. Skip 目录机制
    path_skipdir = os.path.join(path, 'skip')

    # ✅ 2. 完善的错误处理
    try:
        f0 = f0_extractor.extract(audio, uv_interp=False)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # 正常保存
            np.save(path_f0file, f0)
        else:
            # F0 提取失败,移动到 skip 目录
            shutil.move(path_srcfile, path_skipdir)
            print('[Error] F0 extraction failed: ' + path_srcfile)
    except Exception as e:
        # 记录错误并跳过

    # ✅ 3. Pitch Augmentation 支持
    if use_pitch_aug:
        keyshift = random.uniform(-5, 5)
        pitch_aug_dict[file] = keyshift

    # ✅ 4. Progress Bar
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
```

**与我们当前实现的对比**:

| 功能 | 社区实现 | 我们的实现 | 改进方向 |
|-----|---------|-----------|---------|
| 错误处理 | Skip 目录 + 详细日志 | Try-catch + continue | 🔧 添加 skip 机制 |
| 断点续传 | 检查 .npy 文件是否存在 | 无 | 🔧 实现文件检查 |
| 进度追踪 | tqdm 进度条 | 回调函数 | ✅ 已有 |
| Pitch Aug | 支持 | 不支持 | 🔧 添加 |
| 多进程 | 可选 | 无 | 🔧 添加 |

---

## 🎯 改进方案 (Improvement Plan)

### Phase 1: 后端逻辑重构 (Backend Refactor) - **优先级最高**

#### 目标: 复用 preprocess.py,消除重复代码

**改进策略**:

1. **将 `preprocess.py` 转换为可调用模块**

```python
# preprocess.py (重构后)
class AudioPreprocessor:
    """音频预处理器 (可被 API 服务调用)"""

    def __init__(self, config_path: str, device: str = "cuda"):
        self.args = utils.load_config(config_path)
        self.device = device
        self._init_extractors()

    def _init_extractors(self):
        """初始化特征提取器 (单例模式)"""
        self.f0_extractor = F0_Extractor(...)
        self.volume_extractor = Volume_Extractor(...)
        self.units_encoder = Units_Encoder(...)
        self.mel_extractor = Vocoder(...) if ... else None

    def preprocess_dataset(
        self,
        path: str,
        use_pitch_aug: bool = False,
        progress_callback: Optional[Callable] = None,
        resume: bool = True  # 🔧 NEW: 断点续传
    ):
        """预处理数据集 (支持回调函数)"""
        path_srcdir = os.path.join(path, 'audio')
        path_skipdir = os.path.join(path, 'skip')  # 🔧 NEW: Skip 目录

        filelist = traverse_dir(path_srcdir, ...)

        for i, file in enumerate(filelist):
            # 🔧 NEW: 检查是否已处理 (断点续传)
            if resume and self._is_processed(path, file):
                if progress_callback:
                    progress_callback(i, len(filelist), f"跳过已处理: {file}")
                continue

            try:
                # 原有处理逻辑
                audio, _ = librosa.load(path_srcfile, ...)
                f0 = self.f0_extractor.extract(audio, uv_interp=False)

                # 🔧 NEW: F0 提取失败检测
                uv = f0 == 0
                if len(f0[~uv]) == 0:
                    raise ValueError("F0 extraction failed (all unvoiced)")

                # 插值并保存
                f0[uv] = np.interp(...)
                np.save(path_f0file, f0)
                ...

            except Exception as e:
                # 🔧 NEW: 移动到 skip 目录
                os.makedirs(path_skipdir, exist_ok=True)
                skip_path = os.path.join(path_skipdir, file)
                shutil.move(path_srcfile, skip_path)

                # 🔧 NEW: 记录错误信息
                error_log = os.path.join(path, 'errors.log')
                with open(error_log, 'a') as f:
                    f.write(f"{file}: {str(e)}\n")

                if progress_callback:
                    progress_callback(i, len(filelist), f"失败: {file} - {str(e)}")
                continue

            if progress_callback:
                progress_callback(i, len(filelist), f"完成: {file}")

    def _is_processed(self, path: str, file: str) -> bool:
        """检查文件是否已经处理过"""
        binfile = file + '.npy'
        required_files = [
            os.path.join(path, 'units', binfile),
            os.path.join(path, 'f0', binfile),
            os.path.join(path, 'volume', binfile),
        ]
        return all(os.path.exists(f) for f in required_files)

    def validate_dataset(self, path: str) -> dict:
        """🔧 NEW: 预处理前验证数据集"""
        path_srcdir = os.path.join(path, 'audio')

        if not os.path.exists(path_srcdir):
            return {"valid": False, "error": "audio 目录不存在"}

        filelist = traverse_dir(path_srcdir, ...)

        if len(filelist) == 0:
            return {"valid": False, "error": "未找到音频文件"}

        # 检查音频格式
        invalid_files = []
        for file in filelist:
            path_file = os.path.join(path_srcdir, file)
            try:
                audio, sr = librosa.load(path_file, sr=None)
                if sr != 44100:
                    invalid_files.append(f"{file} (采样率: {sr}Hz)")
            except Exception as e:
                invalid_files.append(f"{file} (无法读取)")

        if invalid_files:
            return {
                "valid": False,
                "warning": f"发现 {len(invalid_files)} 个不合规文件",
                "invalid_files": invalid_files[:10]  # 最多显示 10 个
            }

        return {
            "valid": True,
            "total_files": len(filelist),
            "estimated_time": len(filelist) * 5  # 秒 (粗略估算)
        }
```

2. **更新 PreprocessService 复用新模块**

```python
# api/services/preprocess_service.py (重构后)
from preprocess import AudioPreprocessor

class PreprocessService:
    def __init__(self):
        self.preprocessor_cache = {}  # 缓存预处理器实例

    def _get_preprocessor(self, config_path: str) -> AudioPreprocessor:
        """获取或创建预处理器实例 (避免重复加载模型)"""
        if config_path not in self.preprocessor_cache:
            self.preprocessor_cache[config_path] = AudioPreprocessor(
                config_path, device=DEVICE
            )
        return self.preprocessor_cache[config_path]

    async def validate_dataset_before_preprocess(self, dataset_name: str) -> dict:
        """🔧 NEW: 预处理前验证"""
        src_dir = os.path.join(self.raw_dir, dataset_name)
        config_path = os.path.join(BASE_DIR, "configs", "reflow.yaml")

        preprocessor = self._get_preprocessor(config_path)

        # 临时创建 audio 目录软链接 (为了复用验证逻辑)
        temp_path = os.path.join(BASE_DIR, "temp_validate", dataset_name)
        os.makedirs(temp_path, exist_ok=True)
        audio_link = os.path.join(temp_path, "audio")
        if not os.path.exists(audio_link):
            os.symlink(src_dir, audio_link, target_is_directory=True)

        result = preprocessor.validate_dataset(temp_path)

        # 清理临时目录
        shutil.rmtree(temp_path)

        return result

    async def run_full_preprocess(
        self, req: PreprocessRequest, progress_callback: Callable
    ):
        """运行完整预处理 (复用 AudioPreprocessor)"""
        dataset_name = req.dataset_name

        # 1. 准备数据目录
        train_path = os.path.join(BASE_DIR, "data", "train")
        val_path = os.path.join(BASE_DIR, "data", "val")

        # 2. 数据集复制/切片 (保留原有逻辑)
        progress_callback(10, "正在准备数据集...")
        await self._prepare_dataset(req, train_path, val_path)

        # 3. 初始化预处理器
        config_path = os.path.join(BASE_DIR, "configs", "reflow.yaml")
        preprocessor = self._get_preprocessor(config_path)

        # 4. 执行预处理 (复用 preprocess.py 逻辑)
        progress_callback(30, "正在提取训练集特征...")

        def wrapped_callback(current, total, message):
            # 将 preprocessor 的进度转换为 API 进度 (30-80%)
            progress = 30 + int((current / total) * 50)
            progress_callback(progress, message)

        preprocessor.preprocess_dataset(
            train_path,
            use_pitch_aug=req.use_pitch_aug if hasattr(req, 'use_pitch_aug') else False,
            progress_callback=wrapped_callback,
            resume=True  # 🔧 NEW: 支持断点续传
        )

        progress_callback(85, "正在提取验证集特征...")
        preprocessor.preprocess_dataset(
            val_path,
            use_pitch_aug=False,
            progress_callback=lambda c, t, m: progress_callback(
                85 + int((c/t) * 10), m
            ),
            resume=True
        )

        progress_callback(100, "预处理完成")
```

**改进效果**:
- ✅ 消除代码重复 (~300 行代码)
- ✅ 统一维护预处理逻辑
- ✅ 添加 skip 目录和错误日志
- ✅ 支持断点续传
- ✅ 预处理前验证

---

### Phase 2: UI 现代化改造 (UI Modernization)

#### 目标: 从简单表单升级为步骤化向导

**新 UI 架构设计**:

```
┌─────────────────────────────────────────────────┐
│  预处理向导 (Preprocessing Wizard)              │
│                                                  │
│  ┌──┬──┬──┬──┐                                 │
│  │1 │2 │3 │4 │  步骤指示器                     │
│  └──┴──┴──┴──┘                                 │
│   ↑  ↑  ↑  ↑                                   │
│   │  │  │  └─ 执行与监控                       │
│   │  │  └─── 参数配置                          │
│   │  └────── 数据验证                          │
│   └───────── 数据集选择                        │
└─────────────────────────────────────────────────┘
```

**Step 1: 数据集选择与预览**

```vue
<!-- Step1: 数据集选择 -->
<template>
  <n-space vertical size="large">
    <!-- 数据集选择器 -->
    <n-card title="选择数据集">
      <n-select
        v-model:value="selectedDataset"
        :options="datasetOptions"
        size="large"
        @update:value="onDatasetChange"
      >
        <template #header>
          <n-text>从 dataset_raw 中选择待处理的数据集</n-text>
        </template>
      </n-select>
    </n-card>

    <!-- 🔧 NEW: 数据集预览 -->
    <n-card title="数据集概览" v-if="selectedDataset">
      <n-descriptions :column="3">
        <n-descriptions-item label="文件总数">
          <n-tag type="info">{{ datasetInfo.totalFiles }}</n-tag>
        </n-descriptions-item>
        <n-descriptions-item label="总时长">
          <n-tag type="info">{{ datasetInfo.totalDuration }}</n-tag>
        </n-descriptions-item>
        <n-descriptions-item label="平均采样率">
          <n-tag :type="datasetInfo.avgSampleRate === 44100 ? 'success' : 'warning'">
            {{ datasetInfo.avgSampleRate }} Hz
          </n-tag>
        </n-descriptions-item>
      </n-descriptions>

      <!-- 🔧 NEW: 文件列表预览 (前 10 个) -->
      <n-divider />
      <n-list bordered hoverable>
        <n-list-item v-for="file in datasetInfo.sampleFiles" :key="file.name">
          <n-thing :title="file.name">
            <template #description>
              <n-space size="small">
                <n-tag size="tiny">{{ file.duration }}s</n-tag>
                <n-tag size="tiny">{{ file.sampleRate }}Hz</n-tag>
                <n-tag size="tiny">{{ file.channels }}声道</n-tag>
              </n-space>
            </template>
          </n-thing>
        </n-list-item>
      </n-list>
    </n-card>

    <!-- 操作按钮 -->
    <n-button type="primary" size="large" block @click="nextStep" :disabled="!selectedDataset">
      下一步：数据验证
    </n-button>
  </n-space>
</template>

<script setup lang="ts">
// 获取数据集详细信息
const fetchDatasetInfo = async (datasetName: string) => {
  const res = await fetch(`/api/v1/preprocess/datasets/${datasetName}/info`);
  datasetInfo.value = await res.json();
};
</script>
```

**Step 2: 数据验证**

```vue
<!-- Step2: 数据验证 -->
<template>
  <n-space vertical size="large">
    <n-card title="数据集合规性检查">
      <!-- 🔧 NEW: 验证进度 -->
      <n-spin :show="validating">
        <n-result
          v-if="!validating && validationResult"
          :status="validationResult.valid ? 'success' : 'warning'"
          :title="validationResult.valid ? '数据集检查通过' : '数据集存在问题'"
        >
          <template #icon>
            <n-icon v-if="validationResult.valid" color="#52c41a" size="64">
              <checkmark-circle-outline />
            </n-icon>
            <n-icon v-else color="#faad14" size="64">
              <alert-circle-outline />
            </n-icon>
          </template>

          <template #footer>
            <!-- 验证详情 -->
            <n-descriptions :column="2" bordered>
              <n-descriptions-item label="总文件数">
                {{ validationResult.totalFiles }}
              </n-descriptions-item>
              <n-descriptions-item label="合规文件">
                <n-tag type="success">{{ validationResult.validFiles }}</n-tag>
              </n-descriptions-item>
              <n-descriptions-item label="预计处理时间">
                {{ Math.ceil(validationResult.estimatedTime / 60) }} 分钟
              </n-descriptions-item>
              <n-descriptions-item label="问题文件" v-if="validationResult.invalidFiles?.length > 0">
                <n-tag type="warning">{{ validationResult.invalidFiles.length }}</n-tag>
              </n-descriptions-item>
            </n-descriptions>

            <!-- 🔧 NEW: 问题文件列表 -->
            <n-collapse v-if="validationResult.invalidFiles?.length > 0" style="margin-top: 16px">
              <n-collapse-item title="查看问题文件">
                <n-list bordered>
                  <n-list-item v-for="file in validationResult.invalidFiles" :key="file">
                    <n-text type="warning">{{ file }}</n-text>
                  </n-list-item>
                </n-list>

                <!-- 🔧 NEW: 快速修复选项 -->
                <n-space style="margin-top: 12px">
                  <n-button type="info" size="small" @click="goToToolbox">
                    前往工具箱修复
                  </n-button>
                  <n-button type="warning" size="small" @click="skipInvalidFiles">
                    跳过问题文件继续
                  </n-button>
                </n-space>
              </n-collapse-item>
            </n-collapse>
          </template>
        </n-result>
      </n-spin>
    </n-card>

    <!-- 操作按钮 -->
    <n-space>
      <n-button @click="prevStep">上一步</n-button>
      <n-button
        type="primary"
        @click="nextStep"
        :disabled="!validationResult?.valid"
      >
        下一步：参数配置
      </n-button>
    </n-space>
  </n-space>
</template>

<script setup lang="ts">
const runValidation = async () => {
  validating.value = true;
  try {
    const res = await fetch(`/api/v1/preprocess/validate`, {
      method: 'POST',
      body: JSON.stringify({ dataset_name: selectedDataset.value })
    });
    validationResult.value = await res.json();
  } finally {
    validating.value = false;
  }
};
</script>
```

**Step 3: 参数配置** (保留现有UI,微调)

```vue
<!-- Step3: 参数配置 -->
<template>
  <n-space vertical size="large">
    <n-card title="预处理参数">
      <n-form label-placement="left" label-width="140">
        <!-- F0 提取器 -->
        <n-form-item label="F0 提取器">
          <n-select v-model:value="config.f0_extractor" :options="f0Options">
            <template #header>
              <n-alert type="info" size="small">
                推荐使用 FCPE (速度快) 或 RMVPE (质量高)
              </n-alert>
            </template>
          </n-select>
        </n-form-item>

        <!-- 🔧 NEW: Pitch Augmentation -->
        <n-form-item label="音高增强">
          <n-switch v-model:value="config.use_pitch_aug" />
          <template #feedback>
            <n-text depth="3">
              启用后将随机移调 ±5 半音,增加数据多样性
            </n-text>
          </template>
        </n-form-item>

        <!-- 🔧 NEW: 断点续传 -->
        <n-form-item label="断点续传">
          <n-switch v-model:value="config.resume" />
          <template #feedback>
            <n-text depth="3">
              跳过已处理的文件,避免重复计算
            </n-text>
          </template>
        </n-form-item>

        <!-- 其他参数... -->
      </n-form>
    </n-card>

    <!-- 🔧 NEW: 参数预览 -->
    <n-card title="配置预览">
      <n-code :code="JSON.stringify(config, null, 2)" language="json" />
    </n-card>

    <n-space>
      <n-button @click="prevStep">上一步</n-button>
      <n-button type="primary" @click="startPreprocess">
        开始预处理
      </n-button>
    </n-space>
  </n-space>
</template>
```

**Step 4: 执行与监控**

```vue
<!-- Step4: 执行与监控 -->
<template>
  <n-space vertical size="large">
    <!-- 🔧 NEW: 实时日志流 -->
    <n-card title="预处理进度">
      <n-progress
        type="line"
        :percentage="taskProgress"
        :status="taskStatus"
        processing
        height="20"
      >
        <template #default>{{ taskMessage }}</template>
      </n-progress>

      <!-- 🔧 NEW: 详细日志 -->
      <n-divider />
      <n-scrollbar style="max-height: 400px">
        <n-log
          :lines="taskLogs"
          :rows="15"
          language="log"
          trim
        />
      </n-scrollbar>
    </n-card>

    <!-- 🔧 NEW: 失败文件汇总 -->
    <n-card title="处理结果" v-if="taskStatus === 'completed'">
      <n-result status="success" title="预处理完成">
        <template #footer>
          <n-descriptions :column="2" bordered>
            <n-descriptions-item label="成功文件">
              <n-tag type="success">{{ processedFiles }}</n-tag>
            </n-descriptions-item>
            <n-descriptions-item label="失败文件">
              <n-tag type="error">{{ failedFiles.length }}</n-tag>
            </n-descriptions-item>
          </n-descriptions>

          <!-- 🔧 NEW: 失败文件列表 -->
          <n-collapse v-if="failedFiles.length > 0" style="margin-top: 16px">
            <n-collapse-item title="查看失败文件">
              <n-list bordered>
                <n-list-item v-for="file in failedFiles" :key="file.name">
                  <n-thing :title="file.name">
                    <template #description>
                      <n-text type="error">{{ file.error }}</n-text>
                    </template>
                  </n-thing>
                </n-list-item>
              </n-list>

              <!-- 🔧 NEW: 重试失败文件 -->
              <n-button type="primary" size="small" @click="retryFailedFiles" style="margin-top: 12px">
                重试失败文件
              </n-button>
            </n-collapse-item>
          </n-collapse>
        </template>
      </n-result>
    </n-card>
  </n-space>
</template>

<script setup lang="ts">
const pollTaskStatus = async (taskId: string) => {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/v1/preprocess/tasks/${taskId}`);
    const data = await res.json();

    taskProgress.value = data.progress;
    taskMessage.value = data.message;
    taskStatus.value = data.status;

    // 🔧 NEW: 获取详细日志
    if (data.logs) {
      taskLogs.value = data.logs.split('\n');
    }

    // 🔧 NEW: 获取失败文件列表
    if (data.failed_files) {
      failedFiles.value = data.failed_files;
    }

    if (data.status === 'completed' || data.status === 'failed') {
      clearInterval(interval);
    }
  }, 1000);
};
</script>
```

---

### Phase 3: API 路由扩展 (API Routes Extension)

**新增 API 端点**:

```python
# api/routes/preprocess.py (新增)

@router.get("/datasets/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """🔧 NEW: 获取数据集详细信息"""
    src_dir = os.path.join(preprocess_service.raw_dir, dataset_name)

    audio_files = []
    total_duration = 0
    sample_rates = []

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(('.wav', '.flac', '.mp3')):
                file_path = os.path.join(root, f)
                try:
                    audio, sr = librosa.load(file_path, sr=None, duration=1)  # 只读 1 秒预览
                    duration = librosa.get_duration(path=file_path)

                    audio_files.append({
                        "name": f,
                        "duration": round(duration, 2),
                        "sampleRate": sr,
                        "channels": 1 if len(audio.shape) == 1 else audio.shape[0]
                    })
                    total_duration += duration
                    sample_rates.append(sr)
                except:
                    continue

    return {
        "totalFiles": len(audio_files),
        "totalDuration": round(total_duration / 60, 2),  # 分钟
        "avgSampleRate": int(np.mean(sample_rates)) if sample_rates else 0,
        "sampleFiles": audio_files[:10]  # 前 10 个文件
    }

@router.post("/validate")
async def validate_dataset(req: ValidateRequest):
    """🔧 NEW: 预处理前验证"""
    result = await preprocess_service.validate_dataset_before_preprocess(
        req.dataset_name
    )
    return result

@router.get("/tasks/{task_id}/logs")
async def get_task_logs(task_id: str):
    """🔧 NEW: 获取任务详细日志"""
    if task_id not in tasks_db:
        raise HTTPException(404, "Task not found")

    # 读取日志文件
    log_file = os.path.join(BASE_DIR, "logs", f"{task_id}.log")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.read()
    else:
        logs = ""

    return {"logs": logs, **tasks_db[task_id]}

@router.post("/retry-failed")
async def retry_failed_files(req: RetryRequest, background_tasks: BackgroundTasks):
    """🔧 NEW: 重试失败文件"""
    task_id = f"retry_{uuid.uuid4().hex[:8]}"
    update_task(task_id, "pending", 0, "准备重试失败文件...")

    # 从 skip 目录读取失败文件
    skip_dir = os.path.join(BASE_DIR, "data", "train", "skip")
    failed_files = [f for f in os.listdir(skip_dir) if f.endswith('.wav')]

    # 移回 audio 目录
    audio_dir = os.path.join(BASE_DIR, "data", "train", "audio", req.dataset_name)
    for f in failed_files:
        shutil.move(
            os.path.join(skip_dir, f),
            os.path.join(audio_dir, f)
        )

    # 重新执行预处理
    background_tasks.add_task(run_preprocess_task, task_id, req)
    return {"task_id": task_id, "retry_count": len(failed_files)}
```

---

## 📅 实施计划 (Implementation Timeline)

### Week 1: 后端重构 (Backend Refactor)

| 任务 | 优先级 | 预计时间 | 负责人 |
|-----|-------|---------|-------|
| 重构 preprocess.py 为类接口 | P0 | 2天 | Backend |
| 添加 skip 目录机制 | P0 | 0.5天 | Backend |
| 实现断点续传 | P1 | 1天 | Backend |
| 添加数据集验证 | P1 | 1天 | Backend |
| 单元测试 | P1 | 0.5天 | Backend |

### Week 2: API 扩展 + UI 基础 (API + UI Foundation)

| 任务 | 优先级 | 预计时间 | 负责人 |
|-----|-------|---------|-------|
| 新增 API 端点 | P0 | 1天 | Backend |
| UI Step 1-2 (数据集选择+验证) | P0 | 2天 | Frontend |
| UI Step 3 (参数配置) | P1 | 1天 | Frontend |
| UI Step 4 (执行监控) | P1 | 1天 | Frontend |

### Week 3: 集成测试与优化 (Integration & Optimization)

| 任务 | 优先级 | 预计时间 | 负责人 |
|-----|-------|---------|-------|
| 前后端集成测试 | P0 | 1天 | Full Stack |
| UI/UX 优化 | P1 | 1天 | Frontend |
| 性能优化 | P1 | 1天 | Backend |
| 文档更新 | P2 | 0.5天 | All |
| 用户验收测试 | P0 | 0.5天 | All |

**总计**: 3 周 (约 15 个工作日)

---

## 🎯 预期效果 (Expected Outcomes)

### 用户体验改进

| 改进点 | 改进前 | 改进后 | 提升 |
|-------|--------|--------|------|
| 操作流程 | 单一表单,容易出错 | 4 步向导,清晰引导 | ⭐⭐⭐⭐⭐ |
| 错误提示 | "预处理失败" | 详细错误 + 修复建议 | ⭐⭐⭐⭐⭐ |
| 失败恢复 | 重新开始 | 断点续传 + 重试失败 | ⭐⭐⭐⭐ |
| 数据验证 | 运行时报错 | 预处理前检查 | ⭐⭐⭐⭐ |

### 技术指标改进

| 指标 | 改进前 | 改进后 |
|-----|--------|--------|
| 代码重复度 | ~300 行重复代码 | 0 (完全复用) |
| 错误处理覆盖率 | ~30% | 95% |
| 用户操作步骤 | 1 步 (易出错) | 4 步 (引导式) |
| 失败恢复时间 | 重新运行 (浪费) | 断点续传 (秒级) |

---

## 🔧 技术栈 (Tech Stack)

### 前端
- **UI 组件**: Naive UI (n-steps, n-result, n-log)
- **状态管理**: Vue 3 Composition API
- **API 调用**: Axios + Async/Await

### 后端
- **预处理模块**: 重构 preprocess.py 为类接口
- **任务管理**: FastAPI BackgroundTasks
- **日志系统**: Python logging + 文件存储
- **错误追踪**: Skip 目录 + errors.log

---

## 📝 附录 (Appendix)

### A. 代码对比示例

**改进前** (preprocess_service.py:269-322):
```python
# 手动实现特征提取 (与 preprocess.py 重复)
def process_dataset(path, is_train=True):
    path_srcdir = os.path.join(path, "audio", dataset_name)
    filelist = utils.traverse_dir(path_srcdir, ...)

    for i, file in enumerate(filelist):
        # ... 手动加载音频
        audio, _ = librosa.load(path_srcfile, sr=args.data.sampling_rate)

        # ... 手动提取特征
        volume = volume_extractor.extract(audio)
        units_t = units_encoder.encode(audio_t, ...)
        f0 = f0_extractor.extract(audio, uv_interp=False)

        # ... 插值和保存
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(...)
            np.save(path_f0file, f0)
            # ...
```

**改进后**:
```python
# 直接调用 AudioPreprocessor
preprocessor = AudioPreprocessor(config_path)
preprocessor.preprocess_dataset(
    train_path,
    progress_callback=wrapped_callback,
    resume=True
)
```

**代码行数减少**: ~250 行 → ~10 行 (96% 减少)

---

**文档版本**: 1.0
**创建人**: AI Assistant
**审核状态**: 待审核
