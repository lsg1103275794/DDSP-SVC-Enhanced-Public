<template>
  <div class="view-container">
    <!-- 页面标题区域 -->
    <div class="page-header-enhanced">
      <div class="header-background"></div>
      <div class="header-content">
        <div class="title-section">
          <div class="title-icon-section">
            <div class="icon-container">
              <n-icon size="48" color="#6366f1" class="header-icon">
                <musical-notes-outline />
              </n-icon>
              <div class="icon-glow"></div>
            </div>
          </div>
          <div class="title-text">
            <n-h1 class="main-title">音频数据集预处理</n-h1>
            <n-text depth="3" class="subtitle">构建高质量的训练数据集，支持切分、特征提取和预处理</n-text>
            <div class="feature-badges">
              <n-tag size="small" type="info" class="feature-tag">
                <template #icon>
                  <n-icon><flash-outline /></n-icon>
                </template>
                一键预处理
              </n-tag>
              <n-tag size="small" type="success" class="feature-tag">
                <template #icon>
                  <n-icon><speedometer-outline /></n-icon>
                </template>
                智能切分
              </n-tag>
              <n-tag size="small" type="warning" class="feature-tag">
                <template #icon>
                  <n-icon><analytics-outline /></n-icon>
                </template>
                特征提取
              </n-tag>
            </div>
          </div>
        </div>
        <div class="header-stats">
          <div class="stats-container">
            <div class="stat-card">
              <div class="stat-icon">
                <n-icon size="24" color="#6366f1"><document-text-outline /></n-icon>
              </div>
              <div class="stat-content">
                <n-text strong class="stat-number">{{ uploadFiles.length }}</n-text>
                <n-text depth="3" class="stat-label">已上传文件</n-text>
              </div>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-card">
              <div class="stat-icon">
                <n-icon size="24" color="#f59e0b"><folder-outline /></n-icon>
              </div>
              <div class="stat-content">
                <n-text strong class="stat-number">{{ datasetOptions.length }}</n-text>
                <n-text depth="3" class="stat-label">数据集</n-text>
              </div>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-card">
              <div class="stat-icon">
                <n-icon size="24" color="#10b981"><time-outline /></n-icon>
              </div>
              <div class="stat-content">
                <n-text strong class="stat-number">{{ activeTasks.length }}</n-text>
                <n-text depth="3" class="stat-label">运行任务</n-text>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <n-grid :x-gap="20" :y-gap="20" :cols="12" class="main-grid">
      <!-- 左侧：操作指引 (3.5 span) -->
      <n-gi :span="3.5">
        <n-card class="glass-card guide-card">
          <template #header>
            <div class="card-header-with-icon">
              <n-icon size="20" color="#6366f1"><help-circle-outline /></n-icon>
              <span>快速指引</span>
            </div>
          </template>
          
          <div class="guide-content-compact">
            <div class="guide-step-row">
              <div class="step-number-compact">1</div>
              <div class="step-content-compact">
                <n-text strong class="step-title">准备数据</n-text>
                <n-text depth="3" class="step-desc">上传WAV音频文件</n-text>
              </div>
            </div>

            <div class="guide-divider"></div>

            <div class="guide-step-row">
              <div class="step-number-compact">2</div>
              <div class="step-content-compact">
                <n-text strong class="step-title">配置处理</n-text>
                <n-text depth="3" class="step-desc">设置F0提取器等参数</n-text>
              </div>
            </div>

            <div class="guide-divider"></div>

            <div class="guide-step-row">
              <div class="step-number-compact">3</div>
              <div class="step-content-compact">
                <n-text strong class="step-title">开始预处理</n-text>
                <n-text depth="3" class="step-desc">一键完成所有步骤</n-text>
              </div>
            </div>

            <div class="toolbox-section-compact">
              <div class="toolbox-content-compact">
                <n-text strong class="toolbox-title">音频工具箱</n-text>
                <n-text depth="3" class="toolbox-desc">人声分离、降噪等处理</n-text>
              </div>
              <n-button size="tiny" type="primary" ghost @click="goToAudioToolbox" class="toolbox-link-compact">
                <template #icon>
                  <n-icon size="12"><arrow-forward-outline /></n-icon>
                </template>
              </n-button>
            </div>
          </div>
        </n-card>
      </n-gi>

      <!-- 中间：上传与配置 (6 span) -->
      <n-gi :span="6">
        <n-space vertical size="large">
          <!-- 数据导入区 -->
          <n-card class="glass-card">
            <template #header>
              <div class="card-header-with-icon">
                <n-icon size="20" color="#6366f1"><cloud-upload-outline /></n-icon>
                <span>数据导入</span>
              </div>
            </template>
            <n-upload
              multiple
              directory-dnd
              action="/api/v1/preprocess/upload"
              :headers="uploadHeaders"
              @finish="handleUploadFinish"
              @before-upload="handleBeforeUpload"
            >
              <template #header>
                <n-input 
                  v-model:value="uploadSpkName" 
                  placeholder="请输入角色名 (将存入 dataset_raw/角色名)" 
                  style="margin-bottom: 12px"
                  @click.stop
                >
                  <template #prefix>
                    角色名:
                  </template>
                </n-input>
              </template>
              <n-upload-dragger class="custom-dragger">
                <n-icon size="44" :depth="3" class="upload-icon">
                  <cloud-upload-outline />
                </n-icon>
                <div class="upload-text">拖拽音频或文件夹到此处</div>
                <n-text depth="3" class="upload-hint">
                  仅支持 <b style="color: #818cf8">WAV (单声道, 44.1kHz)</b> 格式
                </n-text>
              </n-upload-dragger>
            </n-upload>
          </n-card>

          <!-- 音频预处理配置 -->
          <n-card class="glass-card preprocess-config-card">
            <template #header>
              <div class="card-header-with-icon">
                <n-icon size="20" color="#6366f1"><settings-outline /></n-icon>
                <span>预处理配置</span>
              </div>
            </template>
            
            <n-form label-placement="left" label-width="120" :model="globalConfig">
              <n-space vertical size="large">
                <n-form-item label="待处理数据集">
                  <n-select 
                    v-model:value="globalConfig.dataset_name" 
                    :options="datasetOptions" 
                    placeholder="请选择 dataset_raw 下的目录"
                    class="styled-select"
                  />
                </n-form-item>

                <n-form-item label="选择模型版本">
                  <n-select v-model:value="globalConfig.model_version" :options="modelVersions" class="styled-select" />
                </n-form-item>

                <n-grid :cols="2" :x-gap="20">
                  <n-gi>
                    <n-form-item label="跳过切分">
                      <n-checkbox v-model:checked="globalConfig.skip_slicing" />
                    </n-form-item>
                  </n-gi>
                  <n-gi>
                    <n-form-item label="切分进程数">
                      <n-input-number 
                        v-model:value="globalConfig.slicer_workers" 
                        :min="1" 
                        :max="32" 
                        button-placement="both"
                        class="styled-input-number"
                      />
                    </n-form-item>
                  </n-gi>
                </n-grid>

                <n-form-item label="声音编码器">
                  <n-select v-model:value="globalConfig.encoder" :options="encoderOptions" class="styled-select" />
                </n-form-item>

                <n-form-item label="F0 提取器">
                  <n-select v-model:value="globalConfig.f0_extractor" :options="f0Options" class="styled-select" />
                </n-form-item>

                <n-form-item label="计算设备">
                  <n-select v-model:value="globalConfig.device" :options="deviceOptions" class="styled-select" />
                </n-form-item>

                <div class="action-bar">
                  <n-button type="primary" block size="large" @click="startGlobalPreprocess" :loading="globalLoading" class="main-action-btn">
                    开始预处理 (一键完成)
                  </n-button>
                </div>
              </n-space>
            </n-form>
          </n-card>
        </n-space>
      </n-gi>

      <!-- 右侧：任务与管理 (3 span) -->
      <n-gi :span="3">
        <n-space vertical size="large">
          <n-card class="glass-card">
            <template #header>
              <div class="card-header-with-icon">
                <n-icon size="20" color="#6366f1"><list-outline /></n-icon>
                <span>任务列表</span>
              </div>
            </template>
            <div v-if="activeTasks.length === 0" class="empty-status-small">
              <n-empty description="暂无运行中的任务" size="small" />
            </div>
            <n-scrollbar v-else style="max-height: 240px">
              <n-list hoverable clickable>
                <n-list-item v-for="task in activeTasks" :key="task.id" class="task-item">
                  <n-space vertical size="small">
                    <div class="flex-between">
                      <n-text strong class="task-name">{{ task.name }}</n-text>
                      <n-tag :type="taskStatusType(task.status)" size="tiny" round>{{ task.status }}</n-tag>
                    </div>
                    <n-progress type="line" :percentage="task.progress" :status="taskStatusType(task.status)" processing height="4" />
                    <n-text depth="3" class="task-msg">{{ task.message }}</n-text>
                  </n-space>
                </n-list-item>
              </n-list>
            </n-scrollbar>
          </n-card>

          <n-card class="glass-card">
            <template #header>
              <div class="card-header-with-icon">
                <n-icon size="20" color="#6366f1"><folder-open-outline /></n-icon>
                <span>文件管理器</span>
              </div>
            </template>
            <n-tabs type="line" size="small" animated>
              <n-tab-pane name="upload" tab="上传">
                <n-list hoverable clickable scrollable style="max-height: 320px">
                  <n-list-item v-for="file in uploadFiles" :key="file.id" @click="selectedFile = file">
                    <n-thing :title="file.filename" :description="formatSize(file.size)" class="file-thing" />
                  </n-list-item>
                </n-list>
              </n-tab-pane>
              <n-tab-pane name="processed" tab="已处理">
                <n-list hoverable scrollable style="max-height: 320px">
                  <n-list-item v-for="file in processedFiles" :key="file.id">
                    <n-thing :title="file.filename" :description="file.type" class="file-thing" />
                  </n-list-item>
                </n-list>
              </n-tab-pane>
            </n-tabs>
          </n-card>
        </n-space>
      </n-gi>
    </n-grid>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue';
import { useMessage } from 'naive-ui';
import { 
  CloudUploadOutline,
  HelpCircleOutline,
  SettingsOutline,
  ListOutline,
  FolderOpenOutline,
  HardwareChipOutline,
  ArrowForwardOutline,
  MusicalNotesOutline,
  FlashOutline,
  SpeedometerOutline,
  AnalyticsOutline,
  DocumentTextOutline,
  TimeOutline
} from '@vicons/ionicons5';

const message = useMessage();
const uploadFiles = ref<any[]>([]);
const processedFiles = ref<any[]>([]);
const selectedFile = ref<any>(null);
const activeTasks = ref<any[]>([]);
const uploadSpkName = ref('');

const globalLoading = ref(false);

// 音频分离相关数据
const separateConfig = reactive({
  model: 'BS-Roformer-Resurrection',
  inputFileId: null as string | null,
  stems: '2',
  quality: 'standard'
});

const separateLoading = ref(false);
const separateResults = ref<any[]>([]);
const filesLoading = ref(false);
const availableFiles = ref<any[]>([]);

// MSST 模型选项
const msstModelOptions = [
  { 
    label: 'BS-Roformer-Resurrection (195MB) - 人声分离', 
    value: 'BS-Roformer-Resurrection',
    description: '人声分离专用模型，处理速度快，效果稳定'
  },
  { 
    label: 'KimMelBandRoformer (870MB) - 卡拉OK分离', 
    value: 'KimMelBandRoformer',
    description: '适用于卡拉OK制作，伴奏质量较高'
  },
  { 
    label: 'mel_band_roformer_karaoke_becruily (1.6GB) - 高质量卡拉OK', 
    value: 'mel_band_roformer_karaoke_becruily',
    description: '高质量卡拉OK分离，需要更长处理时间'
  },
  { 
    label: 'denoise_mel_band_roformer (870MB) - 降噪处理', 
    value: 'denoise_mel_band_roformer',
    description: '专门用于音频降噪处理'
  },
  { 
    label: 'dereverb_echo_mbr_fused (434MB) - 去混响', 
    value: 'dereverb_echo_mbr_fused',
    description: '去除音频中的混响和回声'
  }
];

// 输出轨道选项
const stemOptions = [
  { label: '2轨 (人声+伴奏)', value: '2' },
  { label: '4轨 (人声+鼓+贝斯+其他)', value: '4' }
];

// 处理质量选项
const qualityOptions = [
  { label: '快速 (推荐)', value: 'fast', description: '处理速度快，质量适中' },
  { label: '标准', value: 'standard', description: '处理速度与质量的平衡' },
  { label: '高质量', value: 'high', description: '质量最佳，处理时间较长' }
];

// 上传请求头处理
const uploadHeaders = (file: any) => {
  const headers: any = {};
  if (uploadSpkName.value) {
    headers['X-Speaker-Name'] = encodeURIComponent(uploadSpkName.value);
  }
  
  // 确保能从 native file 获取相对路径
  const relativePath = file.file?.webkitRelativePath || file.fullPath;
  if (relativePath) {
    headers['X-Relative-Path'] = encodeURIComponent(relativePath);
  }
  return headers;
};

const handleBeforeUpload = async () => {
  // 不再在前端进行硬性拦截，而是将校验逻辑交由后端处理
  // 这样可以避免因为 webkitRelativePath 等属性在不同浏览器/环境下的差异导致误报
  return true;
};

// 全局配置
const globalConfig = reactive({
  dataset_name: null as string | null,
  model_version: 'DDSP-SVC 6.3',
  skip_slicing: false,
  slicer_workers: 6,
  encoder: 'contentvec768l12tta2x',
  f0_extractor: 'fcpe',
  device: 'cuda'
});

// 选项
const datasetOptions = ref<any[]>([]);
const modelVersions = [{ label: 'DDSP-SVC 6.3', value: 'DDSP-SVC 6.3' }];
const encoderOptions = [
  { label: 'ContentVec 768L12 (推荐)', value: 'contentvec768l12tta2x' },
  { label: 'Hubert-Soft', value: 'hubertsoft' }
];
const f0Options = [
  { label: 'FCPE (推荐)', value: 'fcpe' },
  { label: 'RMVPE', value: 'rmvpe' },
  { label: 'Crepe', value: 'crepe' }
];
const deviceOptions = [
  { label: 'Auto (优先 CUDA)', value: 'auto' },
  { label: 'CUDA:0', value: 'cuda:0' },
  { label: 'CPU', value: 'cpu' }
];

const refreshingDatasets = ref(false);

const fetchDatasets = async () => {
  try {
    refreshingDatasets.value = true;
    console.log('[DEBUG] Fetching datasets...');
    const res = await fetch('/api/v1/preprocess/datasets').then(r => r.json());
    console.log('[DEBUG] Datasets response:', res);
    
    if (Array.isArray(res)) {
      datasetOptions.value = res.map((d: any) => ({
        label: `${d.name} (${d.file_count}个文件)`,
        value: d.name
      }));
      
      // 如果只有一个数据集，自动选择
      if (res.length >= 1 && !globalConfig.dataset_name) {
         // 优先选择第一个非空的数据集，或者直接选择第一个
         const firstValid = res.find((d: any) => d.file_count > 0) || res[0];
         globalConfig.dataset_name = firstValid.name;
      }
    }
  } catch (e) {
    console.error('[ERROR] Failed to fetch datasets:', e);
    message.error('获取数据集失败: ' + String(e));
  } finally {
    refreshingDatasets.value = false;
  }
};

const fetchFiles = async () => {
  try {
    const [upRes, procRes] = await Promise.all([
      fetch('/api/v1/preprocess/files?type=upload'),
      fetch('/api/v1/preprocess/files?type=sliced')
    ]);
    uploadFiles.value = await upRes.json();
    processedFiles.value = await procRes.json();
  } catch (e) {
    message.error('获取文件列表失败');
  }
};

const pollTaskStatus = (taskId: string) => {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/v1/preprocess/tasks/${taskId}`);
      const data = await res.json();
      
      const taskIndex = activeTasks.value.findIndex(t => t.id === taskId);
      if (taskIndex > -1) {
        activeTasks.value[taskIndex] = {
          ...activeTasks.value[taskIndex],
          status: data.status,
          progress: data.progress,
          message: data.message
        };
        
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
          if (data.status === 'completed') {
            message.success(`${activeTasks.value[taskIndex].name} 已完成`);
            fetchFiles();
            fetchDatasets();
          }
        }
      }
    } catch (e) {
      clearInterval(interval);
    }
  }, 1000);
};

const handleUploadFinish = ({ file, event }: { file: any, event?: any }) => {
  message.success(`已上传: ${file.name}`);
  // 上传成功后刷新数据集列表
  setTimeout(() => {
    fetchDatasets();
  }, 500);
  
  // 提取可能的角色名并提示
  try {
    const res = event?.target?.response ? JSON.parse(event.target.response) : null;
    if (res && res.id) {
      // 如果后端返回的是"default"，显示为"未命名角色"
      const displayName = res.id === 'default' ? '未命名角色' : res.id;
      message.info(`成功存入 dataset_raw/${displayName}`);
    }
  } catch (e) {
    // 忽略解析错误
  }
};

const startGlobalPreprocess = async () => {
  if (!globalConfig.dataset_name) return message.warning('请先选择待处理的数据集');
  
  globalLoading.value = true;
  try {
    const res = await fetch('/api/v1/preprocess/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(globalConfig)
    });
    const data = await res.json();
    activeTasks.value.unshift({
      id: data.task_id,
      name: `全流程预处理: ${globalConfig.dataset_name}`,
      status: 'pending',
      progress: 0,
      message: '准备中...'
    });
    pollTaskStatus(data.task_id);
  } catch (e) {
    message.error('启动失败');
  } finally {
    globalLoading.value = false;
  }
};

const taskStatusType = (status: string) => {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'error';
  if (status === 'running') return 'info';
  return 'default';
};

const formatSize = (bytes: number) => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// 跳转到音频工具箱
const goToAudioToolbox = () => {
  // 这里可以跳转到音频工具箱页面或显示提示信息
  message.info('即将跳转到音频工具箱');
  // 暂时使用 console.log，实际应该实现路由跳转
  console.log('跳转到音频工具箱');
  
  // 可以在这里实现路由跳转逻辑
  // router.push('/audio-toolbox')
};

// 获取可用文件列表
const fetchAvailableFiles = async () => {
  try {
    filesLoading.value = true;
    const [uploadRes, processedRes] = await Promise.all([
      fetch('/api/v1/preprocess/files?type=upload'),
      fetch('/api/v1/preprocess/files?type=separated')
    ]);
    
    const uploadFiles = await uploadRes.json();
    const processedFiles = await processedRes.json();
    
    // 合并并格式化文件列表
    const allFiles = [
      ...uploadFiles.map((file: any) => ({
        label: `${file.filename} - ${formatSize(file.size)} - 原始`,
        value: `upload_${file.id}`,
        ...file,
        type: 'upload'
      })),
      ...processedFiles.map((file: any) => ({
        label: `${file.filename} - ${formatSize(file.size)} - 已处理`,
        value: `processed_${file.id}`,
        ...file,
        type: 'processed'
      }))
    ];
    
    availableFiles.value = allFiles;
  } catch (e) {
    message.error('获取文件列表失败');
  } finally {
    filesLoading.value = false;
  }
};

// 开始分离
const startSeparate = async () => {
  if (!separateConfig.inputFileId || !separateConfig.model) {
    message.warning('请选择输入文件和模型');
    return;
  }
  
  separateLoading.value = true;
  try {
    // 解析文件 ID 获取实际文件信息
    const [prefix, id] = separateConfig.inputFileId.split('_');
    const selectedFile = prefix === 'upload' 
      ? uploadFiles.value.find(f => f.id == id)
      : processedFiles.value.find(f => f.id == id);
    
    if (!selectedFile) {
      message.error('未找到选中的文件');
      return;
    }
    
    // 调用分离 API
    const res = await fetch('/api/v1/preprocess/separate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input_id: selectedFile.path || selectedFile.filename,
        model: separateConfig.model,
        stems: parseInt(separateConfig.stems),
        quality: separateConfig.quality
      })
    });
    
    const data = await res.json();
    
    // 添加到任务列表
    activeTasks.value.unshift({
      id: data.task_id,
      name: `音频分离: ${selectedFile.filename}`,
      status: 'pending',
      progress: 0,
      message: '准备 MSST 分离...'
    });
    
    pollTaskStatus(data.task_id);
    message.success('音频分离已开始，请查看任务进度');
    
  } catch (e) {
    message.error('分离启动失败: ' + String(e));
  } finally {
    separateLoading.value = false;
  }
};

// 播放音频
const playAudio = (path: string) => {
  const audio = new Audio(`file://${path}`);
  audio.play().catch(() => {
    message.error('音频播放失败，请检查文件路径');
  });
};

// 下载结果
const downloadResult = (path: string) => {
  const link = document.createElement('a');
  link.href = `file://${path}`;
  link.download = path.split('/').pop() || 'audio.wav';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

onMounted(() => {
  fetchDatasets();
  fetchFiles();
  fetchAvailableFiles();
});
</script>

<style scoped>
.view-container {
  max-width: 1600px;
  margin: 0 auto;
}

.page-header-centered {
  text-align: center;
  margin-bottom: 40px;
}

.main-title {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle {
  font-size: 16px;
}

.glass-card {
  background: rgba(22, 25, 36, 0.6);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.card-header-with-icon {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  font-size: 16px;
}

.guide-content {
  padding: 4px;
}

.guide-step {
  font-weight: 600;
  margin-bottom: 8px;
  color: #e2e8f0;
}

.guide-text {
  font-size: 13px;
  line-height: 1.6;
}

.folder-viz {
  margin-top: 12px;
  background: rgba(0, 0, 0, 0.3);
  padding: 12px;
  border-radius: 8px;
  border: 1px solid rgba(99, 102, 241, 0.2);
}

.folder-viz pre {
  margin: 0;
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  color: #818cf8;
}

.guide-list {
  padding-left: 18px;
  margin: 8px 0;
}

.guide-list li {
  margin-bottom: 6px;
  font-size: 13px;
}

.custom-dragger {
  padding: 40px !important;
  background: rgba(99, 102, 241, 0.03) !important;
  border: 2px dashed rgba(99, 102, 241, 0.2) !important;
  transition: all 0.3s ease;
}

.custom-dragger:hover {
  background: rgba(99, 102, 241, 0.06) !important;
  border-color: rgba(99, 102, 241, 0.4) !important;
}

.upload-icon {
  color: #6366f1;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 4px;
}

.upload-hint {
  font-size: 12px;
}

.styled-select {
  --n-border-radius: 8px !important;
}

.styled-input-number {
  --n-border-radius: 8px !important;
}

.action-bar {
  margin-top: 12px;
}

.main-action-btn {
  height: 50px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
}

.empty-status-small {
  padding: 40px 0;
}

.task-item {
  padding: 12px !important;
  border-radius: 8px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.02);
}

.flex-between {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-name {
  font-size: 13px;
}

.task-msg {
  font-size: 11px;
}

.file-thing {
  --n-title-font-size: 13px !important;
  --n-description-font-size: 11px !important;
}

.full-height-card {
  display: flex;
  flex-direction: column;
}

:deep(.n-card-header) {
  padding: 16px 20px !important;
}

:deep(.n-card__content) {
  padding: 20px !important;
}

:deep(.n-tabs-nav) {
  margin-bottom: 12px;
}

/* 音频分离工具样式 */
.separator-tip {
  background: rgba(99, 102, 241, 0.05);
  padding: 12px;
  border-radius: 8px;
  border: 1px solid rgba(99, 102, 241, 0.1);
  margin-bottom: 4px;
}

.select-header {
  padding: 8px 12px;
  background: rgba(99, 102, 241, 0.02);
  border-bottom: 1px solid rgba(99, 102, 241, 0.1);
}

.file-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 4px 0;
}

.file-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.file-name {
  font-weight: 500;
  color: #e2e8f0;
}

.file-size {
  font-size: 11px;
  color: #94a3b8;
}

.results-section {
  background: rgba(99, 102, 241, 0.02);
  border: 1px solid rgba(99, 102, 241, 0.1);
  border-radius: 8px;
  padding: 16px;
  margin-top: 12px;
}

.result-item {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 8px;
}

.result-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.result-path {
  font-size: 11px;
  font-family: 'Fira Code', monospace;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 工具箱跳转样式 */
.toolbox-prompt {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  text-align: center;
}

.toolbox-icon {
  margin-bottom: 16px;
  opacity: 0.8;
}

.toolbox-content {
  margin-bottom: 20px;
  line-height: 1.6;
}

.toolbox-actions {
  width: 100%;
  display: flex;
  justify-content: center;
}

.go-toolkits-btn {
  height: 48px;
  padding: 0 24px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
  transition: all 0.3s ease;
}

.go-toolkits-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
}

/* 增强的页面头部样式 */
.page-header-enhanced {
  position: relative;
  margin: -20px -20px 24px -20px;
  padding: 32px 24px;
  background: linear-gradient(135deg, 
    rgba(30, 41, 59, 0.9) 0%, 
    rgba(15, 23, 42, 0.95) 50%, 
    rgba(30, 41, 59, 0.9) 100%);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(99, 102, 241, 0.2);
  overflow: hidden;
}

.header-background {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(circle at 20% 30%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 70%, rgba(245, 158, 11, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 40% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
  pointer-events: none;
}

.header-content {
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
  gap: 40px;
}

.title-section {
  display: flex;
  align-items: center;
  gap: 24px;
}

.title-icon-section {
  position: relative;
}

.icon-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.1) 100%);
  border-radius: 20px;
  border: 1px solid rgba(99, 102, 241, 0.3);
}

.icon-glow {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100px;
  height: 100px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.2) 0%, transparent 70%);
  border-radius: 50%;
  animation: icon-pulse 3s ease-in-out infinite;
}

@keyframes icon-pulse {
  0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
  50% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.1); }
}

.title-text {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.main-title {
  margin: 0 !important;
  font-size: 32px;
  font-weight: 700;
  background: linear-gradient(135deg, #e2e8f0 0%, #f1f5f9 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  font-size: 16px;
  color: #94a3b8;
  line-height: 1.5;
  max-width: 500px;
}

.feature-badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.feature-tag {
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-stats {
  display: flex;
  align-items: center;
}

.stats-container {
  display: flex;
  gap: 24px;
  align-items: center;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  backdrop-filter: blur(8px);
  transition: all 0.3s ease;
}

.stat-card:hover {
  background: rgba(255, 255, 255, 0.08);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.stat-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
}

.stat-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-number {
  font-size: 20px;
  font-weight: 700;
  color: #e2e8f0;
}

.stat-label {
  font-size: 12px;
  color: #94a3b8;
}

.stat-divider {
  width: 1px;
  height: 40px;
  background: linear-gradient(to bottom, 
    transparent 0%, 
    rgba(99, 102, 241, 0.3) 20%, 
    rgba(99, 102, 241, 0.3) 80%, 
    transparent 100%);
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .main-grid {
    grid-template-columns: 1fr !important;
  }
  
  .guide-content {
    max-height: 200px !important;
  }
  
  .header-content {
    flex-direction: column;
    gap: 24px;
    text-align: center;
  }
  
  .stats-container {
    flex-wrap: wrap;
    justify-content: center;
  }
}

@media (max-width: 768px) {
  .file-option {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .result-path {
    max-width: 150px;
  }
}
</style>
