<template>
  <div class="tools-container">
    <n-space vertical size="large">
      <!-- 顶部标题 -->
      <div class="page-header">
        <div class="header-content">
          <div class="title-section">
            <n-icon size="36" color="#6366f1" class="header-icon">
              <musical-notes-outline />
            </n-icon>
            <div class="title-text">
              <n-h2 prefix="bar" class="main-title">音频工具箱</n-h2>
              <n-text depth="3" class="subtitle">简单高效的基础音频处理工具，支持批量操作和AI分离</n-text>
            </div>
          </div>
        </div>
      </div>

      <!-- 主要内容区域 -->
      <n-card class="main-card">
        <n-tabs type="line" animated>
          <!-- 切音机 -->
          <n-tab-pane name="slicer" tab="切音机">
            <n-space vertical size="large" class="pane-content">
              <div class="input-group">
                <n-text strong class="label-with-icon">输入文件夹</n-text>
                <n-input v-model:value="slicerConfig.inputDir" placeholder="请输入待处理音频的文件夹路径" />
              </div>
              
              <div class="input-group">
                <n-text strong class="label-with-icon">输出文件夹</n-text>
                <n-input v-model:value="slicerConfig.outputDir" placeholder="请输入处理后的音频保存路径" />
              </div>

              <div class="input-group">
                <div class="label-row">
                  <n-text strong>最大时长 (s)</n-text>
                  <n-badge :value="slicerConfig.maxDuration" color="#6366f1" />
                </div>
                <n-slider v-model:value="slicerConfig.maxDuration" :min="1" :max="60" :step="1" />
              </div>

              <n-button type="primary" block size="large" :loading="processing" @click="startSlicing">
                开始切片
              </n-button>
            </n-space>
          </n-tab-pane>

          <!-- 重采样 -->
          <n-tab-pane name="resample" tab="重采样到 44.1 kHz">
            <n-space vertical size="large" class="pane-content">
              <div class="input-group">
                <n-text strong class="label-with-icon">输入文件夹</n-text>
                <n-input v-model:value="resampleConfig.inputDir" placeholder="请输入音频文件夹路径" />
              </div>
              
              <div class="input-group">
                <n-text strong class="label-with-icon">输出文件夹</n-text>
                <n-input v-model:value="resampleConfig.outputDir" placeholder="请输入重采样后的保存路径" />
              </div>

              <n-button type="primary" block size="large" :loading="processing" @click="startResampling">
                开始重采样
              </n-button>
            </n-space>
          </n-tab-pane>

          <!-- 批量转 WAV -->
          <n-tab-pane name="convert" tab="批量转 WAV">
            <n-space vertical size="large" class="pane-content">
              <div class="input-group">
                <n-text strong class="label-with-icon">输入文件夹</n-text>
                <n-input v-model:value="convertConfig.inputDir" placeholder="请输入音频文件夹路径" />
              </div>
              
              <div class="input-group">
                <n-text strong class="label-with-icon">输出文件夹</n-text>
                <n-input v-model:value="convertConfig.outputDir" placeholder="请输入转换后的保存路径" />
              </div>

              <n-button type="primary" block size="large" :loading="processing" @click="startConverting">
                开始转换
              </n-button>
            </n-space>
          </n-tab-pane>

          <!-- MSST音频分离 -->
          <n-tab-pane name="msst" tab="MSST 音频分离">
            <n-space vertical size="large" class="pane-content">
              <div class="msst-description">
                <n-text depth="3">
                  使用深度学习模型进行音频分离，支持人声提取、卡拉OK制作等功能。
                  <br/>适用于分离人声、伴奏、钢琴等不同音轨。
                </n-text>
              </div>

              <div class="input-group">
                <n-text strong class="label-with-icon">选择模型</n-text>
                <n-select 
                  v-model:value="msstConfig.model" 
                  :options="msstModelOptions" 
                  placeholder="请选择分离模型"
                  class="msst-select"
                >
                  <template #header>
                    <div class="model-header">
                      <n-text depth="3">推荐：BS-Roformer-Resurrection（人声分离专用，195MB）</n-text>
                    </div>
                  </template>
                </n-select>
              </div>

              <div class="input-group">
                <n-text strong class="label-with-icon">输入文件</n-text>
                <n-input 
                  v-model:value="msstConfig.inputFile" 
                  placeholder="请输入音频文件路径或选择上传文件"
                />
              </div>

              <div class="input-group">
                <n-text strong class="label-with-icon">输出目录</n-text>
                <n-input 
                  v-model:value="msstConfig.outputDir" 
                  placeholder="请输入分离结果的保存目录"
                />
              </div>

              <div class="input-group">
                <div class="label-row">
                  <n-text strong>输出轨道数</n-text>
                  <n-badge :value="msstConfig.stems" color="#6366f1" />
                </div>
                <n-select v-model:value="msstConfig.stems" :options="stemOptions" class="msst-select" />
              </div>

              <div class="input-group">
                <div class="label-row">
                  <n-text strong>处理质量</n-text>
                  <n-tag :type="getQualityType(msstConfig.quality)" size="small">
                    {{ getQualityText(msstConfig.quality) }}
                  </n-tag>
                </div>
                <n-select v-model:value="msstConfig.quality" :options="qualityOptions" class="msst-select" />
              </div>

              <n-button type="primary" block size="large" :loading="processing" @click="startMSSTSeparation" class="msst-button">
                开始音频分离
              </n-button>

              <!-- 分离结果展示 -->
              <div v-if="msstResults.length > 0" class="results-section">
                <n-divider>分离结果</n-divider>
                <n-space vertical size="small">
                  <div v-for="result in msstResults" :key="result.filename" class="result-item">
                    <div class="result-header">
                      <n-text strong>{{ result.filename }}</n-text>
                      <n-space>
                        <n-button size="tiny" @click="playAudio(result.path)">
                          <template #icon><n-icon><play-outline /></n-icon></template>
                          播放
                        </n-button>
                        <n-button size="tiny" type="primary" ghost @click="downloadResult(result.path)">
                          <template #icon><n-icon><download-outline /></n-icon></template>
                          下载
                        </n-button>
                      </n-space>
                    </div>
                    <n-text depth="3" class="result-path">{{ result.path }}</n-text>
                  </div>
                </n-space>
              </div>
            </n-space>
          </n-tab-pane>
        </n-tabs>
      </n-card>

      <!-- 任务进度监控 -->
      <n-card v-if="activeTasks.length > 0" title="正在处理的任务" size="small" class="task-card">
        <n-list hoverable clickable>
          <n-list-item v-for="task in activeTasks" :key="task.id">
            <n-space vertical size="small">
              <div class="task-info">
                <n-text strong>{{ task.name }}</n-text>
                <n-tag :type="getStatusType(task.status)" size="small" round>
                  {{ task.statusText }}
                </n-tag>
              </div>
              <n-progress
                type="line"
                :percentage="task.progress"
                :status="getProgressStatus(task.status)"
                indicator-placement="inside"
                processing
              />
              <n-text depth="3" size="small">{{ task.message }}</n-text>
            </n-space>
          </n-list-item>
        </n-list>
      </n-card>
    </n-space>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue';
import { useMessage } from 'naive-ui';
import { 
  MusicalNotesOutline,
  PlayOutline,
  DownloadOutline
} from '@vicons/ionicons5';

const message = useMessage();
const processing = ref(false);

// 配置项
const slicerConfig = reactive({
  inputDir: '',
  outputDir: '',
  maxDuration: 10
});

const resampleConfig = reactive({
  inputDir: '',
  outputDir: ''
});

const convertConfig = reactive({
  inputDir: '',
  outputDir: ''
});

// MSST 配置
const msstConfig = reactive({
  model: 'BS-Roformer-Resurrection',
  inputFile: '',
  outputDir: '',
  stems: '2',
  quality: 'standard'
});

const msstResults = ref<any[]>([]);

// MSST 模型选项
const msstModelOptions = [
  { 
    label: 'BS-Roformer-Resurrection (195MB) - 人声分离专用', 
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
    label: 'dereverb_echo_mbr_fused (434MB) - 去混响处理', 
    value: 'dereverb_echo_mbr_fused',
    description: '专门用于去除混响和回声'
  }
];

// 轨道数选项
const stemOptions = [
  { label: '2轨 (人声+伴奏)', value: '2' },
  { label: '4轨 (人声+钢琴+其他)', value: '4' }
];

// 质量选项
const qualityOptions = [
  { label: '快速', value: 'fast', description: '处理速度快，质量一般' },
  { label: '标准', value: 'standard', description: '平衡速度和质量' },
  { label: '高质量', value: 'high', description: '质量最佳，处理较慢' }
];

// 任务列表
interface Task {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  statusText: string;
  progress: number;
  message: string;
}

const activeTasks = ref<Task[]>([]);

// 状态辅助函数
const getStatusType = (status: string) => {
  switch (status) {
    case 'completed': return 'success';
    case 'failed': return 'error';
    case 'running': return 'info';
    default: return 'default';
  }
};

// 质量辅助函数
const getQualityType = (quality: string) => {
  switch (quality) {
    case 'fast': return 'warning';
    case 'standard': return 'info';
    case 'high': return 'success';
    default: return 'default';
  }
};

const getQualityText = (quality: string) => {
  switch (quality) {
    case 'fast': return '快速';
    case 'standard': return '标准';
    case 'high': return '高质量';
    default: return '标准';
  }
};

const getProgressStatus = (status: string) => {
  switch (status) {
    case 'completed': return 'success';
    case 'failed': return 'error';
    default: return 'info';
  }
};

// 模拟任务启动改为真实启动
const startSlicing = async () => {
  if (!slicerConfig.inputDir || !slicerConfig.outputDir) {
    return message.warning('请填写输入和输出路径');
  }
  await startToolTask('slice', {
    input_dir: slicerConfig.inputDir,
    output_dir: slicerConfig.outputDir,
    max_duration: slicerConfig.maxDuration
  }, `音频切片: ${slicerConfig.inputDir}`);
};

const startResampling = async () => {
  if (!resampleConfig.inputDir || !resampleConfig.outputDir) {
    return message.warning('请填写输入和输出路径');
  }
  await startToolTask('resample', {
    input_dir: resampleConfig.inputDir,
    output_dir: resampleConfig.outputDir
  }, `重采样: ${resampleConfig.inputDir}`);
};

const startConverting = async () => {
  if (!convertConfig.inputDir || !convertConfig.outputDir) {
    return message.warning('请填写输入和输出路径');
  }
  await startToolTask('convert', {
    input_dir: convertConfig.inputDir,
    output_dir: convertConfig.outputDir
  }, `格式转换: ${convertConfig.inputDir}`);
};

const startToolTask = async (type: string, payload: any, name: string) => {
  processing.value = true;
  try {
    const res = await fetch(`/api/v1/preprocess/tool/${type}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const newTask: Task = {
      id: data.task_id,
      name,
      status: 'pending',
      statusText: '准备中',
      progress: 0,
      message: '正在初始化任务...'
    };
    activeTasks.value.unshift(newTask);
    pollTaskStatus(data.task_id);
    message.success('任务已启动');
  } catch (e) {
    message.error('任务启动失败');
  } finally {
    processing.value = false;
  }
};

const pollTaskStatus = (taskId: string) => {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/v1/preprocess/tasks/${taskId}`);
      const data = await res.json();
      
      const task = activeTasks.value.find(t => t.id === taskId);
      if (task) {
        task.status = data.status;
        task.progress = data.progress;
        task.message = data.message;
        
        if (data.status === 'completed') {
          task.statusText = '已完成';
          clearInterval(interval);
        } else if (data.status === 'failed') {
          task.statusText = '失败';
          clearInterval(interval);
        } else if (data.status === 'running') {
          task.statusText = '处理中';
        }
      } else {
        clearInterval(interval);
      }
    } catch (e) {
      console.error('Polling error:', e);
      clearInterval(interval);
    }
  }, 1000);
};

// 移除旧的模拟函数
const addTask = (name: string) => {};
const simulateProgress = (id: string) => {};

// MSST 音频分离函数
const startMSSTSeparation = async () => {
  if (!msstConfig.model || !msstConfig.inputFile || !msstConfig.outputDir) {
    return message.warning('请填写模型、输入文件和输出目录');
  }
  
  await startToolTask('msst-separate', {
    model: msstConfig.model,
    input_file: msstConfig.inputFile,
    output_dir: msstConfig.outputDir,
    stems: parseInt(msstConfig.stems),
    quality: msstConfig.quality
  }, `MSST分离: ${msstConfig.inputFile}`);
};

// 音频播放功能
const playAudio = (filePath: string) => {
  // 创建一个临时的音频元素来播放
  const audio = new Audio(filePath);
  audio.play().catch(err => {
    console.error('音频播放失败:', err);
    message.error('音频播放失败');
  });
};

// 文件下载功能
const downloadResult = (filePath: string) => {
  const link = document.createElement('a');
  link.href = filePath;
  link.download = filePath.split('/').pop() || 'download.wav';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
</script>

<style scoped>
.tools-container {
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}

.header-content {
  display: flex;
  align-items: center;
  gap: 20px;
}

.title-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-icon {
  opacity: 0.9;
  filter: drop-shadow(0 2px 4px rgba(99, 102, 241, 0.3));
}

.title-text {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.main-title {
  margin: 0 !important;
  color: #e2e8f0;
}

.subtitle {
  font-size: 16px;
}

.main-card {
  background: rgba(30, 41, 59, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.pane-content {
  padding: 24px 0;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.label-with-icon {
  color: #818cf8;
  font-size: 15px;
}

.label-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* MSST 分离样式 */
.msst-description {
  background: rgba(99, 102, 241, 0.05);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 20px;
  line-height: 1.6;
}

.msst-select {
  --n-border-radius: 8px !important;
}

.msst-button {
  height: 50px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
  transition: all 0.3s ease;
}

.msst-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
}

.results-section {
  background: rgba(15, 23, 42, 0.4);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 12px;
  padding: 20px;
  margin-top: 20px;
}

.result-item {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
  transition: all 0.3s ease;
}

.result-item:hover {
  background: rgba(255, 255, 255, 0.04);
  transform: translateY(-1px);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.result-path {
  font-size: 12px;
  font-family: 'Fira Code', monospace;
  color: #94a3b8;
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-header {
  padding: 8px 12px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 6px;
  margin-bottom: 8px;
}
</style>
