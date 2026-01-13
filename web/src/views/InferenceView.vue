<template>
  <div class="view-container">
    <n-h1 class="apple-title">音频推理 (SVC)</n-h1>
    
    <n-grid :x-gap="24" :y-gap="24" :cols="3">
      <!-- 模型与参数 -->
      <n-gi :span="1">
        <n-card title="模型与参数" class="apple-card">
          <n-form label-placement="top">
            <n-form-item label="选择模型">
              <n-select 
                v-model:value="selectedModel" 
                :options="modelOptions" 
                placeholder="选择训练好的模型"
              />
            </n-form-item>
            
            <n-form-item v-if="selectedModel" label="选择权重 (Checkpoint)">
              <n-select 
                v-model:value="selectedCkpt" 
                :options="ckptOptions" 
                placeholder="选择权重文件"
              />
            </n-form-item>

            <n-form-item label="音调偏移 (Semitones)">
              <n-slider v-model:value="params.pitch_shift" :min="-12" :max="12" :step="1" />
              <template #feedback>
                {{ params.pitch_shift > 0 ? '+' : '' }}{{ params.pitch_shift }} 半音
              </template>
            </n-form-item>

            <n-form-item label="F0 提取器">
              <n-select v-model:value="params.f0_extractor" :options="f0Extractors" />
            </n-form-item>

            <n-form-item label="说话人 ID">
              <n-input-number v-model:value="params.spk_id" :min="1" />
            </n-form-item>
          </n-form>
        </n-card>

        <!-- AudioNoise 增强选项 -->
        <n-card title="音频增强 (AudioNoise)" class="apple-card mt-24">
          <n-form label-placement="top">
            <n-divider title-placement="left">F0 平滑</n-divider>
            <n-grid :cols="2" :x-gap="12">
              <n-gi>
                <n-form-item label="启用 F0 平滑">
                  <n-switch v-model:value="params.f0_smooth" />
                </n-form-item>
              </n-gi>
              <n-gi>
                <n-form-item label="八度错误修正">
                  <n-switch v-model:value="params.octave_fix" />
                </n-form-item>
              </n-gi>
            </n-grid>
            <n-form-item v-if="params.f0_smooth" label="平滑截止频率 (Hz)">
              <n-slider v-model:value="params.f0_smooth_cutoff" :min="5" :max="50" :step="1" />
              <template #feedback>{{ params.f0_smooth_cutoff }} Hz</template>
            </n-form-item>

            <n-divider title-placement="left">LFO 调制</n-divider>
            <n-grid :cols="2" :x-gap="12">
              <n-gi>
                <n-form-item label="颤音 (Vibrato)">
                  <n-switch v-model:value="params.vibrato" />
                </n-form-item>
              </n-gi>
              <n-gi>
                <n-form-item label="震音 (Tremolo)">
                  <n-switch v-model:value="params.tremolo" />
                </n-form-item>
              </n-gi>
            </n-grid>
            <div v-if="params.vibrato">
              <n-form-item label="颤音频率 (Hz)">
                <n-slider v-model:value="params.vibrato_rate" :min="3" :max="8" :step="0.5" />
                <template #feedback>{{ params.vibrato_rate }} Hz</template>
              </n-form-item>
              <n-form-item label="颤音深度">
                <n-slider v-model:value="params.vibrato_depth" :min="0.01" :max="0.05" :step="0.005" />
                <template #feedback>{{ (params.vibrato_depth * 100).toFixed(1) }}%</template>
              </n-form-item>
            </div>
            <div v-if="params.tremolo">
              <n-form-item label="震音频率 (Hz)">
                <n-slider v-model:value="params.tremolo_rate" :min="2" :max="8" :step="0.5" />
                <template #feedback>{{ params.tremolo_rate }} Hz</template>
              </n-form-item>
              <n-form-item label="震音深度">
                <n-slider v-model:value="params.tremolo_depth" :min="0.05" :max="0.3" :step="0.01" />
                <template #feedback>{{ (params.tremolo_depth * 100).toFixed(0) }}%</template>
              </n-form-item>
            </div>

            <n-divider title-placement="left">音频效果</n-divider>
            <n-form-item label="效果预设">
              <n-select v-model:value="params.effects_preset" :options="effectsPresets" />
            </n-form-item>
            <n-grid :cols="2" :x-gap="12">
              <n-gi>
                <n-form-item label="合唱效果">
                  <n-switch v-model:value="params.chorus" />
                </n-form-item>
              </n-gi>
              <n-gi>
                <n-form-item label="混响效果">
                  <n-switch v-model:value="params.reverb" />
                </n-form-item>
              </n-gi>
            </n-grid>
            <n-form-item v-if="params.reverb" label="混响混合比">
              <n-slider v-model:value="params.reverb_mix" :min="0" :max="0.5" :step="0.05" />
              <template #feedback>{{ (params.reverb_mix * 100).toFixed(0) }}%</template>
            </n-form-item>
          </n-form>
        </n-card>
      </n-gi>

      <!-- 输入与输出 -->
      <n-gi :span="2">
        <n-card title="推理控制" class="apple-card">
          <n-grid :cols="2" :x-gap="24">
            <n-gi>
              <n-h3>1. 选择输入音频</n-h3>
              <n-select 
                v-model:value="selectedInput" 
                :options="inputOptions" 
                placeholder="选择上传或分离后的音频"
              />
              <div v-if="selectedInput" class="mt-12">
                <audio controls :src="getInputUrl(selectedInput)" class="apple-audio"></audio>
              </div>
            </n-gi>
            <n-gi>
              <n-h3>2. 执行转换</n-h3>
              <n-button 
                type="primary" 
                block 
                size="large" 
                :loading="loading"
                :disabled="!canInfer"
                @click="startInference"
              >
                开始转换
              </n-button>
            </n-gi>
          </n-grid>

          <n-divider />

          <!-- 结果展示 -->
          <div v-if="activeTask" class="mt-24">
            <n-text strong>{{ activeTask.message }}</n-text>
            <n-progress
              type="line"
              :percentage="activeTask.progress"
              :status="activeTask.status === 'failed' ? 'error' : 'primary'"
              processing
              class="mt-12"
            />
          </div>

          <div v-if="resultUrl" class="mt-24 result-box">
            <n-h3>转换结果</n-h3>
            <audio controls :src="resultUrl" class="apple-audio"></audio>
            <div class="mt-12">
              <n-button secondary @click="downloadResult">
                下载结果
              </n-button>
            </div>
          </div>
        </n-card>
      </n-gi>
    </n-grid>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, reactive } from 'vue';
import { useMessage } from 'naive-ui';

const message = useMessage();
const loading = ref(false);
const models = ref<any[]>([]);
const inputs = ref<any[]>([]);
const selectedModel = ref<string | null>(null);
const selectedCkpt = ref<string | null>(null);
const selectedInput = ref<string | null>(null);
const activeTask = ref<any>(null);
const resultUrl = ref<string | null>(null);

const params = reactive({
  pitch_shift: 0,
  f0_extractor: 'rmvpe',
  spk_id: 1,
  // AudioNoise 增强参数
  f0_smooth: false,
  f0_smooth_cutoff: 20.0,
  median_kernel: 3,
  octave_fix: false,
  vibrato: false,
  vibrato_rate: 5.5,
  vibrato_depth: 0.02,
  vibrato_delay: 0.2,
  tremolo: false,
  tremolo_rate: 4.0,
  tremolo_depth: 0.1,
  effects_preset: 'none',
  chorus: false,
  reverb: false,
  reverb_mix: 0.2
});

const f0Extractors = [
  { label: 'RMVPE (推荐)', value: 'rmvpe' },
  { label: 'FCPE', value: 'fcpe' },
  { label: 'Crepe', value: 'crepe' }
];

const effectsPresets = [
  { label: '无效果', value: 'none' },
  { label: '自然增强', value: 'natural' },
  { label: '空间感', value: 'spacious' },
  { label: '复古', value: 'vintage' },
  { label: '干净', value: 'clean' }
];

const modelOptions = computed(() => 
  models.value.map(m => ({ label: m.name, value: m.name }))
);

const ckptOptions = computed(() => {
  const m = models.value.find(m => m.name === selectedModel.value);
  return m ? m.checkpoints.map((c: string) => ({ label: c, value: c })) : [];
});

const inputOptions = computed(() => 
  inputs.value.map(i => ({ label: i.filename, value: i.id }))
);

const canInfer = computed(() => 
  selectedCkpt.value && selectedInput.value
);

const fetchData = async () => {
  try {
    const [modelRes, inputRes] = await Promise.all([
      fetch('/api/v1/inference/models'),
      fetch('/api/v1/preprocess/files?type=upload') // 这里也可以包含 separated
    ]);
    const modelData = await modelRes.json();
    models.value = modelData.models;
    inputs.value = await inputRes.json();
    
    // 尝试获取分离后的音频作为输入
    const sepRes = await fetch('/api/v1/preprocess/files?type=separated');
    const sepData = await sepRes.json();
    inputs.value = [...inputs.value, ...sepData];
  } catch (e) {
    message.error('获取数据失败');
  }
};

const getInputUrl = (id: string) => {
  const item = inputs.value.find(i => i.id === id);
  if (!item) return '';
  
  if (item.type === 'upload') {
    return `/api/v1/system/static/uploads/${item.filename}`;
  } else if (item.type === 'separated') {
    const [dirId, filename] = item.id.includes(':') ? item.id.split(':') : ['', item.filename];
    return `/api/v1/system/static/processed/separated/${dirId}/${filename}`;
  } else if (item.type === 'sliced') {
    const [dirId, filename] = item.id.includes(':') ? item.id.split(':') : ['', item.filename];
    return `/api/v1/system/static/processed/sliced/${dirId}/${filename}`;
  }
  return '';
};

const startInference = async () => {
  loading.value = true;
  resultUrl.value = null;
  try {
    const res = await fetch('/api/v1/inference/convert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input_id: selectedInput.value,
        model_path: `${selectedModel.value}/${selectedCkpt.value}`,
        pitch_shift: params.pitch_shift,
        f0_extractor: params.f0_extractor,
        spk_id: params.spk_id,
        // AudioNoise 增强参数
        f0_smooth: params.f0_smooth,
        f0_smooth_cutoff: params.f0_smooth_cutoff,
        median_kernel: params.median_kernel,
        octave_fix: params.octave_fix,
        vibrato: params.vibrato,
        vibrato_rate: params.vibrato_rate,
        vibrato_depth: params.vibrato_depth,
        vibrato_delay: params.vibrato_delay,
        tremolo: params.tremolo,
        tremolo_rate: params.tremolo_rate,
        tremolo_depth: params.tremolo_depth,
        effects_preset: params.effects_preset,
        chorus: params.chorus,
        reverb: params.reverb,
        reverb_mix: params.reverb_mix
      })
    });
    const data = await res.json();
    pollTask(data.task_id);
  } catch (e) {
    message.error('提交任务失败');
    loading.value = false;
  }
};

const downloadResult = () => {
  if (!resultUrl.value) return;
  const link = document.createElement('a');
  link.href = resultUrl.value;
  link.download = resultUrl.value.split('/').pop() || 'result.wav';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

const pollTask = async (taskId: string) => {
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/v1/preprocess/tasks/${taskId}`);
      const data = await res.json();
      activeTask.value = data;
      
      if (data.status === 'completed') {
        clearInterval(timer);
        loading.value = false;
        // 解析文件名，后端 message 格式: "转换成功: filename.wav"
        const filename = data.message.split(': ')[1];
        resultUrl.value = `/api/v1/system/static/output/${filename}`;
        message.success('音频转换完成');
      } else if (data.status === 'failed') {
        clearInterval(timer);
        loading.value = false;
        message.error(data.message);
      }
    } catch (e) {
      clearInterval(timer);
      loading.value = false;
    }
  }, 1000);
};

onMounted(fetchData);
</script>

<style scoped>
.view-container {
  max-width: 1200px;
  margin: 0 auto;
}
.apple-title {
  font-size: 34px;
  font-weight: 700;
  letter-spacing: -1px;
  margin-bottom: 32px;
  color: #fff;
}
.apple-card {
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px) saturate(180%);
  box-shadow: 
    0 8px 32px rgba(0, 0, 0, 0.25),
    0 0 0 1px rgba(255, 255, 255, 0.05) inset;
  transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  position: relative;
  overflow: hidden;
}

.apple-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
}

.apple-card::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.02) 0%, transparent 50%, rgba(236, 72, 153, 0.02) 100%);
  pointer-events: none;
}

.apple-card:hover {
  transform: translateY(-6px);
  box-shadow: 
    0 16px 48px rgba(0, 0, 0, 0.35),
    0 0 0 1px rgba(99, 102, 241, 0.2) inset,
    0 0 24px rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.3);
}
.apple-audio {
  width: 100%;
  height: 40px;
  border-radius: 8px;
}
.mt-12 { margin-top: 12px; }
.mt-24 { margin-top: 24px; }
.result-box {
  padding: 28px;
  background: linear-gradient(145deg, rgba(99, 102, 241, 0.15) 0%, rgba(79, 70, 229, 0.08) 100%);
  border-radius: 16px;
  border: 1px solid rgba(99, 102, 241, 0.3);
  box-shadow: 
    0 8px 32px rgba(99, 102, 241, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  position: relative;
  overflow: hidden;
  animation: slideInUp 0.6s ease-out;
}

.result-box::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, #6366f1, #818cf8, #ec4899);
  border-radius: 16px 16px 0 0;
}

.result-box::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.05) 0%, transparent 50%);
  pointer-events: none;
}

/* 现代化音频播放器 */
.apple-audio {
  width: 100%;
  height: 48px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
}

.apple-audio:hover {
  border-color: rgba(99, 102, 241, 0.3);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.1);
}

/* 现代化表单元素 */
.view-container :deep(.n-form-item-label) {
  font-weight: 500 !important;
  color: rgba(255, 255, 255, 0.9) !important;
  font-size: 14px;
  margin-bottom: 8px;
}

.view-container :deep(.n-input),
.view-container :deep(.n-select),
.view-container :deep(.n-input-number) {
  border-radius: 12px !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  background: rgba(255, 255, 255, 0.05) !important;
  transition: all 0.3s ease !important;
}

.view-container :deep(.n-input:focus-within),
.view-container :deep(.n-select:focus-within),
.view-container :deep(.n-input-number:focus-within) {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
  background: rgba(255, 255, 255, 0.08) !important;
}

.view-container :deep(.n-input-number-input),
.view-container :deep(.n-input__input),
.view-container :deep(.n-select-input) {
  color: #ffffff !important;
}

/* 现代化滑块 */
.view-container :deep(.n-slider-rail) {
  background: rgba(255, 255, 255, 0.1) !important;
  border-radius: 6px !important;
  height: 6px !important;
}

.view-container :deep(.n-slider-fill) {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
  border-radius: 6px !important;
}

.view-container :deep(.n-slider-button) {
  background: #6366f1 !important;
  border: 2px solid #ffffff !important;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4) !important;
  width: 20px !important;
  height: 20px !important;
  transition: all 0.3s ease !important;
}

.view-container :deep(.n-slider-button:hover) {
  transform: scale(1.1);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.6) !important;
}

/* 现代化按钮 */
.view-container :deep(.n-button) {
  border-radius: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.025em !important;
  transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
  position: relative !important;
  overflow: hidden !important;
  height: 48px !important;
  font-size: 15px !important;
}

.view-container :deep(.n-button)::before {
  content: '' !important;
  position: absolute !important;
  top: 0 !important;
  left: -100% !important;
  width: 100% !important;
  height: 100% !important;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent) !important;
  transition: left 0.6s ease !important;
}

.view-container :deep(.n-button):hover::before {
  left: 100% !important;
}

.view-container :deep(.n-button--primary) {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
  border: 1px solid rgba(99, 102, 241, 0.3) !important;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
}

.view-container :deep(.n-button--primary):hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4) !important;
  border-color: rgba(99, 102, 241, 0.5) !important;
}

.view-container :deep(.n-button--primary):disabled {
  background: rgba(255, 255, 255, 0.1) !important;
  border-color: rgba(255, 255, 255, 0.1) !important;
  color: rgba(255, 255, 255, 0.5) !important;
  box-shadow: none !important;
  transform: none !important;
}

.view-container :deep(.n-button--secondary) {
  background: rgba(255, 255, 255, 0.08) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  backdrop-filter: blur(10px) !important;
  color: rgba(255, 255, 255, 0.9) !important;
}

.view-container :deep(.n-button--secondary):hover {
  background: rgba(255, 255, 255, 0.12) !important;
  border-color: rgba(255, 255, 255, 0.25) !important;
  color: #ffffff !important;
}

/* 现代化标题 */
.view-container :deep(.n-h3) {
  font-size: 18px !important;
  font-weight: 600 !important;
  color: rgba(255, 255, 255, 0.95) !important;
  margin-bottom: 16px !important;
}

/* 现代化分割线 */
.view-container :deep(.n-divider) {
  border-color: rgba(255, 255, 255, 0.1) !important;
  margin: 24px 0 !important;
}

/* 现代化选项卡片 */
.view-container :deep(.n-select-menu) {
  backdrop-filter: blur(20px) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
  background: rgba(30, 41, 59, 0.95) !important;
}

.view-container :deep(.n-select-option) {
  border-radius: 8px !important;
  margin: 2px 4px !important;
  transition: all 0.3s ease !important;
  padding: 12px 16px !important;
}

.view-container :deep(.n-select-option--selected) {
  background: rgba(99, 102, 241, 0.2) !important;
  color: #818cf8 !important;
}

.view-container :deep(.n-select-option:hover) {
  background: rgba(255, 255, 255, 0.08) !important;
}

/* 进度条优化 */
.view-container :deep(.n-progress) {
  border-radius: 8px !important;
}

.view-container :deep(.n-progress-content) {
  border-radius: 8px !important;
}

.view-container :deep(.n-progress-indicator--line) {
  border-radius: 8px !important;
}

/* 动画效果 */
.view-container {
  animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(40px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 卡片进入动画延迟 */
.apple-card:nth-child(1) { animation-delay: 0.1s; }
.apple-card:nth-child(2) { animation-delay: 0.2s; }

/* 响应式优化 */
@media (max-width: 768px) {
  .view-container {
    padding: 0 12px;
  }
  
  .result-box {
    padding: 20px;
  }
  
  .view-container :deep(.n-button) {
    height: 44px !important;
    font-size: 14px !important;
  }
  
  .apple-audio {
    height: 40px;
  }
}
</style>
