<template>
  <div class="view-container">
    <n-h1 class="apple-title">模型训练</n-h1>
    
    <n-grid :x-gap="24" :y-gap="24" :cols="3">
      <!-- 训练配置 -->
      <n-gi :span="1">
        <n-card title="训练配置" class="apple-card">
          <n-form v-if="config" label-placement="top">
            <n-form-item label="模型类型">
              <n-input :value="config.model.type" disabled />
            </n-form-item>
            <n-form-item label="批次大小 (Batch Size)">
              <n-input-number v-model:value="config.train.batch_size" :min="1" />
            </n-form-item>
            <n-form-item label="保存间隔 (Steps)">
              <n-input-number v-model:value="config.train.save_interval" :min="100" />
            </n-form-item>
            <n-form-item label="学习率">
              <n-input-number v-model:value="config.train.lr" :step="0.0001" />
            </n-form-item>
            
            <n-space vertical class="mt-24">
              <n-button 
                type="primary" 
                block 
                size="large" 
                :loading="status.status === 'running'"
                @click="startTraining"
              >
                {{ status.status === 'running' ? '正在训练...' : '开始训练' }}
              </n-button>
              <n-button 
                v-if="status.status === 'running'" 
                type="error" 
                secondary 
                block 
                @click="stopTraining"
              >
                停止训练
              </n-button>
            </n-space>
          </n-form>
          <n-skeleton v-else text :repeat="10" />
        </n-card>
      </n-gi>

      <!-- 实时状态与日志 -->
      <n-gi :span="2">
        <n-card title="实时状态" class="apple-card">
          <template #header-extra>
            <n-tag :type="status.status === 'running' ? 'success' : 'info'" round>
              {{ status.status.toUpperCase() }}
            </n-tag>
          </template>
          
          <n-grid :cols="2" :x-gap="12">
            <n-gi>
              <n-statistic label="当前 Step" :value="currentStep" />
            </n-gi>
            <n-gi>
              <n-statistic label="PID" :value="status.pid || 'N/A'" />
            </n-gi>
          </n-grid>

          <n-divider />
          
          <div class="log-container">
            <div class="log-header">
              <n-text depth="3">训练日志 (train.log)</n-text>
              <n-button quaternary circle size="small" @click="fetchStatus">
                <n-icon><refresh-outline /></n-icon>
              </n-button>
            </div>
            <pre class="log-content">{{ logPreview }}</pre>
          </div>
          
          <div class="mt-24">
            <n-button secondary block @click="openTensorBoard">
              <template #icon>
                <n-icon><stats-chart-outline /></n-icon>
              </template>
              打开 TensorBoard
            </n-button>
          </div>
        </n-card>
      </n-gi>
    </n-grid>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { useMessage } from 'naive-ui';
import { RefreshOutline, StatsChartOutline } from '@vicons/ionicons5';

const message = useMessage();
const config = ref<any>(null);
const status = ref<any>({ status: 'idle' });
const logPreview = ref('等待训练开始...');
const currentStep = ref(0);
let statusTimer: any = null;

const fetchConfig = async () => {
  try {
    const res = await fetch('/api/v1/train/config');
    config.value = await res.json();
  } catch (e) {
    message.error('获取配置失败');
  }
};

const fetchStatus = async () => {
  try {
    const res = await fetch('/api/v1/train/status');
    status.value = await res.json();
    
    // 如果正在运行，可以尝试读取日志片段
    if (status.value.status === 'running') {
      const logRes = await fetch('/api/v1/train/logs?lines=50');
      const logData = await logRes.json();
      logPreview.value = logData.logs;
      
      // 解析 Step (简单示例: "Step: 1234")
      const stepMatch = logData.logs.match(/Step:\s*(\d+)/i);
      if (stepMatch) currentStep.value = parseInt(stepMatch[1]);
    }
  } catch (e) {
    console.error(e);
  }
};

const startTraining = async () => {
  try {
    const res = await fetch('/api/v1/train/start', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'success') {
        message.success('训练已启动');
        fetchStatus();
      } else {
        message.error(data.message);
      }
    } else {
      message.error('启动失败');
    }
  } catch (e) {
    message.error('启动失败');
  }
};

const stopTraining = async () => {
  try {
    await fetch('/api/v1/train/stop', { method: 'POST' });
    message.success('已发送停止信号');
    fetchStatus();
  } catch (e) {
    message.error('停止失败');
  }
};

const openTensorBoard = () => {
  window.open('http://localhost:6006', '_blank');
};

onMounted(() => {
  fetchConfig();
  fetchStatus();
  statusTimer = setInterval(fetchStatus, 3000);
});

onUnmounted(() => {
  if (statusTimer) clearInterval(statusTimer);
});
</script>

<style scoped>
.view-container {
  max-width: 1400px;
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
  backdrop-filter: blur(20px);
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

.apple-card:hover {
  transform: translateY(-6px);
  box-shadow: 
    0 16px 48px rgba(0, 0, 0, 0.35),
    0 0 0 1px rgba(99, 102, 241, 0.2) inset,
    0 0 24px rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.3);
}
.mt-24 {
  margin-top: 24px;
}

/* 现代化按钮优化 */
.view-container :deep(.n-button) {
  border-radius: 12px;
  font-weight: 500;
  letter-spacing: 0.025em;
  transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  position: relative;
  overflow: hidden;
}

.view-container :deep(.n-button)::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
  transition: left 0.6s ease;
}

.view-container :deep(.n-button):hover::before {
  left: 100%;
}

.view-container :deep(.n-button--primary) {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  border: 1px solid rgba(99, 102, 241, 0.3);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
}

.view-container :deep(.n-button--primary):hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
  border-color: rgba(99, 102, 241, 0.5);
}

.view-container :deep(.n-button--error) {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.view-container :deep(.n-button--secondary) {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
}

/* 统计卡片优化 */
.view-container :deep(.n-statistic) {
  padding: 20px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: all 0.3s ease;
}

.view-container :deep(.n-statistic):hover {
  background: rgba(255, 255, 255, 0.05);
  transform: translateY(-2px);
}

.view-container :deep(.n-statistic-label) {
  color: rgba(255, 255, 255, 0.7) !important;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 8px;
}

.view-container :deep(.n-statistic-value) {
  color: #ffffff !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  background: linear-gradient(135deg, #ffffff 0%, #818cf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* 标签样式优化 */
.view-container :deep(.n-tag) {
  border-radius: 20px;
  padding: 6px 16px;
  font-weight: 500;
  border: none;
}

.view-container :deep(.n-tag--success) {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
}

.view-container :deep(.n-tag--info) {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  color: white;
}

/* 输入框优化 */
.view-container :deep(.n-input),
.view-container :deep(.n-input-number) {
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.05);
  transition: all 0.3s ease;
}

.view-container :deep(.n-input:focus-within),
.view-container :deep(.n-input-number:focus-within) {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  background: rgba(255, 255, 255, 0.08);
}

.view-container :deep(.n-input-number-input),
.view-container :deep(.n-input__input) {
  color: #ffffff !important;
}

/* 分割线优化 */
.view-container :deep(.n-divider) {
  border-color: rgba(255, 255, 255, 0.1);
  margin: 24px 0;
}

/* 页面进入动画 */
.view-container {
  animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 卡片加载状态 */
.apple-card.loading {
  position: relative;
  overflow: hidden;
}

.apple-card.loading::after {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { left: -100%; }
  100% { left: 100%; }
}
.log-container {
  background: linear-gradient(145deg, rgba(0, 0, 0, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
  border-radius: 12px;
  padding: 20px;
  margin-top: 20px;
  border: 1px solid rgba(99, 102, 241, 0.2);
  backdrop-filter: blur(10px);
  box-shadow: 
    0 4px 16px rgba(0, 0, 0, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  position: relative;
  overflow: hidden;
}

.log-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, #6366f1, #818cf8, #ec4899);
  border-radius: 12px 12px 0 0;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.log-content {
  height: 320px;
  overflow-y: auto;
  font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: #10b981;
  margin: 0;
  white-space: pre-wrap;
  background: rgba(0, 0, 0, 0.4);
  padding: 16px;
  border-radius: 8px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  position: relative;
}

.log-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(180deg, transparent 0%, rgba(99, 102, 241, 0.05) 100%);
  pointer-events: none;
  border-radius: 8px;
}

/* 自定义滚动条样式 */
.log-content::-webkit-scrollbar {
  width: 6px;
}

.log-content::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 3px;
}

.log-content::-webkit-scrollbar-thumb {
  background: rgba(16, 185, 129, 0.6);
  border-radius: 3px;
}

.log-content::-webkit-scrollbar-thumb:hover {
  background: rgba(16, 185, 129, 0.8);
}
</style>
