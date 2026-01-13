<!-- Apple 风格模拟示例 (Naive UI 风格) -->
<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <n-layout has-sider position="absolute" style="background-color: #1c1c1e;">
      <!-- 磨砂玻璃侧边栏 -->
      <n-layout-sider
        bordered
        collapse-mode="width"
        :collapsed-width="64"
        :width="240"
        style="backdrop-filter: blur(20px); background-color: rgba(44, 44, 46, 0.8);"
      >
        <n-menu :options="menuOptions" />
      </n-layout-sider>

      <n-layout-content content-style="padding: 24px;">
        <n-space vertical size="large">
          <!-- 标题区域 -->
          <n-h1 style="color: #ffffff; font-weight: 600; margin-bottom: 8px;">音频预处理</n-h1>
          <n-text depth="3" style="color: rgba(235, 235, 245, 0.6);">
            使用 MSST 技术进行人声分离与干声提取
          </n-text>

          <!-- Apple 风格卡片容器 -->
          <n-grid :x-gap="16" :y-gap="16" :cols="2">
            <n-gi>
              <n-card
                title="上传音频"
                :bordered="false"
                style="border-radius: 16px; background-color: #2c2c2e; color: #ffffff;"
              >
                <n-upload multiple directory-dnd action="/api/upload">
                  <n-upload-dragger>
                    <n-text style="font-size: 16px; color: #ffffff;">点击或拖拽音频至此</n-text>
                    <br />
                    <n-text depth="3" style="font-size: 12px; color: rgba(235, 235, 245, 0.6);">
                      支持 wav, mp3, flac 格式
                    </n-text>
                  </n-upload-dragger>
                </n-upload>
              </n-card>
            </n-gi>
            
            <n-gi>
              <n-card
                title="分离设置"
                :bordered="false"
                style="border-radius: 16px; background-color: #2c2c2e; color: #ffffff;"
              >
                <n-form-item label="分离模型">
                  <n-select :options="modelOptions" placeholder="选择 MSST 模型" />
                </n-form-item>
                <n-form-item label="去混响强度">
                  <n-slider :default-value="50" :step="1" />
                </n-form-item>
                <n-button type="primary" block style="border-radius: 8px; background-color: #007aff;">
                  开始处理
                </n-button>
              </n-card>
            </n-gi>
          </n-grid>
        </n-space>
      </n-layout-content>
    </n-layout>
  </n-config-provider>
</template>

<script setup>
import { darkTheme } from 'naive-ui'

// 这里的配置仅为 Apple 风格色调演示
const themeOverrides = {
  common: {
    primaryColor: '#007AFF', // Apple Blue
    cardColor: '#2C2C2E',
    bodyColor: '#1C1C1E',
    textColor1: '#FFFFFF',
    textColor2: '#EBEBF5',
    borderRadius: '12px'
  }
}

const menuOptions = [
  { label: '音频预处理', key: 'prep' },
  { label: '模型训练', key: 'train' },
  { label: '音频推理', key: 'infer' }
]

const modelOptions = [
  { label: 'BS-Roformer (Vocal)', value: 'vocal' },
  { label: 'MDX-Net (DeReverb)', value: 'dereverb' }
]

// 模拟 API 调用逻辑
const handleUpload = ({ file }) => {
  console.log('正在上传至 /api/v1/upload:', file.name)
}

const startProcess = () => {
  console.log('提交任务至 /api/v1/preprocess/separate')
}
</script>

<style scoped>
/* 模拟 Apple 风格的边界阴影与层级 */
.n-card {
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
  transition: transform 0.2s ease;
}
.n-card:hover {
  transform: translateY(-2px);
}
</style>
