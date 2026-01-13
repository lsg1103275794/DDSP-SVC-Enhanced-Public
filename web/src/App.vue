<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <n-global-style />
    <n-message-provider>
      <n-layout has-sider position="absolute" class="root-layout">
        <!-- 苹果风格侧边栏 -->
        <n-layout-sider
          bordered
          collapse-mode="width"
          :collapsed-width="64"
          :width="240"
          class="apple-sider"
        >
          <div class="logo-container">
            <n-text strong class="logo-text">DDSP-SVC</n-text>
          </div>
          <n-menu
            v-model:value="activeKey"
            :options="menuOptions"
            class="apple-menu"
          />
        </n-layout-sider>

        <!-- 主内容区域 -->
        <n-layout-content content-style="padding: 32px;" class="main-content">
          <router-view v-slot="{ Component }">
            <transition name="fade-slide" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </n-layout-content>
      </n-layout>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { ref, h, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { darkTheme, NIcon } from 'naive-ui';
import type { MenuOption } from 'naive-ui';
import { 
  MusicalNotesOutline, 
  HardwareChipOutline, 
  MicOutline,
  BuildOutline,
  SettingsOutline
} from '@vicons/ionicons5';

const router = useRouter();
const route = useRoute();
const activeKey = ref(route.name as string);

function renderIcon(icon: any) {
  return () => h(NIcon, null, { default: () => h(icon) });
}

watch(() => route.name, (newName) => {
  activeKey.value = newName as string;
});

const menuOptions: MenuOption[] = [
  {
    label: '音频预处理',
    key: 'Preprocess',
    icon: renderIcon(MusicalNotesOutline),
    onClick: () => router.push('/preprocess')
  },
  {
    label: '模型训练',
    key: 'Train',
    icon: renderIcon(HardwareChipOutline),
    onClick: () => router.push('/train')
  },
  {
    label: '音频推理',
    key: 'Inference',
    icon: renderIcon(MicOutline),
    onClick: () => router.push('/inference')
  },
  {
    label: '工具箱',
    key: 'Tools',
    icon: renderIcon(BuildOutline),
    onClick: () => router.push('/tools')
  },
  {
    label: '系统设置',
    key: 'Settings',
    icon: renderIcon(SettingsOutline),
    disabled: true
  }
];

const themeOverrides = {
  common: {
    // 中国用户喜爱的靛蓝色系
    primaryColor: '#6366f1', // 靛蓝色 - 更加温暖现代
    primaryColorHover: '#818cf8', // 亮一点点的靛蓝
    primaryColorPressed: '#4f46e5', // 深一点的靛蓝
    
    // 毛玻璃卡片背景
    cardColor: 'rgba(23, 28, 41, 0.8)',
    bodyColor: '#0f111a', // 更深邃的背景
    
    // 现代化圆角
    borderRadius: '12px',
    borderRadiusLarge: '16px',
    
    // 成功色 - 更加明亮
    successColor: '#10b981',
    
    // 字体优化
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  },
  Menu: {
    itemColorActive: 'rgba(99, 102, 241, 0.15)',
    itemTextColorActive: '#818cf8',
    itemIconColorActive: '#818cf8',
    itemColorHover: 'rgba(255, 255, 255, 0.05)',
    itemTextColorHover: '#ffffff',
    itemHeight: '48px',
    borderRadius: '10px',
  },
  Layout: {
    siderColor: '#161924',
    headerColor: '#161924',
    footerColor: '#161924',
  },
  Card: {
    borderRadius: '16px',
    color: 'rgba(22, 25, 36, 0.7)',
    borderColor: 'rgba(255, 255, 255, 0.08)',
    titleTextColor: '#e2e8f0',
  },
  Button: {
    // 按钮高度
    height: '44px',
    heightMedium: '40px',
    heightSmall: '36px',
    heightTiny: '32px',
    
    // 圆角
    borderRadius: '12px',
    borderRadiusTiny: '8px',
    
    // 字体
    fontWeight: '500',
    
    // 过渡动画
    rippleDuration: '0.6s',
  },
  Input: {
    // 输入框高度
    height: '44px',
    borderRadius: '12px',
    
    // 字体
    fontSize: '14px',
    
    // 边框
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderHover: '1px solid rgba(99, 102, 241, 0.5)',
    borderFocus: '1px solid #6366f1',
    
    // 背景
    color: 'rgba(30, 41, 59, 0.6)',
    colorFocus: 'rgba(30, 41, 59, 0.8)',
    placeholderColor: 'rgba(255, 255, 255, 0.4)',
  },
  Progress: {
    // 进度条高度
    height: '6px',
    borderRadius: '3px',
    
    // 颜色
    fillColor: '#6366f1',
    fillColorFinished: '#10b981',
    fillColorProcessing: '#f59e0b',
  }
};
</script>

<style>
/* 主布局背景 */
.root-layout {
  background: transparent !important;
  position: relative;
}

/* 现代化侧边栏 */
.apple-sider {
  background: rgba(30, 41, 59, 0.85) !important;
  backdrop-filter: blur(20px) saturate(180%);
  border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
  position: relative;
  overflow: hidden;
}

.apple-sider::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, transparent 50%, rgba(236, 72, 153, 0.05) 100%);
  pointer-events: none;
}

/* Logo 容器优化 */
.logo-container {
  height: 72px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  position: relative;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.logo-container::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 24px;
  right: 24px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
}

/* Logo 文字现代化 */
.logo-text {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #ffffff 0%, #818cf8 50%, #ec4899 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  position: relative;
}

.logo-text::before {
  content: '';
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  background: linear-gradient(135deg, #6366f1, #ec4899, #f59e0b);
  border-radius: 8px;
  z-index: -1;
  opacity: 0.1;
  filter: blur(8px);
  animation: logoGlow 3s ease-in-out infinite alternate;
}

@keyframes logoGlow {
  0% { opacity: 0.1; transform: scale(1); }
  100% { opacity: 0.2; transform: scale(1.02); }
}

/* 主内容区域 */
.main-content {
  background: transparent;
  position: relative;
  overflow-x: hidden;
}

.main-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle at 25% 25%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 75% 75%, rgba(236, 72, 153, 0.06) 0%, transparent 50%),
    radial-gradient(circle at 50% 90%, rgba(245, 158, 11, 0.04) 0%, transparent 50%);
  pointer-events: none;
}

/* 现代化页面切换动画 */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateX(30px) scale(0.98);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateX(-30px) scale(0.98);
}

/* 现代化菜单样式 */
.apple-menu {
  padding: 16px 12px !important;
}

.apple-menu .n-menu-item {
  margin: 4px 0;
  border-radius: 12px !important;
  transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
}

.apple-menu .n-menu-item:hover {
  background: rgba(255, 255, 255, 0.08) !important;
  transform: translateX(4px);
}

.apple-menu .n-menu-item.n-menu-item--selected {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.15) 100%) !important;
  border: 1px solid rgba(99, 102, 241, 0.3) !important;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2) !important;
}

.apple-menu .n-menu-item--selected .n-menu-item-content {
  color: #818cf8 !important;
  font-weight: 600 !important;
}

/* 现代化滚动条 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.6), rgba(79, 70, 229, 0.6));
  border-radius: 4px;
  transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(79, 70, 229, 0.8));
}

/* 响应式优化 */
@media (max-width: 768px) {
  .apple-sider {
    width: 200px !important;
  }
  
  .logo-container {
    height: 64px;
    padding: 0 16px;
  }
  
  .logo-text {
    font-size: 18px;
  }
}

/* 加载状态动画 */
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.loading-shimmer {
  position: relative;
  overflow: hidden;
}

.loading-shimmer::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
  animation: shimmer 2s infinite;
}
</style>
