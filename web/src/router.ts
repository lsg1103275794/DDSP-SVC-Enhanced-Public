import { createRouter, createWebHistory } from 'vue-router';
import PreprocessView from './views/PreprocessView.vue';
import TrainView from './views/TrainView.vue';
import InferenceView from './views/InferenceView.vue';
import ToolsView from './views/ToolsView.vue';

const routes = [
  {
    path: '/',
    redirect: '/preprocess',
  },
  {
    path: '/preprocess',
    name: 'Preprocess',
    component: PreprocessView,
  },
  {
    path: '/train',
    name: 'Train',
    component: TrainView,
  },
  {
    path: '/inference',
    name: 'Inference',
    component: InferenceView,
  },
  {
    path: '/tools',
    name: 'Tools',
    component: ToolsView,
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
