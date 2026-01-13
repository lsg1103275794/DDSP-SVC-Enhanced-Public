import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 10000,
});

export const systemApi = {
  getStatus: () => api.get('/status'),
  uploadFile: (formData: FormData) => api.post('/upload', formData),
};

export const preprocessApi = {
  separate: (data: any) => api.post('/preprocess/separate', data),
  slice: (data: any) => api.post('/preprocess/slice', data),
  getTaskStatus: (taskId: string) => api.get(`/preprocess/tasks/${taskId}`),
};

export const trainApi = {
  getConfig: () => api.get('/train/config'),
  start: (configPath: string) => api.post('/train/start', null, { params: { config_path: configPath } }),
  stop: () => api.post('/train/stop'),
  getStatus: () => api.get('/train/status'),
};

export const inferenceApi = {
  getModels: () => api.get('/inference/models'),
  convert: (data: any) => api.post('/inference/convert', data),
};

export default api;
