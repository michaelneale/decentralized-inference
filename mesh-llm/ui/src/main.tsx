import React from 'react';
import ReactDOM from 'react-dom/client';

import '@xyflow/react/dist/style.css';
import 'highlight.js/styles/github-dark.css';
import './index.css';
import { App } from './App';

type ThemeMode = 'auto' | 'light' | 'dark';

const THEME_STORAGE_KEY = 'mesh-llm-theme';
const media = window.matchMedia('(prefers-color-scheme: dark)');

const readThemeMode = (): ThemeMode => {
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === 'light' || stored === 'dark' || stored === 'auto' ? stored : 'auto';
};

const applyTheme = (mode: ThemeMode = readThemeMode()) => {
  const dark = mode === 'dark' || (mode === 'auto' && media.matches);
  document.documentElement.classList.toggle('dark', dark);
  document.documentElement.style.colorScheme = mode === 'auto' ? 'light dark' : dark ? 'dark' : 'light';
};

applyTheme();
media.addEventListener('change', () => {
  if (readThemeMode() === 'auto') applyTheme('auto');
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
