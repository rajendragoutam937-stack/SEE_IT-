const THEME_STORAGE_KEY = 'vision_theme';

export function initThemeToggle({ lightBtn, darkBtn, onThemeChange }) {
  function applyTheme(theme) {
    const isDark = theme === 'dark';
    document.documentElement.classList.toggle('dark', isDark);

    if (lightBtn) {
      lightBtn.classList.toggle('is-active', !isDark);
      lightBtn.setAttribute('aria-pressed', String(!isDark));
    }
    if (darkBtn) {
      darkBtn.classList.toggle('is-active', isDark);
      darkBtn.setAttribute('aria-pressed', String(isDark));
    }

    try {
      localStorage.setItem(THEME_STORAGE_KEY, isDark ? 'dark' : 'light');
    } catch {
      // Ignore storage failures
    }

    if (typeof onThemeChange === 'function') {
      onThemeChange(isDark ? 'dark' : 'light');
    }
  }

  let initialTheme = 'light';
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') {
      initialTheme = saved;
    }
  } catch {
    initialTheme = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
  }

  applyTheme(initialTheme);

  if (lightBtn) {
    lightBtn.addEventListener('click', () => applyTheme('light'));
  }
  if (darkBtn) {
    darkBtn.addEventListener('click', () => applyTheme('dark'));
  }
}
