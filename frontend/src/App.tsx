import { useEffect, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom';
import AdminAnnotationsPage from './pages/AdminAnnotationsPage';
import AnnotationPage from './pages/AnnotationPage';
import WorkspacePage from './pages/WorkspacePage';
import type { ThemeMode } from './components/ThemeToggle';

const THEME_STORAGE_KEY = 'clinic-codex-theme';

function getInitialTheme(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'dark';
  }

  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === 'dark' || stored === 'light') {
    return stored;
  }

  if (window.matchMedia?.('(prefers-color-scheme: light)').matches) {
    return 'light';
  }

  return 'dark';
}

function LegacyAnalysisRedirect() {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();

  useEffect(() => {
    if (id) {
      navigate(`/?analysis=${encodeURIComponent(id)}`, { replace: true });
      return;
    }

    navigate('/', { replace: true });
  }, [id, navigate]);

  return null;
}

function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(getInitialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  const toggleTheme = () => {
    setThemeMode((current) => (current === 'dark' ? 'light' : 'dark'));
  };

  return (
    <BrowserRouter>
      <div className="app-shell h-screen w-screen overflow-hidden flex flex-col p-4" data-theme={themeMode}>
        <main className="flex-1 overflow-hidden rounded-xl">
          <Routes>
            <Route path="/" element={<WorkspacePage themeMode={themeMode} onToggleTheme={toggleTheme} />} />
            <Route path="/admin/annotations" element={<AdminAnnotationsPage themeMode={themeMode} onToggleTheme={toggleTheme} />} />
            <Route path="/dashboard" element={<Navigate to="/" replace />} />
            <Route path="/analysis/:id" element={<LegacyAnalysisRedirect />} />
            <Route path="/annotate/:id" element={<AnnotationPage themeMode={themeMode} onToggleTheme={toggleTheme} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
