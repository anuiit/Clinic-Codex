import { useEffect, useState } from 'react';
import './styles/app-theme.css';
import './components/ui/ui-primitives.css';
import appChromeStyles from './components/AppChrome.module.css';
import { BrowserRouter, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom';
import AdminAnnotationsPage from './pages/AdminAnnotationsPage';
import AnnotationPage from './pages/AnnotationPage';
import WorkspacePage from './pages/WorkspacePage';
import type { ThemeMode } from './components/ThemeToggle';
import type { AdminTab } from './pages/adminAnnotations/model';

const THEME_STORAGE_KEY = 'clinic-codex-theme';

function getInitialTheme(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'dark';
  }

  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === 'dark' || stored === 'light') {
    return stored;
  }

  return 'dark';
}

function isAdminTab(value: string | undefined): value is AdminTab {
  return value === 'review' || value === 'dataset' || value === 'training';
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

function AdminAnnotationsRoute({
  themeMode,
  onToggleTheme,
}: {
  themeMode: ThemeMode;
  onToggleTheme: () => void;
}) {
  const navigate = useNavigate();
  const { tab } = useParams<{ tab?: string }>();

  useEffect(() => {
    if (tab && !isAdminTab(tab)) {
      navigate('/admin/annotations/review', { replace: true });
    }
  }, [navigate, tab]);

  return (
    <AdminAnnotationsPage
      themeMode={themeMode}
      onToggleTheme={onToggleTheme}
      initialTab={isAdminTab(tab) ? tab : 'review'}
      onNavigateTab={(nextTab) => navigate(`/admin/annotations/${nextTab}`)}
    />
  );
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
      <div className={`${appChromeStyles.owner} app-shell h-screen w-screen overflow-hidden flex flex-col`} data-theme={themeMode}>
        <main className="flex-1 overflow-hidden">
          <Routes>
            <Route path="/" element={<WorkspacePage themeMode={themeMode} onToggleTheme={toggleTheme} />} />
            <Route path="/admin/annotation" element={<Navigate to="/admin/annotations" replace />} />
            <Route
              path="/admin/annotations"
              element={<Navigate to="/admin/annotations/review" replace />}
            />
            <Route
              path="/admin/annotations/:tab"
              element={<AdminAnnotationsRoute themeMode={themeMode} onToggleTheme={toggleTheme} />}
            />
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
