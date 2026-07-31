import { useEffect, useState } from 'react';
import './styles/app-theme.css';
import './components/ui/ui-primitives.css';
import appChromeStyles from './components/AppChrome.module.css';
import { BrowserRouter, Navigate, Route, Routes, useNavigate, useParams } from 'react-router';
import { AuthProvider, AuthStatusMenu, useAuth } from './auth/AuthContext';
import { RouteGuard } from './auth/RouteGuard';
import LoginPage from './pages/LoginPage';
import AdminAnnotationsPage from './pages/AdminAnnotationsPage';
import AnnotationPage from './pages/AnnotationPage';
import WorkspacePage from './pages/WorkspacePage';
import type { ThemeMode } from './components/ThemeToggle';
import type { AdminTab } from './pages/adminAnnotations/model';
import { RuntimeVersionProvider } from './components/RuntimeVersionProvider';

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
function AdminAnnotationsLanding() {
  const { authEnabled, hasPermission } = useAuth();
  if (!authEnabled) return <Navigate to="/admin/annotations/review" replace />;

  return (
    <RouteGuard>
      {hasPermission('annotation.queue.read') ? (
        <Navigate to="/admin/annotations/review" replace />
      ) : hasPermission('training.read') ? (
        <Navigate to="/admin/annotations/training" replace />
      ) : (
        <div role="alert" className="ui-alert ui-alert--danger m-6 p-4">
          Accès refusé : aucun onglet d’administration ne vous est attribué.
        </div>
      )}
    </RouteGuard>
  );
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

  if (!isAdminTab(tab)) {
    return <Navigate to="/admin/annotations/review" replace />;
  }

  return (
    <RouteGuard permission={tab === 'training' ? 'training.read' : 'annotation.queue.read'}>
      <AdminAnnotationsPage
        themeMode={themeMode}
        onToggleTheme={onToggleTheme}
        initialTab={tab}
        onNavigateTab={(nextTab) => navigate(`/admin/annotations/${nextTab}`)}
        authSlot={<AuthStatusMenu />}
      />
    </RouteGuard>
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
      <RuntimeVersionProvider>
        <AuthProvider>
          <div className={`${appChromeStyles.owner} app-shell flex h-screen w-screen flex-col overflow-hidden max-md:h-auto max-md:min-h-[100dvh] max-md:overflow-y-auto`} data-theme={themeMode}>
            <main className="flex-1 overflow-hidden max-md:flex-none max-md:overflow-visible">
              <Routes>
              <Route path="/login" element={<LoginPage themeMode={themeMode} onToggleTheme={toggleTheme} />} />
              <Route
                path="/"
                element={(
                  <RouteGuard>
                    <WorkspacePage themeMode={themeMode} onToggleTheme={toggleTheme} authSlot={<AuthStatusMenu />} />
                  </RouteGuard>
                )}
              />
              <Route path="/admin/annotation" element={<Navigate to="/admin/annotations" replace />} />
              <Route
                path="/admin/annotations"
                element={<AdminAnnotationsLanding />}
              />
              <Route
                path="/admin/annotations/:tab"
                element={(
                  <RouteGuard>
                    <AdminAnnotationsRoute themeMode={themeMode} onToggleTheme={toggleTheme} />
                  </RouteGuard>
                )}
              />
              <Route path="/dashboard" element={<Navigate to="/" replace />} />
              <Route path="/analysis/:id" element={<LegacyAnalysisRedirect />} />
              <Route
                path="/annotate/:id"
                element={(
                  <RouteGuard>
                    <AnnotationPage themeMode={themeMode} onToggleTheme={toggleTheme} authSlot={<AuthStatusMenu />} />
                  </RouteGuard>
                )}
              />
              </Routes>
            </main>
          </div>
        </AuthProvider>
      </RuntimeVersionProvider>
    </BrowserRouter>
  );
}

export default App;
