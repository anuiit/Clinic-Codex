import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { AuthSurface } from '../auth/AuthSurface';
import { useAuth } from '../auth/AuthContext';
import type { ThemeMode } from '../components/ThemeToggle';
import { getBootstrapStatus } from '../services/api';

type LoginPageProps = {
  themeMode?: ThemeMode;
  onToggleTheme?: () => void;
};

const FALLBACK_REDIRECT = '/admin/annotations/review';

function sanitizeRedirect(value: string | null) {
  if (!value || !value.startsWith('/') || value.startsWith('//')) {
    return FALLBACK_REDIRECT;
  }
  return value;
}

function errorMessage(error: unknown) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: unknown } }).response;
    const data = response?.data;
    if (data && typeof data === 'object') {
      const message = (data as { message?: unknown }).message;
      const legacy = (data as { error?: unknown }).error;
      if (typeof message === 'string' && message.trim()) {
        return message;
      }
      if (typeof legacy === 'string' && legacy.trim()) {
        return legacy;
      }
    }
  }
  return 'Unable to sign in';
}

function errorCode(error: unknown) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: unknown } }).response;
    const data = response?.data;
    if (data && typeof data === 'object') {
      const code = (data as { error_code?: unknown }).error_code;
      return typeof code === 'string' ? code : null;
    }
  }
  return null;
}

export default function LoginPage({
  themeMode = 'dark',
  onToggleTheme = () => undefined,
}: LoginPageProps = {}) {
  const auth = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTarget = useMemo(() => {
    const redirect = new URLSearchParams(location.search).get('redirect');
    return sanitizeRedirect(redirect);
  }, [location.search]);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirmation, setPasswordConfirmation] = useState('');
  const [bootstrapAvailable, setBootstrapAvailable] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);

  useEffect(() => {
    let active = true;
    void getBootstrapStatus()
      .then((status) => {
        if (active) setBootstrapAvailable(status.auth_enabled && status.bootstrap_available);
      })
      .catch(() => {
        // Login remains available if the optional local setup probe fails.
      });
    return () => {
      active = false;
    };
  }, []);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    if (bootstrapAvailable && password !== passwordConfirmation) {
      setError('Passwords do not match.');
      setSubmitting(false);
      return;
    }
    try {
      if (bootstrapAvailable) {
        await auth.createFirstAdmin({ email, password });
      } else {
        await auth.login({ email, password });
      }
      navigate(redirectTarget, { replace: true });
    } catch (caught) {
      if (errorCode(caught) === 'BOOTSTRAP_UNAVAILABLE') {
        setBootstrapAvailable(false);
      }
      setError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthSurface
      eyebrow={bootstrapAvailable ? 'Configuration locale sécurisée' : 'Secure access'}
      title={bootstrapAvailable ? 'Créer le premier compte administrateur' : 'Sign in'}
      subtitle={
        bootstrapAvailable
          ? 'Aucun compte n’existe encore. Créez l’administrateur de cette installation locale.'
          : 'Use your account to access the admin area.'
      }
      themeMode={themeMode}
      onToggleTheme={onToggleTheme}
    >
      <form
        aria-label={bootstrapAvailable ? 'Create administrator account' : 'Sign in'}
        aria-describedby={error ? 'login-error' : undefined}
        className="flex flex-col gap-5"
        onSubmit={handleSubmit}
      >
        {error ? (
          <div id="login-error" ref={errorRef} tabIndex={-1} role="alert" className="ui-alert ui-alert--danger p-3 text-sm">
            {error}
          </div>
        ) : null}

        <label className="flex flex-col gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.18em] text-[color:var(--text-soft)]">
            Email
          </span>
          <input
            type="email"
            required
            autoComplete="email"
            className="ui-input h-11 px-3"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="prenom.nom@example.com"
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.18em] text-[color:var(--text-soft)]">
            Password
          </span>
          <input
            aria-label="Password"
            type="password"
            required
            minLength={bootstrapAvailable ? 12 : undefined}
            maxLength={bootstrapAvailable ? 256 : undefined}
            autoComplete={bootstrapAvailable ? 'new-password' : 'current-password'}
            className="ui-input h-11 px-3"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={bootstrapAvailable ? '12 characters minimum' : 'Your password'}
          />
          {bootstrapAvailable ? (
            <span className="text-xs text-[color:var(--text-muted)]">12 caractères minimum.</span>
          ) : null}
        </label>

        {bootstrapAvailable ? (
          <label className="flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-[0.18em] text-[color:var(--text-soft)]">
              Confirm password
            </span>
            <input
              aria-label="Confirm password"
              type="password"
              required
              minLength={12}
              maxLength={256}
              autoComplete="new-password"
              className="ui-input h-11 px-3"
              value={passwordConfirmation}
              onChange={(event) => setPasswordConfirmation(event.target.value)}
              placeholder="Repeat your password"
            />
          </label>
        ) : null}

        <button
          type="submit"
          className="ui-action-primary w-full justify-center rounded-2xl px-4 py-3 text-sm"
          disabled={submitting}
        >
          {submitting
            ? bootstrapAvailable
              ? 'Creating account…'
              : 'Signing in…'
            : bootstrapAvailable
              ? 'Create administrator account'
              : 'Sign in'}
        </button>
      </form>
    </AuthSurface>
  );
}
