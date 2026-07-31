import axios from "axios";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { ShieldCheck } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router";
import {
  createFirstAdmin as createFirstAdminRequest,
  getAuthSession,
  login as loginRequest,
  logout as logoutRequest,
} from "../services/api";
import type { AuthSession, AuthUser, LoginPayload } from "../types";

type AuthStatus = "loading" | "disabled" | "unauthenticated" | "authenticated" | "error";
type AuthContextValue = {
  authEnabled: boolean;
  status: AuthStatus;
  user: AuthUser | null;
  createFirstAdmin: (payload: LoginPayload) => Promise<AuthSession>;
  login: (payload: LoginPayload) => Promise<AuthSession>;
  logout: () => Promise<void>;
  hasPermission: (permission: string) => boolean;
};
const AuthContext = createContext<AuthContextValue | null>(null);
function isUnauthorized(error: unknown): boolean {
  return (axios.isAxiosError(error) && error.response?.status === 401) || (error instanceof Response && error.status === 401);
}
export function AuthProvider({ children }: { children: ReactNode }) {
  const bypassAuthForSmoke = import.meta.env.VITE_AUTH_BYPASS === "1";
  const [authEnabled, setAuthEnabled] = useState(!bypassAuthForSmoke);
  const [status, setStatus] = useState<AuthStatus>(bypassAuthForSmoke ? "disabled" : "loading");
  const [user, setUser] = useState<AuthUser | null>(null);
  const applySession = useCallback((session: AuthSession) => {
    setAuthEnabled(session.auth_enabled);
    setUser(session.user);
    setStatus(!session.auth_enabled ? "disabled" : session.user ? "authenticated" : "unauthenticated");
    return session;
  }, []);
  useEffect(() => {
    if (bypassAuthForSmoke) return;
    let active = true;
    void getAuthSession().then((session) => { if (active) applySession(session); }).catch((error: unknown) => {
      if (!active) return;
      setUser(null);
      setAuthEnabled(true);
      setStatus(isUnauthorized(error) ? "unauthenticated" : "error");
    });
    return () => { active = false; };
  }, [applySession, bypassAuthForSmoke]);
  const login = useCallback(async (payload: LoginPayload) => applySession(await loginRequest(payload)), [applySession]);
  const createFirstAdmin = useCallback(
    async (payload: LoginPayload) => applySession(await createFirstAdminRequest(payload)),
    [applySession],
  );
  const logout = useCallback(async () => {
    await logoutRequest();
    setUser(null);
    setStatus(authEnabled ? "unauthenticated" : "disabled");
  }, [authEnabled]);
  const hasPermission = useCallback((permission: string) => user?.permissions.includes(permission) ?? false, [user]);
  const value = useMemo(
    () => ({ authEnabled, status, user, createFirstAdmin, login, logout, hasPermission }),
    [authEnabled, createFirstAdmin, hasPermission, login, logout, status, user],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
// AuthProvider and its hook intentionally share this context module.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used within AuthProvider");
  return value;
}
export function AuthStatusMenu() {
  const location = useLocation();
  const navigate = useNavigate();
  const [logoutError, setLogoutError] = useState<string | null>(null);
  const { authEnabled, status, user, logout, hasPermission } = useAuth();
  if (!authEnabled || status === "loading") return null;
  if (!user) return <Link to="/login" className="ui-action-ghost rounded-full px-3 py-1.5 text-sm">Se connecter</Link>;

  const adminPath = hasPermission("annotation.queue.read")
    ? "/admin/annotations/review"
    : hasPermission("training.read")
      ? "/admin/annotations/training"
      : null;
  const showAdminAccess = adminPath !== null && !location.pathname.startsWith("/admin/");

  return (
    <div className="flex flex-wrap items-center justify-end gap-2 text-sm">
      {showAdminAccess ? (
        <Link
          to={adminPath}
          aria-label="Administration"
          className="ui-action-ghost gap-1.5 rounded-full px-2 py-1.5 sm:px-3"
        >
          <ShieldCheck size={15} aria-hidden="true" />
          <span className="hidden sm:inline">Administration</span>
        </Link>
      ) : null}
      <span className="rounded-full border border-[color:var(--field-border)] px-3 py-1.5">
        {user.email} · {user.roles[0]}
      </span>
      <button
        type="button"
        className="ui-action-ghost rounded-full px-3 py-1.5"
        onClick={() => {
          setLogoutError(null);
          void logout()
            .then(() => navigate("/login", { replace: true }))
            .catch(() => setLogoutError("La déconnexion a échoué. Réessayez."));
        }}
      >
        Déconnexion
      </button>
      {logoutError ? (
        <span role="alert" className="text-xs text-[color:var(--danger)]">
          {logoutError}
        </span>
      ) : null}
    </div>
  );
}
