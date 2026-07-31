import type { ReactNode } from "react";
import { Navigate, useLocation } from "react-router";
import { useAuth } from "./AuthContext";

type RouteGuardProps = { children: ReactNode; permission?: string };

export function RouteGuard({ children, permission }: RouteGuardProps) {
  const location = useLocation();
  const { authEnabled, hasPermission, status, user } = useAuth();
  if (status === "loading") return <div role="status" aria-live="polite" className="p-6 text-sm text-[color:var(--text-muted)]">Vérification de la session…</div>;
  if (!authEnabled) return <>{children}</>;
  if (status === "error") return <div role="alert" className="ui-alert ui-alert--danger m-6 p-4">Impossible de vérifier votre session. Réessayez plus tard.</div>;
  if (!user) return <Navigate to={`/login?redirect=${encodeURIComponent(`${location.pathname}${location.search}`)}`} replace />;
  if (permission && !hasPermission(permission)) return <div role="alert" className="ui-alert ui-alert--danger m-6 p-4">Accès refusé : votre rôle ne permet pas d’ouvrir cette page.</div>;
  return <>{children}</>;
}
