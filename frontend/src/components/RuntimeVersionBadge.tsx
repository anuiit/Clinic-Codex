import { useContext } from 'react';
import { RuntimeVersionContext } from './RuntimeVersionContext';

function prefixedVersion(version: string): string {
  if (version.toLowerCase().startsWith('v')) return version;
  return /^\d+\.\d+(?:\.\d+)?(?:[-+].*)?$/.test(version)
    ? `v${version}`
    : version;
}

export function RuntimeVersionBadge({ className = '' }: { className?: string }) {
  const runtime = useContext(RuntimeVersionContext);
  const appLabel = runtime.app_version
    ? `${runtime.app_name} ${prefixedVersion(runtime.app_version)}`
    : runtime.app_name;
  const modelLabel = runtime.model_version
    ? `Modèle ${prefixedVersion(runtime.model_version)}`
    : 'Modèle —';

  return (
    <div
      className={`flex min-w-0 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-[color:var(--border-subtle)] bg-[var(--surface-muted)] px-2 py-1 font-mono text-[0.66rem] leading-none text-[color:var(--text-soft)] ${className}`}
      aria-label={`${appLabel}, ${modelLabel}`}
      data-testid="runtime-version"
      title={`${appLabel} · ${modelLabel}`}
    >
      <span>{appLabel}</span>
      <span aria-hidden="true" className="text-[color:var(--divider)]">·</span>
      <span className="max-w-40 truncate">{modelLabel}</span>
    </div>
  );
}
