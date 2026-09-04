import { useEffect, useState } from 'react';
import { Activity, BarChart3, Beaker, Box, Code, Database, Image, Layers, Sliders, ToggleLeft, Zap } from 'lucide-react';
import { getClasses, getRuntimeVersion, getSimilar, adminAnnotationMediaUrl } from '../services/api';
import { isArchetypeGalleryEnabled } from './workspace/archetypeFlag';
import type { SimilarResult } from '../types';

interface ModelInfo {
  app_name: string;
  app_version: string;
  model_version: string | null;
  num_classes: number;
  class_names: string[];
  embedding_dim: number | null;
  sample_coverage: { total: number; covered: number; covered_list: string[] } | null;
}

function useModelInfo(): { data: ModelInfo | null; loading: boolean; error: string | null } {
  const [data, setData] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getRuntimeVersion(),
      getClasses(),
      fetch('/samples/coverage').then(r => r.ok ? r.json() : null).catch(() => null),
    ])
      .then(([version, classes, coverage]) => {
        if (cancelled) return;
        setData({
          app_name: version.app_name,
          app_version: version.app_version,
          model_version: version.model_version,
          num_classes: classes.num_classes,
          class_names: classes.class_names,
          embedding_dim: 128,
          sample_coverage: coverage ? {
            total: coverage.total_classes,
            covered: coverage.covered_classes,
            covered_list: coverage.covered,
          } : null,
        });
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load model info');
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  return { data, loading, error };
}

function StatCard({ icon: Icon, label, value, sub }: { icon: React.ComponentType<{ size?: number; className?: string }>; label: string; value: string | number; sub?: string }) {
  return (
    <div className="flex items-start gap-3 rounded border border-[color:var(--border-subtle)] p-4">
      <Icon size={20} className="mt-0.5 shrink-0 text-[color:var(--text-muted)]" />
      <div className="min-w-0">
        <div className="ui-text-meta">{label}</div>
        <div className="text-lg font-semibold">{value}</div>
        {sub && <div className="ui-text-meta mt-0.5 truncate">{sub}</div>}
      </div>
    </div>
  );
}

function FeatureFlagRow({ label, description, enabled, onToggle }: { label: string; description: string; enabled: boolean; onToggle: () => void }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded border border-[color:var(--border-subtle)] p-3">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        <div className="ui-text-meta text-xs">{description}</div>
      </div>
      <button
        onClick={onToggle}
        className={`flex h-6 w-10 shrink-0 items-center rounded-full px-0.5 transition-colors ${enabled ? 'bg-[color:var(--accent-primary)]' : 'bg-[color:var(--border-subtle)]'}`}
        role="switch"
        aria-checked={enabled}
      >
        <span className={`block h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${enabled ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
    </div>
  );
}

function EmbeddingExplorer() {
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [results, setResults] = useState<SimilarResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      setImageDataUrl(reader.result as string);
      setResults(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleExplore = async () => {
    if (!imageDataUrl) return;
    setLoading(true);
    setError(null);
    try {
      const result = await getSimilar(imageDataUrl, [0, 0, 100, 100], 10);
      setResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Similarity lookup failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Zap size={18} />
        <h3 className="text-base font-semibold">Embedding Explorer</h3>
      </div>
      <p className="ui-text-meta text-sm">
        Drop a cropped glyph image to find its nearest archetype neighbors via the embedding space.
      </p>

      <div className="flex flex-wrap items-start gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="ui-text-meta mb-1 block text-xs font-semibold">Glyph Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            className="ui-input w-full text-sm"
          />
        </div>
        <button
          onClick={handleExplore}
          disabled={!imageDataUrl || loading}
          className="ui-btn ui-btn--primary mt-5 px-4 py-2 text-sm"
        >
          {loading ? 'Exploring...' : 'Find Neighbors'}
        </button>
      </div>

      {imageDataUrl && (
        <div className="flex items-start gap-4">
          <div className="shrink-0">
            <div className="ui-text-meta mb-1 text-xs">Query Image</div>
            <img src={imageDataUrl} alt="Query" className="h-24 w-24 rounded border border-[color:var(--border-subtle)] object-contain" />
          </div>
          {results && (
            <div className="flex-1 min-w-0">
              <div className="ui-text-meta mb-2 text-xs">Nearest Neighbors ({results.results.length})</div>
              <div className="flex flex-wrap gap-2">
                {results.results.map((item) => (
                  <div key={item.rank} className="flex flex-col items-center gap-1 rounded border border-[color:var(--border-subtle)] p-2" style={{ width: 80 }}>
                    {item.asset ? (
                      <img src={adminAnnotationMediaUrl(item.asset)} alt={item.class_name} className="h-12 w-12 object-contain" />
                    ) : (
                      <div className="flex h-12 w-12 items-center justify-center bg-[color:var(--bg-subtle)] text-xs text-[color:var(--text-muted)]">—</div>
                    )}
                    <div className="text-xs font-medium truncate w-full text-center">{item.class_name}</div>
                    <div className={`text-[10px] ${item.band === 'high' ? 'text-green-600' : item.band === 'moderate' ? 'text-amber-600' : 'text-red-600'}`}>
                      {(item.similarity * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && <div className="ui-alert ui-alert--danger text-sm">{error}</div>}
    </div>
  );
}

export default function DevLabPage() {
  const { data, loading, error } = useModelInfo();
  const [archetypeEnabled, setArchetypeEnabled] = useState(isArchetypeGalleryEnabled());

  const toggleArchetype = () => {
    const next = !archetypeEnabled;
    try {
      window.localStorage.setItem('clinic.showArchetypes', next ? '1' : '0');
    } catch { /* noop */ }
    setArchetypeEnabled(next);
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl space-y-8 px-6 py-8">
        <div className="flex items-center gap-3">
          <Beaker size={24} className="text-[color:var(--accent-primary)]" />
          <div>
            <h1 className="text-xl font-bold">Dev Lab</h1>
            <p className="ui-text-meta text-sm">Model introspection, feature flags, and embedding exploration</p>
          </div>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-[color:var(--text-muted)]">
            <Activity size={16} className="animate-spin" />
            <span className="text-sm">Loading model info...</span>
          </div>
        )}

        {error && (
          <div className="ui-alert ui-alert--danger text-sm">{error}</div>
        )}

        {data && (
          <>
            <section className="space-y-3">
              <div className="flex items-center gap-2">
                <Database size={18} />
                <h2 className="text-base font-semibold">Model Info</h2>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard icon={Box} label="App" value={data.app_name} sub={`v${data.app_version}`} />
                <StatCard icon={Layers} label="Model" value={data.model_version ?? 'N/A'} sub="Active classifier" />
                <StatCard icon={BarChart3} label="Classes" value={data.num_classes} sub="Nahuatl glyph taxonomy" />
                <StatCard icon={Code} label="Embedding Dim" value={data.embedding_dim ?? 'N/A'} sub="DINOv2 + projection" />
              </div>
            </section>

            <section className="space-y-3">
              <div className="flex items-center gap-2">
                <Image size={18} />
                <h2 className="text-base font-semibold">Sample Coverage</h2>
              </div>
              {data.sample_coverage ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="ui-text-meta mb-1 text-xs">
                        {data.sample_coverage.covered} / {data.sample_coverage.total} classes have exemplar images
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-[color:var(--bg-subtle)]">
                        <div
                          className="h-full rounded-full bg-[color:var(--accent-primary)] transition-all"
                          style={{ width: `${(data.sample_coverage.covered / data.sample_coverage.total) * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-sm font-semibold tabular-nums">
                      {((data.sample_coverage.covered / data.sample_coverage.total) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <details className="text-xs">
                    <summary className="ui-text-meta cursor-pointer">View covered classes</summary>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {data.sample_coverage.covered_list.map((name) => (
                        <span key={name} className="rounded bg-[color:var(--bg-subtle)] px-1.5 py-0.5 font-mono text-[10px]">{name}</span>
                      ))}
                    </div>
                  </details>
                </div>
              ) : (
                <div className="ui-text-meta text-sm">No sample data indexed. Place exemplar images in the data/samples/ directory.</div>
              )}
            </section>

            <section className="space-y-3">
              <div className="flex items-center gap-2">
                <Sliders size={18} />
                <h2 className="text-base font-semibold">Feature Flags</h2>
              </div>
              <div className="space-y-2">
                <FeatureFlagRow
                  label="Archetype Gallery (F1-v0)"
                  description="Show exemplar thumbnails next to top-K predictions in the workspace trust panel."
                  enabled={archetypeEnabled}
                  onToggle={toggleArchetype}
                />
              </div>
            </section>

            <section className="space-y-3 rounded border border-[color:var(--border-subtle)] p-5">
              <EmbeddingExplorer />
            </section>

            <section className="space-y-3 rounded border border-dashed border-[color:var(--border-subtle)] p-5">
              <div className="flex items-center gap-2">
                <ToggleLeft size={18} />
                <h3 className="text-base font-semibold">Text Query (Coming Soon)</h3>
              </div>
              <p className="ui-text-meta text-sm">
                Search glyphs by Nahuatl word, semantic description, or visual attributes. The text encoder will map queries into the same embedding space for cross-modal retrieval.
              </p>
              <div className="flex gap-2">
                <input
                  type="text"
                  disabled
                  placeholder='e.g. "eau", "serpent a plumes", "temple"'
                  className="ui-input flex-1 text-sm opacity-50"
                />
                <button disabled className="ui-btn ui-btn--primary px-4 py-2 text-sm opacity-50">Search</button>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
