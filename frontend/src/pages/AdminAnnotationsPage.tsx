import { useCallback, useEffect, useMemo, useState } from 'react';
import ThemeToggle, { type ThemeMode } from '../components/ThemeToggle';
import {
  adminAnnotationMediaUrl,
  getAdminAnnotationQueue,
  getAdminTrainingSummary,
  getLatestAdminTrainingJob,
  modifyAdminAnnotationElement,
  setAdminAnnotationReviewStatus,
  startAdminTrainingJob,
} from '../services/api';
import type {
  AdminAnnotationAnalysis,
  AdminAnnotationElement,
  AdminAnnotationModifyPayload,
  AdminAnnotationQueue,
  AdminAnnotationReviewStatus,
  AdminTrainingJob,
  AdminTrainingSummary,
} from '../types';

type AdminTab = 'review' | 'dataset' | 'training';

type AdminAnnotationsPageProps = {
  themeMode?: ThemeMode;
  onToggleTheme?: () => void;
};

const ADMIN_TABS: Array<{ id: AdminTab; label: string; description: string }> = [
  { id: 'review', label: 'Review', description: 'Approve, reject, and inspect submitted elements.' },
  { id: 'dataset', label: 'Dataset', description: 'Visualize approved and rejected crops before training.' },
  { id: 'training', label: 'Training', description: 'Inspect approved-only data and guarded local training status.' },
];

const STATUS_LABEL: Record<AdminAnnotationReviewStatus, string> = {
  pending: 'Pending',
  approved: 'Approved',
  rejected: 'Rejected',
};

const STATUS_CLASS: Record<AdminAnnotationReviewStatus, string> = {
  pending: 'ui-chip--accent',
  approved: 'admin-chip--success',
  rejected: 'ui-chip--danger',
};

function formatBbox(bbox: number[]) {
  return bbox.join(', ');
}

function StatusBadge({ status }: { status: AdminAnnotationReviewStatus }) {
  return (
    <span
      aria-label={`Review status ${status}`}
      className={`ui-chip ${STATUS_CLASS[status]}`}
    >
      {STATUS_LABEL[status]}
    </span>
  );
}

function CountCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="ui-section p-3">
      <div className="ui-text-eyebrow text-[0.65rem]">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-[color:var(--text-main)]">{value}</div>
    </div>
  );
}

function AdminHeader({ themeMode = 'dark', onToggleTheme }: AdminAnnotationsPageProps) {
  return (
    <header className="app-header rounded-2xl p-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <p className="ui-text-eyebrow">Local admin</p>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-[color:var(--text-heading)]">
            Annotation admin console
          </h1>
          <p className="mt-2 max-w-3xl ui-text-body-sm">
            Review submitted annotation elements, inspect the approved-only dataset, and prepare local
            classifier training without leaving the design system.
          </p>
        </div>
        {onToggleTheme ? (
          <ThemeToggle mode={themeMode} onToggle={onToggleTheme} className="shrink-0" />
        ) : null}
      </div>
    </header>
  );
}

function AdminTabs({ activeTab, onSelect }: { activeTab: AdminTab; onSelect: (tab: AdminTab) => void }) {
  return (
    <div className="ui-panel p-2">
      <div role="tablist" aria-label="Admin annotation sections" className="grid gap-2 md:grid-cols-3">
        {ADMIN_TABS.map((tab) => {
          const selected = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              id={`admin-${tab.id}-tab`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`admin-${tab.id}-panel`}
              aria-label={tab.label}
              className={`admin-tab ${selected ? 'admin-tab--active' : ''}`}
              onClick={() => onSelect(tab.id)}
            >
              <span className="admin-tab__label">{tab.label}</span>
              <span className="admin-tab__description">{tab.description}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function toBboxTuple(values: number[]): [number, number, number, number] {
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0, values[3] ?? 0];
}

function ElementEditor({
  element,
  mutating,
  onCancel,
  onModify,
}: {
  element: AdminAnnotationElement;
  mutating: boolean;
  onCancel: () => void;
  onModify: (element: AdminAnnotationElement, payload: AdminAnnotationModifyPayload) => void;
}) {
  const [className, setClassName] = useState(element.class_name);
  const [bbox, setBbox] = useState<[number, number, number, number]>(toBboxTuple(element.bbox));
  const [clientError, setClientError] = useState<string | null>(null);

  const setBboxValue = (position: number, value: string) => {
    const next = [...bbox] as [number, number, number, number];
    next[position] = Number(value);
    setBbox(next);
  };

  const submit = (approveAfterSave: boolean) => {
    setClientError(null);
    if (!className.trim()) {
      setClientError('Class name is required before saving changes.');
      return;
    }
    if (bbox.some((value) => !Number.isFinite(value))) {
      setClientError('BBox values must be finite numbers.');
      return;
    }
    if (bbox[2] <= 0 || bbox[3] <= 0) {
      setClientError('BBox width and height must be positive.');
      return;
    }

    onModify(element, {
      class_name: className,
      bbox,
      approve_after_save: approveAfterSave || undefined,
    });
  };

  return (
    <form className="ui-section mt-3 p-4" onSubmit={(event) => event.preventDefault()}>
      <div className="flex flex-col gap-3">
        <div>
          <label className="ui-title-sm" htmlFor={`class-${element.key}`}>
            Class name
          </label>
          <input
            id={`class-${element.key}`}
            className="ui-input mt-1 w-full px-3 py-2"
            value={className}
            disabled={mutating}
            onChange={(event) => setClassName(event.target.value)}
          />
        </div>

        <fieldset className="grid gap-2 sm:grid-cols-4">
          <legend className="ui-title-sm sm:col-span-4">BBox [x, y, w, h]</legend>
          {['x', 'y', 'w', 'h'].map((label, index) => (
            <label key={label} className="ui-text-caption flex flex-col gap-1 uppercase tracking-wide">
              {label}
              <input
                className="ui-input px-3 py-2"
                type="number"
                value={bbox[index]}
                disabled={mutating}
                onChange={(event) => setBboxValue(index, event.target.value)}
                aria-label={`BBox ${label} for element ${element.index}`}
              />
            </label>
          ))}
        </fieldset>

        <p className="ui-text-caption">
          Saving rewrites canonical metadata and regenerates the crop. Save changes returns the element to pending;
          Save & approve records a fresh approved fingerprint after the crop exists.
        </p>

        {clientError ? <div role="alert" className="ui-alert ui-alert--danger p-3">{clientError}</div> : null}

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="ui-action-ghost rounded-full px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            disabled={mutating}
            onClick={() => submit(false)}
          >
            Save changes
          </button>
          <button
            type="button"
            className="ui-action-primary px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-50"
            disabled={mutating}
            onClick={() => submit(true)}
          >
            Save & approve
          </button>
          <button
            type="button"
            className="ui-action-ghost rounded-full px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            disabled={mutating}
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      </div>
    </form>
  );
}

type ReviewStatusFilter = AdminAnnotationReviewStatus | 'all';

type ReviewRow = {
  analysis: AdminAnnotationAnalysis;
  element: AdminAnnotationElement;
  diagnostics: string[];
};

const REVIEW_STATUS_FILTER_LABEL: Record<ReviewStatusFilter, string> = {
  all: 'All statuses',
  pending: 'Pending',
  approved: 'Approved',
  rejected: 'Rejected',
};

function diagnosticsByElementKey(queue: AdminAnnotationQueue) {
  const diagnosticsByKey = new Map<string, string[]>();
  for (const diagnostic of queue.diagnostics) {
    if (!diagnostic.key) {
      continue;
    }
    diagnosticsByKey.set(diagnostic.key, [
      ...(diagnosticsByKey.get(diagnostic.key) ?? []),
      `${diagnostic.code}: ${diagnostic.message}`,
    ]);
  }
  return diagnosticsByKey;
}

function reviewRows(queue: AdminAnnotationQueue): ReviewRow[] {
  const diagnosticsByKey = diagnosticsByElementKey(queue);
  return queue.analyses.flatMap((analysis) =>
    analysis.elements.map((element) => ({
      analysis,
      element,
      diagnostics: diagnosticsByKey.get(element.key) ?? [],
    })),
  );
}

function trainabilityCopy(row: ReviewRow) {
  const { element, diagnostics } = row;
  if (element.trainable) {
    return 'Approved with a fresh source fingerprint and crop present; this item is eligible for approved-only training.';
  }
  if (element.review_status === 'rejected') {
    return 'Rejected elements are excluded from training.';
  }
  if (element.review_status === 'pending') {
    return 'Pending elements do not train until an operator approves them.';
  }
  if (!element.crop_exists) {
    return 'Approved but missing a crop, so this item is blocked from training until the crop is regenerated.';
  }
  if (element.stale_decision) {
    return 'Approved but the saved decision is stale; save or reapprove after regeneration before training.';
  }
  if (diagnostics.length) {
    return diagnostics[0];
  }
  return 'Approved but not currently trainable; inspect diagnostics before launching training.';
}

function ReviewQueueRow({ row, selected, onSelect }: { row: ReviewRow; selected: boolean; onSelect: (element: AdminAnnotationElement) => void }) {
  const { analysis, element, diagnostics } = row;
  return (
    <li>
      <button
        type="button"
        role="option"
        aria-selected={selected}
        aria-label={`Select review element ${element.index} ${element.class_name || 'Unnamed'} from ${analysis.analysis_id}`}
        className={`ui-row grid w-full gap-3 p-3 text-left transition md:grid-cols-[5rem_1fr] ${selected ? 'outline outline-2 outline-[color:var(--accent-primary)]' : ''}`}
        onClick={() => onSelect(element)}
      >
        <span className="ui-crop-shell flex h-16 items-center justify-center overflow-hidden">
          {element.crop_exists ? (
            <img
              src={adminAnnotationMediaUrl(element.crop_url)}
              alt={`Queue crop ${element.index} for ${element.class_name}`}
              className="h-16 w-full object-contain"
            />
          ) : (
            <span className="px-2 text-center text-xs text-[color:var(--danger-text)]">Missing crop</span>
          )}
        </span>
        <span className="min-w-0 space-y-2">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-[color:var(--text-heading)]">
              #{element.index} · {element.class_name || 'Unnamed'}
            </span>
            <StatusBadge status={element.review_status} />
            {element.trainable ? <span className="ui-chip ui-chip--ready">Trainable</span> : null}
            {!element.trainable && element.review_status === 'approved' ? <span className="ui-chip ui-chip--accent">Needs diagnostics</span> : null}
          </span>
          <span className="block ui-text-caption">
            {analysis.analysis_id} · BBox [{formatBbox(element.bbox)}] · {element.crop_exists ? 'crop present' : 'crop missing'}
          </span>
          {diagnostics.length ? <span className="block truncate ui-text-caption">{diagnostics[0]}</span> : null}
        </span>
      </button>
    </li>
  );
}

function ReviewElementInspector({
  row,
  filteredRows,
  selectedIndex,
  mutating,
  editing,
  onSelect,
  onReview,
  onModify,
  onEdit,
  onCancelEdit,
}: {
  row: ReviewRow | null;
  filteredRows: ReviewRow[];
  selectedIndex: number;
  mutating: boolean;
  editing: boolean;
  onSelect: (element: AdminAnnotationElement) => void;
  onReview: (element: AdminAnnotationElement, status: AdminAnnotationReviewStatus) => void;
  onModify: (element: AdminAnnotationElement, payload: AdminAnnotationModifyPayload) => void;
  onEdit: (element: AdminAnnotationElement) => void;
  onCancelEdit: () => void;
}) {
  if (!row) {
    return (
      <aside className="ui-empty-state p-6" aria-label="Selected element inspector">
        Select a compact queue row to inspect one annotation element.
      </aside>
    );
  }

  const { analysis, element, diagnostics } = row;
  const canGoPrevious = selectedIndex > 0;
  const canGoNext = selectedIndex >= 0 && selectedIndex < filteredRows.length - 1;

  return (
    <aside className="ui-panel p-5 lg:sticky lg:top-4" aria-labelledby="review-inspector-heading">
      <div className="flex flex-col gap-4">
        <div>
          <p className="ui-text-eyebrow">Selected inspector</p>
          <h2 id="review-inspector-heading" className="mt-2 text-xl font-bold text-[color:var(--text-heading)]">
            Element #{element.index} · {element.class_name || 'Unnamed'}
          </h2>
          <p className="mt-1 break-all ui-text-caption">Analysis {analysis.analysis_id}</p>
        </div>

        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
          <div>
            <div className="mb-1 ui-text-caption">Original image</div>
            <div className="ui-crop-shell overflow-hidden">
              {analysis.image_exists ? (
                <img
                  src={adminAnnotationMediaUrl(analysis.image_url)}
                  alt={`Original submission ${analysis.analysis_id}`}
                  className="h-40 w-full object-contain"
                />
              ) : (
                <div className="flex h-40 items-center justify-center px-3 text-center text-sm text-[color:var(--danger-text)]">
                  Missing original image
                </div>
              )}
            </div>
          </div>
          <div>
            <div className="mb-1 ui-text-caption">Selected crop</div>
            <div className="ui-crop-shell overflow-hidden">
              {element.crop_exists ? (
                <img
                  src={adminAnnotationMediaUrl(element.crop_url)}
                  alt={`Crop ${element.index} for ${element.class_name}`}
                  className="h-40 w-full object-contain"
                />
              ) : (
                <div className="flex h-40 items-center justify-center px-2 text-center text-sm text-[color:var(--danger-text)]">
                  Missing crop
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <StatusBadge status={element.review_status} />
          {element.trainable ? <span className="ui-chip ui-chip--ready">Trainable</span> : <span className="ui-chip ui-chip--accent">Not trainable</span>}
          {element.stale_decision ? <span className="ui-chip ui-chip--accent">Stale decision</span> : null}
        </div>

        <dl className="grid gap-2 text-sm md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
          <div><dt className="ui-text-caption">BBox</dt><dd>[{formatBbox(element.bbox)}]</dd></div>
          <div><dt className="ui-text-caption">Source fingerprint</dt><dd className="break-all">{element.source_fingerprint}</dd></div>
          <div className="md:col-span-2 lg:col-span-1 xl:col-span-2">
            <dt className="ui-text-caption">Crop path</dt><dd className="break-all">{element.crop_path}</dd>
          </div>
        </dl>

        <section className="ui-section p-4" aria-label="Trainability diagnostics">
          <h3 className="ui-title-sm">Training eligibility</h3>
          <p className="mt-2 ui-text-body-sm">{trainabilityCopy(row)}</p>
          {diagnostics.length ? (
            <ul className="mt-2 list-disc space-y-1 pl-5 ui-text-caption">
              {diagnostics.map((diagnostic) => <li key={diagnostic}>{diagnostic}</li>)}
            </ul>
          ) : null}
        </section>

        <nav className="flex flex-wrap gap-2" aria-label="Inspector navigation">
          <button
            type="button"
            className="ui-action-ghost rounded-full px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!canGoPrevious}
            onClick={() => onSelect(filteredRows[selectedIndex - 1].element)}
          >
            Previous element
          </button>
          <button
            type="button"
            className="ui-action-ghost rounded-full px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!canGoNext}
            onClick={() => onSelect(filteredRows[selectedIndex + 1].element)}
          >
            Next element
          </button>
        </nav>

        {selectedIndex < 0 ? (
          <div className="ui-alert ui-alert--accent p-3 text-sm">
            The selected element is outside the active filters. Clear filters to use previous/next navigation.
          </div>
        ) : null}

        <section className="ui-section p-4" aria-label="Review actions and consequences">
          <h3 className="ui-title-sm">Actions and consequences</h3>
          <div className="mt-3 grid gap-3">
            <div>
              <button
                type="button"
                className="ui-action-primary px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-50"
                disabled={mutating || element.review_status === 'approved'}
                onClick={() => onReview(element, 'approved')}
              >
                Approve element {element.index}
              </button>
              <p className="mt-1 ui-text-caption">Marks this element approved and eligible only when crop and fingerprint checks are valid.</p>
            </div>
            <div>
              <button
                type="button"
                className="ui-action-ghost admin-action-danger rounded-full px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                disabled={mutating || element.review_status === 'rejected'}
                onClick={() => onReview(element, 'rejected')}
              >
                Reject element {element.index}
              </button>
              <p className="mt-1 ui-text-caption">Excludes this element from approved-only training while keeping it visible for audit context.</p>
            </div>
            <div>
              <button
                type="button"
                className="ui-action-ghost rounded-full px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                disabled={mutating}
                onClick={() => onEdit(element)}
              >
                Edit element {element.index}
              </button>
              <p className="mt-1 ui-text-caption">Changes class or bbox, regenerates the crop, and requires an explicit save before training.</p>
            </div>
          </div>
        </section>

        {editing ? (
          <ElementEditor
            element={element}
            mutating={mutating}
            onCancel={onCancelEdit}
            onModify={onModify}
          />
        ) : null}
      </div>
    </aside>
  );
}

function QueueCounters({ queue }: { queue: AdminAnnotationQueue }) {
  return (
    <section aria-label="Review counters" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      <CountCard label="Total" value={queue.counts.total} />
      <CountCard label="Pending" value={queue.counts.pending} />
      <CountCard label="Approved" value={queue.counts.approved} />
      <CountCard label="Rejected" value={queue.counts.rejected} />
      <CountCard label="Trainable" value={queue.counts.trainable} />
    </section>
  );
}

function QueueDiagnostics({ queue }: { queue: AdminAnnotationQueue }) {
  if (!queue.diagnostics.length) {
    return null;
  }

  return (
    <section className="ui-alert ui-alert--accent p-4">
      <h2 className="font-semibold">Diagnostics</h2>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
        {queue.diagnostics.map((diagnostic, idx) => (
          <li key={`${diagnostic.code}-${diagnostic.key ?? idx}`}>
            <span className="font-medium">{diagnostic.code}:</span> {diagnostic.message}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ReviewTab({
  queue,
  selectedKey,
  mutatingKey,
  editingKey,
  onSelect,
  onReview,
  onModify,
  onEdit,
  onCancelEdit,
}: {
  queue: AdminAnnotationQueue;
  selectedKey: string | null;
  mutatingKey: string | null;
  editingKey: string | null;
  onSelect: (element: AdminAnnotationElement) => void;
  onReview: (element: AdminAnnotationElement, status: AdminAnnotationReviewStatus) => void;
  onModify: (element: AdminAnnotationElement, payload: AdminAnnotationModifyPayload) => void;
  onEdit: (element: AdminAnnotationElement) => void;
  onCancelEdit: () => void;
}) {
  const [statusFilter, setStatusFilter] = useState<ReviewStatusFilter>('all');
  const [classFilter, setClassFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const rows = useMemo(() => reviewRows(queue), [queue]);
  const classOptions = useMemo(() => [...new Set(rows.map((row) => row.element.class_name || 'Unnamed'))].sort(), [rows]);
  const filteredRows = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return rows.filter((row) => {
      const className = row.element.class_name || 'Unnamed';
      const statusMatches = statusFilter === 'all' || row.element.review_status === statusFilter;
      const classMatches = classFilter === 'all' || className === classFilter;
      const queryMatches = !normalizedQuery || [
        row.analysis.analysis_id,
        String(row.element.index),
        className,
        formatBbox(row.element.bbox),
        ...row.diagnostics,
      ].join(' ').toLowerCase().includes(normalizedQuery);
      return statusMatches && classMatches && queryMatches;
    });
  }, [classFilter, rows, searchQuery, statusFilter]);

  if (!rows.length) {
    return <div className="ui-empty-state p-6">No submitted annotations are waiting for review.</div>;
  }

  const selectedRow = rows.find((row) => row.element.key === selectedKey) ?? rows[0] ?? null;
  const selectedIndex = filteredRows.findIndex((row) => row.element.key === selectedRow?.element.key);

  return (
    <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(22rem,0.85fr)]">
      <div className="min-w-0 space-y-4">
        <div className="ui-panel p-5">
          <p className="ui-text-eyebrow">Compact review queue</p>
          <h2 className="mt-2 text-xl font-bold text-[color:var(--text-heading)]">Select one element, inspect one detail panel</h2>
          <p className="mt-2 ui-text-body-sm">
            Use status/class filters or text search to shorten long lists. Selection is separate from editing, so the queue stays compact while the inspector keeps context.
          </p>
        </div>

        <div className="ui-section grid gap-3 p-4 md:grid-cols-3">
          <label className="ui-title-sm flex flex-col gap-2">
            Review status filter
            <select
              className="ui-select px-3 py-2"
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value as ReviewStatusFilter)}
            >
              {Object.entries(REVIEW_STATUS_FILTER_LABEL).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label className="ui-title-sm flex flex-col gap-2">
            Review class filter
            <select className="ui-select px-3 py-2" value={classFilter} onChange={(event) => setClassFilter(event.target.value)}>
              <option value="all">All classes</option>
              {classOptions.map((className) => <option key={className} value={className}>{className}</option>)}
            </select>
          </label>
          <label className="ui-title-sm flex flex-col gap-2">
            Search queue
            <input
              className="ui-input px-3 py-2"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Analysis, class, bbox…"
            />
          </label>
        </div>

        {filteredRows.length ? (
          <ul role="listbox" aria-label="Compact review queue" className="space-y-2">
            {filteredRows.map((row) => (
              <ReviewQueueRow
                key={row.element.key}
                row={row}
                selected={row.element.key === selectedRow?.element.key}
                onSelect={onSelect}
              />
            ))}
          </ul>
        ) : (
          <div className="ui-empty-state p-6">No review elements match the active filters.</div>
        )}
      </div>

      <ReviewElementInspector
        row={selectedRow}
        filteredRows={filteredRows}
        selectedIndex={selectedIndex}
        mutating={mutatingKey === selectedRow?.element.key}
        editing={editingKey === selectedRow?.element.key}
        onSelect={onSelect}
        onReview={onReview}
        onModify={onModify}
        onEdit={onEdit}
        onCancelEdit={onCancelEdit}
      />
    </section>
  );
}

type DatasetBucket = 'trainable' | 'approved_nontrainable' | 'rejected' | 'pending';
type DatasetStatusFilter = DatasetBucket | 'all';

type DatasetRow = {
  element: AdminAnnotationElement;
  bucket: DatasetBucket;
  diagnostics: string[];
};

const DATASET_BUCKET_LABEL: Record<DatasetStatusFilter, string> = {
  all: 'All statuses',
  trainable: 'Trainable approved',
  approved_nontrainable: 'Approved nontrainable',
  rejected: 'Rejected',
  pending: 'Pending',
};

function datasetBucketFor(element: AdminAnnotationElement): DatasetBucket {
  if (element.trainable) {
    return 'trainable';
  }
  if (element.review_status === 'approved') {
    return 'approved_nontrainable';
  }
  return element.review_status;
}

function datasetRows(queue: AdminAnnotationQueue): DatasetRow[] {
  const diagnosticsByKey = diagnosticsByElementKey(queue);

  return queue.analyses.flatMap((analysis) =>
    analysis.elements.map((element) => ({
      element,
      bucket: datasetBucketFor(element),
      diagnostics: diagnosticsByKey.get(element.key) ?? [],
    })),
  );
}

function classDistribution(rows: DatasetRow[], bucket: DatasetBucket): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const row of rows) {
    if (row.bucket !== bucket) {
      continue;
    }
    const className = row.element.class_name || 'Unnamed';
    counts.set(className, (counts.get(className) ?? 0) + 1);
  }
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
}

function DatasetMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="ui-section p-3">
      <div className="ui-text-caption font-semibold uppercase tracking-wide">{label}</div>
      <div className="mt-1 text-xl font-black text-[color:var(--text-main)]">{value}</div>
    </div>
  );
}

function DatasetDistribution({ title, rows }: { title: string; rows: Array<[string, number]> }) {
  return (
    <section className="ui-section p-4">
      <h3 className="ui-title-sm">{title}</h3>
      {rows.length ? (
        <ul className="mt-3 flex flex-wrap gap-2">
          {rows.map(([className, count]) => (
            <li key={className} className="ui-chip">
              {className}: {count}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 ui-text-caption">No items in this bucket.</p>
      )}
    </section>
  );
}

function DatasetCard({ row, selected, onSelect }: { row: DatasetRow; selected: boolean; onSelect: (element: AdminAnnotationElement) => void }) {
  const { element } = row;
  return (
    <li>
      <button
        type="button"
        role="option"
        aria-selected={selected}
        aria-label={`Select dataset element ${element.index} ${element.class_name || 'Unnamed'}`}
        className={`ui-row grid w-full gap-3 p-4 text-left md:grid-cols-[8rem_1fr] ${selected ? 'outline outline-2 outline-[color:var(--accent-primary)]' : ''}`}
        onClick={() => onSelect(element)}
      >
        <span className="ui-crop-shell flex h-28 items-center justify-center overflow-hidden">
          {element.crop_exists ? (
            <img
              src={adminAnnotationMediaUrl(element.crop_url)}
              alt={`Dataset crop ${element.index} for ${element.class_name}`}
              className="h-28 w-full object-contain"
            />
          ) : (
            <span className="px-2 text-center text-sm text-[color:var(--danger-text)]">Missing crop</span>
          )}
        </span>
        <span className="min-w-0 space-y-2">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-[color:var(--text-heading)]">
              {element.class_name || 'Unnamed'} · {element.analysis_id} #{element.index}
            </span>
            <StatusBadge status={element.review_status} />
            <span className={`ui-chip ${row.bucket === 'trainable' ? 'ui-chip--ready' : row.bucket === 'rejected' ? 'ui-chip--danger' : 'ui-chip--accent'}`}>
              {DATASET_BUCKET_LABEL[row.bucket]}
            </span>
          </span>
          <span className="block ui-text-body-sm">
            BBox [{formatBbox(element.bbox)}] · {element.crop_exists ? 'crop present' : 'crop missing'}
          </span>
          {element.stale_decision ? <span className="block ui-text-caption">Stale decision: this item is not trainable.</span> : null}
          {row.diagnostics.length ? <span className="block truncate ui-text-caption">{row.diagnostics[0]}</span> : null}
          {row.bucket === 'rejected' ? (
            <span className="block ui-text-caption">Rejected crops are retained for review context and do not train.</span>
          ) : null}
        </span>
      </button>
    </li>
  );
}

function DatasetInspector({ row, onJumpToReview }: { row: DatasetRow | null; onJumpToReview: (element: AdminAnnotationElement) => void }) {
  if (!row) {
    return <aside className="ui-empty-state p-6">Select a dataset row to inspect training diagnostics.</aside>;
  }

  const { element } = row;
  return (
    <aside className="ui-panel p-5 lg:sticky lg:top-4" aria-labelledby="dataset-inspector-heading">
      <div className="flex flex-col gap-4">
        <div>
          <p className="ui-text-eyebrow">Dataset inspector</p>
          <h2 id="dataset-inspector-heading" className="mt-2 text-xl font-bold text-[color:var(--text-heading)]">
            Dataset element #{element.index} · {element.class_name || 'Unnamed'}
          </h2>
          <p className="mt-1 break-all ui-text-caption">{element.analysis_id}</p>
        </div>

        <div className="ui-crop-shell overflow-hidden">
          {element.crop_exists ? (
            <img
              src={adminAnnotationMediaUrl(element.crop_url)}
              alt={`Selected dataset crop ${element.index} for ${element.class_name}`}
              className="h-48 w-full object-contain"
            />
          ) : (
            <div className="flex h-48 items-center justify-center px-2 text-center text-sm text-[color:var(--danger-text)]">
              Missing crop
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <StatusBadge status={element.review_status} />
          <span className={`ui-chip ${row.bucket === 'trainable' ? 'ui-chip--ready' : row.bucket === 'rejected' ? 'ui-chip--danger' : 'ui-chip--accent'}`}>
            {DATASET_BUCKET_LABEL[row.bucket]}
          </span>
        </div>

        <dl className="grid gap-2 text-sm md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
          <div><dt className="ui-text-caption">BBox</dt><dd>[{formatBbox(element.bbox)}]</dd></div>
          <div><dt className="ui-text-caption">Crop</dt><dd>{element.crop_exists ? 'Present' : 'Missing'}</dd></div>
          <div className="md:col-span-2 lg:col-span-1 xl:col-span-2"><dt className="ui-text-caption">Crop path</dt><dd className="break-all">{element.crop_path}</dd></div>
        </dl>

        <section className="ui-section p-4" aria-label="Dataset training diagnostics">
          <h3 className="ui-title-sm">Training diagnostics</h3>
          <p className="mt-2 ui-text-body-sm">
            {row.bucket === 'trainable'
              ? 'This crop is approved, fresh, present, and included in approved-only training.'
              : row.bucket === 'rejected'
                ? 'Rejected crops are excluded from training and kept visible for review context.'
                : row.bucket === 'pending'
                  ? 'Pending crops are excluded until an operator approves them.'
                  : 'Approved but currently nontrainable; resolve diagnostics before expecting it in training.'}
          </p>
          {row.diagnostics.length ? (
            <ul className="mt-2 list-disc space-y-1 pl-5 ui-text-caption">
              {row.diagnostics.map((diagnostic) => <li key={diagnostic}>{diagnostic}</li>)}
            </ul>
          ) : null}
        </section>

        <button
          type="button"
          className="ui-action-ghost rounded-full px-3 py-2 text-sm font-semibold"
          onClick={() => onJumpToReview(element)}
        >
          Jump to review element {element.index}
        </button>
      </div>
    </aside>
  );
}

function DatasetTab({ queue, onJumpToReview }: { queue: AdminAnnotationQueue; onJumpToReview: (element: AdminAnnotationElement) => void }) {
  const [statusFilter, setStatusFilter] = useState<DatasetStatusFilter>('all');
  const [classFilter, setClassFilter] = useState('all');
  const [selectedDatasetKey, setSelectedDatasetKey] = useState<string | null>(null);
  const rows = useMemo(() => datasetRows(queue), [queue]);
  const classOptions = useMemo(() => [...new Set(rows.map((row) => row.element.class_name || 'Unnamed'))].sort(), [rows]);
  const filteredRows = useMemo(() => rows.filter((row) => {
    const statusMatches = statusFilter === 'all' || row.bucket === statusFilter;
    const classMatches = classFilter === 'all' || (row.element.class_name || 'Unnamed') === classFilter;
    return statusMatches && classMatches;
  }), [classFilter, rows, statusFilter]);
  const bucketCounts = useMemo(() => rows.reduce<Record<DatasetBucket, number>>(
    (counts, row) => ({ ...counts, [row.bucket]: counts[row.bucket] + 1 }),
    { trainable: 0, approved_nontrainable: 0, rejected: 0, pending: 0 },
  ), [rows]);
  const selectedRow = filteredRows.find((row) => row.element.key === selectedDatasetKey) ?? filteredRows[0] ?? null;

  return (
    <section className="space-y-4">
      <div className="ui-panel p-5">
        <p className="ui-text-eyebrow">Dataset</p>
        <h2 className="mt-2 text-xl font-bold text-[color:var(--text-heading)]">Approved-only dataset overview</h2>
        <p className="mt-2 ui-text-body-sm">
          Trainable crops are exactly approved, fresh, crop-present elements. Rejected and pending crops stay visible for
          operator diagnostics but are excluded from training.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <DatasetMetric label="Trainable" value={bucketCounts.trainable} />
        <DatasetMetric label="Approved nontrainable" value={bucketCounts.approved_nontrainable} />
        <DatasetMetric label="Rejected" value={bucketCounts.rejected} />
        <DatasetMetric label="Pending" value={bucketCounts.pending} />
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <DatasetDistribution title="Trainable class distribution" rows={classDistribution(rows, 'trainable')} />
        <DatasetDistribution title="Rejected class distribution" rows={classDistribution(rows, 'rejected')} />
      </div>

      <div className="ui-section grid gap-3 p-4 md:grid-cols-2">
        <label className="ui-title-sm flex flex-col gap-2">
          Dataset status filter
          <select
            className="ui-select px-3 py-2"
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value as DatasetStatusFilter)}
          >
            {Object.entries(DATASET_BUCKET_LABEL).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <label className="ui-title-sm flex flex-col gap-2">
          Dataset class filter
          <select className="ui-select px-3 py-2" value={classFilter} onChange={(event) => setClassFilter(event.target.value)}>
            <option value="all">All classes</option>
            {classOptions.map((className) => <option key={className} value={className}>{className}</option>)}
          </select>
        </label>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(22rem,0.85fr)]">
        {filteredRows.length ? (
          <ul role="listbox" aria-label="Dataset review rows" className="grid gap-3">
            {filteredRows.map((row) => (
              <DatasetCard
                key={row.element.key}
                row={row}
                selected={row.element.key === selectedRow?.element.key}
                onSelect={(element) => setSelectedDatasetKey(element.key)}
              />
            ))}
          </ul>
        ) : (
          <div className="ui-empty-state p-6">No dataset crops match the selected filters.</div>
        )}

        <DatasetInspector row={selectedRow} onJumpToReview={onJumpToReview} />
      </div>
    </section>
  );
}

function formatJsonValue(value: unknown) {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return JSON.stringify(value);
}

function TrainingMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="ui-section p-3">
      <div className="ui-text-caption font-semibold uppercase tracking-wide">{label}</div>
      <div className="mt-1 text-xl font-black text-[color:var(--text-main)]">{value}</div>
    </div>
  );
}

const TRAINING_JOB_POLL_INTERVAL_MS = 1000;

function formatClassSummary(classes: string[]) {
  return classes.length ? classes.join(', ') : 'No approved classes yet';
}

function TrainingJobPanel({ job }: { job: AdminTrainingJob | null }) {
  if (!job) {
    return <div className="ui-empty-state p-4">No local training job has been recorded yet.</div>;
  }

  return (
    <section className="ui-section p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="ui-title-sm">Latest job {job.run_id}</h3>
        <span className={`ui-chip ${job.status === 'succeeded' ? 'ui-chip--ready' : job.status === 'failed' ? 'ui-chip--danger' : 'ui-chip--accent'}`}>
          {job.status}
        </span>
        {job.dry_run ? <span className="ui-chip">Dry run</span> : null}
      </div>
      <dl className="mt-3 grid gap-2 text-sm md:grid-cols-2">
        <div><dt className="ui-text-caption">Device</dt><dd>{job.device}</dd></div>
        <div><dt className="ui-text-caption">Batch size</dt><dd>{job.batch_size}</dd></div>
        <div><dt className="ui-text-caption">Started</dt><dd>{job.started_at ?? '—'}</dd></div>
        <div><dt className="ui-text-caption">Exit code</dt><dd>{job.exit_code ?? '—'}</dd></div>
      </dl>
      {job.command?.length ? (
        <p className="mt-3 ui-text-caption">Command: <code>{job.command.join(' ')}</code></p>
      ) : null}
      {job.log_tail?.length ? (
        <pre className="mt-3 max-h-52 overflow-auto rounded-lg bg-[color:var(--surface-strong)] p-3 text-xs text-[color:var(--text-body)]">
          {job.log_tail.join('\n')}
        </pre>
      ) : (
        <p className="mt-3 ui-text-caption">No log output captured yet.</p>
      )}
    </section>
  );
}

function TrainingTab() {
  const [summary, setSummary] = useState<AdminTrainingSummary | null>(null);
  const [latestJob, setLatestJob] = useState<AdminTrainingJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [dryRun, setDryRun] = useState(true);
  const [device, setDevice] = useState('auto');
  const [batchSize, setBatchSize] = useState(16);
  const [notes, setNotes] = useState('');

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await getAdminTrainingSummary();
      setSummary(next);
      setLatestJob(next.latest_job ?? null);
      setDevice(next.parameters.editable.device[0] ?? 'auto');
      setBatchSize(next.parameters.editable.batch_size.default);
    } catch {
      setError('Unable to load local training summary.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void loadSummary();
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [loadSummary]);

  useEffect(() => {
    if (latestJob?.status !== 'running') {
      return;
    }
    const interval = window.setInterval(async () => {
      try {
        const response = await getLatestAdminTrainingJob();
        setLatestJob(response.job);
      } catch {
        setError('Unable to refresh latest training job status.');
      }
    }, TRAINING_JOB_POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [latestJob?.status]);

  const startJob = async () => {
    if (!summary) {
      return;
    }
    if (latestJob?.status === 'running') {
      setError('A local training job is already running.');
      return;
    }
    setError(null);
    setStarting(true);
    try {
      const response = await startAdminTrainingJob({ dry_run: dryRun, device, batch_size: batchSize, notes });
      setLatestJob(response.job);
    } catch {
      setError('Could not start the guarded local training job. Check that the feature flag is enabled and the backend is local.');
    } finally {
      setStarting(false);
    }
  };

  if (loading) {
    return <div className="ui-panel p-4">Loading training summary…</div>;
  }

  if (!summary) {
    return <div role="alert" className="ui-alert ui-alert--danger p-4">{error ?? 'Training summary is unavailable.'}</div>;
  }

  const jobRunning = latestJob?.status === 'running';
  const disabled = !summary.launch_allowed_for_request || starting || jobRunning;
  const batchBounds = summary.parameters.editable.batch_size;
  const disabledByDefault = summary.launch_disabled_reasons.some((reason) => reason.startsWith('disabled_by_default:'));
  const modelDirOverrideActive = Boolean(summary.paths.model_dir_override_active);
  const launchState = !summary.launch_allowed_for_request
    ? 'Not launchable'
    : 'Ready for local launch';
  const launchMode = dryRun ? 'Dry run selected' : 'Full training selected';
  const launchModeCopy = dryRun
    ? 'Dry run validates approved-only export/training wiring and the candidate artifact generation path without writing runtime artifacts.'
    : 'Full training creates a candidate package under backend/model_registry/versions/<version_id>/; it does not modify the live backend/codex_model/ runtime. Inspect manifest, model-card, and checksums, then run scripts/promote_model.py <version_id> and restart the backend to activate it.';

  return (
    <section className="space-y-4">
      <div className="ui-panel p-5">
        <p className="ui-text-eyebrow">Training</p>
        <h2 className="mt-2 text-xl font-bold text-[color:var(--text-heading)]">Guarded local training</h2>
        <p className="mt-2 ui-text-body-sm">
          Browser launch stays disabled by default. A local operator must start the backend with
          <code> ENABLE_ADMIN_TRAINING_JOBS=1</code> from a loopback session before this tab can launch the
          Bash retrain wrapper.
        </p>
      </div>

      {!summary.launch_allowed_for_request ? (
        <div role="alert" className="ui-alert ui-alert--accent p-4">
          <strong>Not launchable: launch disabled for this request.</strong>
          <ul className="mt-2 list-disc pl-5 text-sm">
            {summary.launch_disabled_reasons.map((reason) => <li key={reason}>{reason}</li>)}
          </ul>
          {disabledByDefault ? (
            <p className="mt-2 text-sm">
              Enable intentionally by restarting the local backend with <code>ENABLE_ADMIN_TRAINING_JOBS=1</code>;
              the committed default remains disabled.
            </p>
          ) : null}
          <p className="mt-2 text-sm">CLI alternative: <code>bash scripts/retrain.sh --dry-run</code></p>
        </div>
      ) : null}

      {modelDirOverrideActive ? (
        <div role="alert" className="ui-alert ui-alert--accent p-4">
          <strong>MODEL_DIR override active.</strong>
          <p className="mt-2 text-sm">
            Promotion to <code>backend/codex_model/</code> may not affect the loaded runtime until <code>MODEL_DIR</code>
            is unset or promotion targets the matching explicit runtime path.
          </p>
        </div>
      ) : null}

      {error ? <div role="alert" className="ui-alert ui-alert--danger p-4">{error}</div> : null}

      <div className="grid gap-3 md:grid-cols-5">
        <TrainingMetric label="Trainable" value={summary.data.trainable} />
        <TrainingMetric label="Approved" value={summary.data.approved} />
        <TrainingMetric label="Rejected" value={summary.data.rejected} />
        <TrainingMetric label="Pending" value={summary.data.pending} />
        <TrainingMetric label="Classes" value={summary.data.classes.length} />
      </div>

      <section className="ui-section p-4">
        <h3 className="ui-title-sm">Approved per-class counts</h3>
        {Object.keys(summary.data.per_class).length ? (
          <ul className="mt-3 flex flex-wrap gap-2">
            {Object.entries(summary.data.per_class).map(([className, count]) => (
              <li key={className} className="ui-chip ui-chip--ready">{className}: {count}</li>
            ))}
          </ul>
        ) : (
          <p className="mt-2 ui-text-caption">No trainable approved crops yet.</p>
        )}
      </section>

      <section className="ui-section p-4" aria-label="Training pre-action summary">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="ui-title-sm">Pre-action summary</h3>
          <span className={`ui-chip ${summary.launch_allowed_for_request ? 'ui-chip--ready' : 'ui-chip--accent'}`}>
            {launchState}
          </span>
        </div>
        <p className="mt-2 ui-text-body-sm">
          <strong>{launchMode}:</strong> {launchModeCopy}
        </p>
        <dl className="mt-3 grid gap-2 text-sm md:grid-cols-2 lg:grid-cols-3">
          <div><dt className="ui-text-caption">Will train</dt><dd>{summary.data.trainable} approved crop{summary.data.trainable === 1 ? '' : 's'}</dd></div>
          <div><dt className="ui-text-caption">Classes</dt><dd>{formatClassSummary(summary.data.classes)}</dd></div>
          <div><dt className="ui-text-caption">Device</dt><dd>{device}</dd></div>
          <div><dt className="ui-text-caption">Batch size</dt><dd>{batchSize}</dd></div>
          <div><dt className="ui-text-caption">Latest job</dt><dd>{latestJob ? `${latestJob.run_id} (${latestJob.status})` : 'No recorded job'}</dd></div>
          <div><dt className="ui-text-caption">Launch permission</dt><dd>{summary.launch_allowed_for_request ? 'Allowed for this local request' : 'Blocked by backend guard'}</dd></div>
          <div><dt className="ui-text-caption">Activation</dt><dd>Promote candidate explicitly, then restart backend</dd></div>
        </dl>
      </section>

      <section className="ui-section grid gap-3 p-4 md:grid-cols-4">
        <label className="ui-title-sm flex flex-col gap-2">
          Dry run
          <select className="ui-select px-3 py-2" value={dryRun ? 'yes' : 'no'} onChange={(event) => setDryRun(event.target.value === 'yes')}>
            <option value="yes">Yes</option>
            <option value="no">No, full training</option>
          </select>
        </label>
        <label className="ui-title-sm flex flex-col gap-2">
          Device
          <select className="ui-select px-3 py-2" value={device} onChange={(event) => setDevice(event.target.value)}>
            {summary.parameters.editable.device.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </label>
        <label className="ui-title-sm flex flex-col gap-2">
          Batch size
          <input
            className="ui-input px-3 py-2"
            type="number"
            min={batchBounds.min}
            max={batchBounds.max}
            value={batchSize}
            onChange={(event) => setBatchSize(Number(event.target.value))}
          />
        </label>
        <label className="ui-title-sm flex flex-col gap-2">
          Notes
          <input className="ui-input px-3 py-2" value={notes} onChange={(event) => setNotes(event.target.value)} />
        </label>
        <div className="md:col-span-4">
          <p className="mb-3 ui-text-caption">
            Dry run checks the approved-only export/training path. Full training writes a candidate registry package,
            not the live runtime; inspect outputs, promote explicitly, then restart the backend.
          </p>
          <button
            type="button"
            className="ui-action-primary px-4 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-50"
            disabled={disabled}
            onClick={startJob}
          >
            {starting ? 'Starting…' : jobRunning ? 'Training job running' : dryRun ? 'Start dry run' : 'Start training'}
          </button>
        </div>
      </section>

      <TrainingJobPanel job={latestJob} />

      <section className="grid gap-3 lg:grid-cols-3">
        <div className="ui-section p-4">
          <h3 className="ui-title-sm">Artifact status</h3>
          <dl className="mt-3 space-y-2 text-sm">
            {Object.entries(summary.artifacts).map(([key, value]) => (
              <div key={key}><dt className="ui-text-caption">{key}</dt><dd className="break-all">{formatJsonValue(value)}</dd></div>
            ))}
          </dl>
        </div>
        <div className="ui-section p-4">
          <h3 className="ui-title-sm">Resolved paths</h3>
          <dl className="mt-3 space-y-2 text-sm">
            {Object.entries(summary.paths).map(([key, value]) => (
              <div key={key}><dt className="ui-text-caption">{key}</dt><dd className="break-all">{formatJsonValue(value)}</dd></div>
            ))}
          </dl>
        </div>
        <div className="ui-section p-4">
          <h3 className="ui-title-sm">Read-only config</h3>
          <dl className="mt-3 space-y-2 text-sm">
            {Object.entries(summary.parameters.config).flatMap(([section, value]) =>
              Object.entries((value as Record<string, unknown>) ?? {}).slice(0, 5).map(([key, entry]) => (
                <div key={`${section}.${key}`}><dt className="ui-text-caption">{section}.{key}</dt><dd>{formatJsonValue(entry)}</dd></div>
              )),
            )}
          </dl>
        </div>
      </section>
    </section>
  );
}

function AdminAnnotationsPage({ themeMode = 'dark', onToggleTheme }: AdminAnnotationsPageProps) {
  const [queue, setQueue] = useState<AdminAnnotationQueue | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [mutatingKey, setMutatingKey] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<AdminTab>('review');

  const loadQueue = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      setQueue(await getAdminAnnotationQueue());
    } catch {
      setLoadError('Unable to load the local annotation review queue.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void loadQueue();
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [loadQueue]);

  const handleSelectElement = useCallback((element: AdminAnnotationElement) => {
    setSelectedKey(element.key);
    setEditingKey((current) => (current === element.key ? current : null));
  }, []);

  const handleReview = async (
    element: AdminAnnotationElement,
    status: AdminAnnotationReviewStatus,
  ) => {
    setActionError(null);
    setSelectedKey(element.key);
    setMutatingKey(element.key);
    try {
      await setAdminAnnotationReviewStatus(element.analysis_id, element.index, status);
      setQueue(await getAdminAnnotationQueue());
    } catch {
      setActionError(`Could not mark element ${element.index} as ${status}. The visible status was not changed.`);
    } finally {
      setMutatingKey(null);
    }
  };

  const handleModify = async (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => {
    setActionError(null);
    setSelectedKey(element.key);
    setMutatingKey(element.key);
    try {
      await modifyAdminAnnotationElement(element.analysis_id, element.index, payload);
      setQueue(await getAdminAnnotationQueue());
      setEditingKey(null);
    } catch {
      setActionError(`Could not save changes for element ${element.index}. The visible annotation was not changed.`);
    } finally {
      setMutatingKey(null);
    }
  };

  const queueElementKeys = queue?.analyses.flatMap((analysis) => analysis.elements.map((element) => element.key)) ?? [];
  const effectiveSelectedKey = selectedKey && queueElementKeys.includes(selectedKey) ? selectedKey : (queueElementKeys[0] ?? null);
  const effectiveEditingKey = editingKey && queueElementKeys.includes(editingKey) ? editingKey : null;

  return (
    <div className="admin-console h-full overflow-auto p-6" data-theme={themeMode}>
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <AdminHeader themeMode={themeMode} onToggleTheme={onToggleTheme} />

        <div role="alert" className="ui-alert ui-alert--accent p-4">
          <strong>Local/dev-only:</strong> {queue?.warning ?? 'This admin page is not production-secured.'}
        </div>

        {loading ? <div className="ui-panel p-4">Loading review queue…</div> : null}

        {loadError ? (
          <div role="alert" className="ui-alert ui-alert--danger p-4">
            {loadError}
          </div>
        ) : null}

        {actionError ? (
          <div role="alert" className="ui-alert ui-alert--danger p-4">
            {actionError}
          </div>
        ) : null}

        {queue ? (
          <>
            <QueueCounters queue={queue} />
            <QueueDiagnostics queue={queue} />
            <AdminTabs activeTab={activeTab} onSelect={setActiveTab} />

            {ADMIN_TABS.map((tab) => (
              <section
                key={tab.id}
                id={`admin-${tab.id}-panel`}
                role="tabpanel"
                aria-label={`${tab.label} tab panel`}
                hidden={activeTab !== tab.id}
              >
                {activeTab === tab.id && tab.id === 'review' ? (
                  <ReviewTab
                    queue={queue}
                    selectedKey={effectiveSelectedKey}
                    mutatingKey={mutatingKey}
                    editingKey={effectiveEditingKey}
                    onSelect={handleSelectElement}
                    onReview={handleReview}
                    onModify={handleModify}
                    onEdit={(element) => {
                      setSelectedKey(element.key);
                      setEditingKey(element.key);
                    }}
                    onCancelEdit={() => setEditingKey(null)}
                  />
                ) : null}
                {activeTab === tab.id && tab.id === 'dataset' ? (
                  <DatasetTab
                    queue={queue}
                    onJumpToReview={(element) => {
                      setActiveTab('review');
                      setSelectedKey(element.key);
                      setEditingKey(null);
                    }}
                  />
                ) : null}
                {activeTab === tab.id && tab.id === 'training' ? <TrainingTab /> : null}
              </section>
            ))}
          </>
        ) : null}
      </div>
    </div>
  );
}

export default AdminAnnotationsPage;
