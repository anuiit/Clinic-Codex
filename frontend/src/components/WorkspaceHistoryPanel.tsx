import { PanelLeftClose, PanelLeftOpen, Search, Trash2 } from 'lucide-react';
import type { AnalysisRecord } from '../types';

type Props = {
  records: AnalysisRecord[];
  filteredRecords: AnalysisRecord[];
  currentRecordId: string | null;
  historyOpen: boolean;
  filter: string;
  onFilterChange: (value: string) => void;
  onSelectRecord: (record: AnalysisRecord) => void;
  onDeleteRecord: (id: string) => void;
  onToggleHistoryOpen: (open: boolean) => void;
  expandLabel: string;
  collapseLabel: string;
  filterPlaceholder: string;
  noAnalyses: string;
  noFilterMatch: string;
  elementsSuffix: string;
  deleteLabel: string;
};

export function WorkspaceHistoryPanel({
  records,
  filteredRecords,
  currentRecordId,
  historyOpen,
  filter,
  onFilterChange,
  onSelectRecord,
  onDeleteRecord,
  onToggleHistoryOpen,
  expandLabel,
  collapseLabel,
  filterPlaceholder,
  noAnalyses,
  noFilterMatch,
  elementsSuffix,
  deleteLabel,
}: Props) {
  return (
    <aside
      data-testid="workspace-history-sidebar"
      className={`flex flex-col overflow-hidden transition-[padding,border-color,background-color] duration-200 ease-out ${historyOpen ? 'min-h-0 rounded-2xl border border-stone-800 bg-stone-900/75 p-3' : 'min-h-0 items-center rounded-2xl border border-stone-800 bg-stone-900/75 py-3'}`}
    >
      {!historyOpen ? (
        <div className="flex h-full w-full flex-col items-center overflow-hidden">
          <button
            type="button"
            onClick={() => onToggleHistoryOpen(true)}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-stone-950 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100"
            title={expandLabel}
          >
            <PanelLeftOpen size={18} />
          </button>

          <div className="my-3 h-px w-8 shrink-0 bg-stone-800" />

          <div className="flex flex-1 w-full flex-col items-center gap-2 overflow-y-auto px-1 py-1">
            {records.map((record) => (
              <button
                key={record.id}
                type="button"
                onClick={() => onSelectRecord(record)}
                className={`relative overflow-hidden rounded-lg border transition-colors ${
                  currentRecordId === record.id
                    ? 'border-amber-500/60 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]'
                    : 'border-stone-800 hover:border-stone-600'
                }`}
                title={record.imageName}
              >
                <img
                  src={record.imageDataUrl}
                  alt={record.imageName}
                  className="h-10 w-10 object-cover"
                />
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-stone-800 pb-3" data-testid="workspace-history-header">
            <div className="flex min-w-0 items-center gap-2">
              <h2 className="text-sm font-semibold text-stone-100">History</h2>
              <span className="rounded-full border border-stone-800 bg-stone-950/70 px-2 py-0.5 text-xs font-medium tabular-nums text-stone-400">
                {filteredRecords.length} total
              </span>
            </div>
            <button
              type="button"
              onClick={() => onToggleHistoryOpen(false)}
              className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-950 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100"
              title={collapseLabel}
            >
              <PanelLeftClose size={18} />
            </button>
          </div>

          <div className="relative mb-3">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-500" />
            <input
              value={filter}
              onChange={(event) => onFilterChange(event.target.value)}
              placeholder={filterPlaceholder}
              className="w-full rounded-xl border border-stone-700 bg-stone-950 px-10 py-2.5 text-sm text-stone-100 outline-none transition-colors focus:border-amber-400"
            />
          </div>

          <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1" data-testid="workspace-history-list">
            {records.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                {noAnalyses}
              </div>
            ) : filteredRecords.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                {noFilterMatch}
              </div>
            ) : (
              filteredRecords.map((record) => {
                const isActive = currentRecordId === record.id;
                const badges = [
                  ...new Set(
                    record.result.elements
                      .map((element, idx) => (!element.rejected ? (record.annotations ?? {})[idx] ?? element.class_name : null))
                      .filter((value): value is string => Boolean(value)),
                  ),
                ].slice(0, 3);

                return (
                  <div
                    key={record.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => onSelectRecord(record)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onSelectRecord(record);
                      }
                    }}
                    className={`group rounded-xl border p-2.5 transition-colors ${isActive ? 'border-amber-500/60 bg-amber-500/10 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]' : 'border-stone-800 bg-stone-950/70 hover:border-stone-700'}`}
                  >
                    <div className="flex items-start gap-3">
                      <img src={record.imageDataUrl} alt={record.imageName} className="h-16 w-16 rounded-lg border border-stone-800 object-cover" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="truncate text-sm font-medium text-stone-100">{record.imageName}</p>
                            <p className="mt-1 text-xs text-stone-500">{new Date(record.timestamp).toLocaleDateString()} · {record.result.num_elements} {elementsSuffix}</p>
                          </div>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              onDeleteRecord(record.id);
                            }}
                            className="rounded-lg p-1.5 text-stone-500 transition-colors hover:bg-stone-800 hover:text-red-300"
                            aria-label={`${deleteLabel} ${record.imageName}`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {badges.map((badge) => (
                            <span key={badge} className="rounded-md bg-amber-400/10 px-2 py-1 text-[11px] text-amber-300">
                              {badge}
                            </span>
                          ))}
                          {record.result.elements.some((element) => element.rejected) && (
                            <span className="rounded-md bg-red-400/10 px-2 py-1 text-[11px] text-red-300">
                              {record.result.elements.filter((element) => element.rejected).length} rejected
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </>
      )}
    </aside>
  );
}

export default WorkspaceHistoryPanel;
