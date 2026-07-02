import { PanelLeftClose, PanelLeftOpen, Search, Trash2 } from 'lucide-react';
import type { AnalysisRecord } from '../types';
import { getWorkspaceElementClassName } from '../pages/workspace/workspaceViewUtils';
import sidebarStyles from './SidebarChrome.module.css';
import workspaceStyles from '../pages/workspace/WorkspaceChrome.module.css';

export type WorkspaceHistoryPanelLabels = {
  expandHistory: string;
  collapseHistory: string;
  filterPlaceholder: string;
  noAnalyses: string;
  noFilterMatch: string;
  elementsSuffix: string;
  rejectedSuffix: string;
  historyTitle: string;
  totalSuffix: string;
  deleteLabel: string;
};

export type WorkspaceHistoryPanelProps = {
  records: AnalysisRecord[];
  filteredRecords: AnalysisRecord[];
  currentRecordId: string | null;
  filter: string;
  historyOpen: boolean;
  labels: WorkspaceHistoryPanelLabels;
  onFilterChange: (value: string) => void;
  onToggleHistoryOpen: (open: boolean) => void;
  onSelectRecord: (record: AnalysisRecord) => void;
  onRemoveRecord: (id: string) => void;
};

export function WorkspaceHistoryPanel({
  records,
  filteredRecords,
  currentRecordId,
  filter,
  historyOpen,
  labels,
  onFilterChange,
  onToggleHistoryOpen,
  onSelectRecord,
  onRemoveRecord,
}: WorkspaceHistoryPanelProps) {
  return (
    <aside
      data-testid="workspace-history-sidebar"
      className={`${sidebarStyles.owner} ${workspaceStyles.owner} app-sidebar sidebar-shell flex flex-col overflow-hidden transition-[padding,background-color] duration-200 ease-out ${historyOpen ? 'min-h-0 rounded-none p-0' : 'min-h-0 items-center rounded-none py-0'}`}
    >
      {!historyOpen ? (
        <div className="flex h-full w-full flex-col items-center overflow-hidden">
          <button
            type="button"
            onClick={() => onToggleHistoryOpen(true)}
            className="ui-icon-button h-10 w-10 shrink-0 rounded-none"
            title={labels.expandHistory}
          >
            <PanelLeftOpen size={18} />
          </button>

          <div className="my-3 h-px w-8 shrink-0 bg-[var(--divider)]" />

          <div className="flex w-full flex-1 flex-col items-center gap-2 overflow-y-auto px-1 py-1">
            {records.map((record) => (
              <button
                key={record.id}
                type="button"
                onClick={() => onSelectRecord(record)}
                className={`ui-row ui-row--hover relative overflow-hidden rounded-none transition-colors ${
                  currentRecordId === record.id
                    ? 'ui-row--active outline outline-1 outline-[color:var(--border-strong)]'
                    : ''
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
          <div className="workspace-history-header app-sidebar__header sidebar-header mb-0 flex items-center justify-between gap-2" data-testid="workspace-history-header">
            <div className="flex min-w-0 items-center gap-2">
              <h2 className="ui-title-sm">{labels.historyTitle}</h2>
              <span className="workspace-history-count ui-chip tabular-nums">
                {filteredRecords.length} {labels.totalSuffix}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onToggleHistoryOpen(false)}
                className="ui-icon-button h-10 w-10 rounded-none"
                title={labels.collapseHistory}
              >
                <PanelLeftClose size={18} />
              </button>
            </div>
          </div>

          <div className="workspace-history-search relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[color:var(--text-muted)]" />
            <input
              value={filter}
              onChange={(event) => onFilterChange(event.target.value)}
              placeholder={labels.filterPlaceholder}
              className="ui-input w-full px-10 py-2.5"
            />
          </div>

          <div className="min-h-0 flex-1 space-y-0 overflow-y-auto pr-0" data-testid="workspace-history-list">
            {records.length === 0 ? (
              <div className="ui-empty-state px-4 py-8 text-center">
                {labels.noAnalyses}
              </div>
            ) : filteredRecords.length === 0 ? (
              <div className="ui-empty-state px-4 py-8 text-center">
                {labels.noFilterMatch}
              </div>
            ) : (
              filteredRecords.map((record) => {
                const isActive = currentRecordId === record.id;
                const badges = [
                  ...new Set(
                    record.result.elements
                      .map((element, idx) => (!element.rejected ? getWorkspaceElementClassName(record, idx) : null))
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
                    className={`selection-card workspace-history-row group relative rounded-none transition-colors ${isActive ? 'selection-card--active before:absolute before:inset-y-2 before:left-0 before:w-1 before:rounded-none before:bg-[var(--accent)]' : 'ui-row--hover'}`}
                  >
                    <div className="flex items-start gap-3">
                      <img src={record.imageDataUrl} alt={record.imageName} className="workspace-history-thumb h-16 w-16 rounded-none border border-[color:var(--border-subtle)] object-cover" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="ui-title-sm truncate">{record.imageName}</p>
                            <p className="ui-text-meta mt-1">{new Date(record.timestamp).toLocaleDateString()} · {record.result.num_elements} {labels.elementsSuffix}</p>
                          </div>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              onRemoveRecord(record.id);
                            }}
                            className="workspace-history-delete rounded-none p-1.5 text-[color:var(--text-muted)] transition-colors hover:bg-[var(--danger-soft)] hover:text-[var(--danger-text)]"
                            aria-label={`${labels.deleteLabel} ${record.imageName}`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {badges.map((badge) => (
                            <span key={badge} className="ui-chip ui-chip--accent">
                              {badge}
                            </span>
                          ))}
                          {record.result.elements.some((element) => element.rejected) && (
                            <span className="ui-chip ui-chip--danger">
                              {record.result.elements.filter((element) => element.rejected).length} {labels.rejectedSuffix}
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
