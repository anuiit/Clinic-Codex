import type { KeyboardEvent } from "react";
import { AlertCircle, CheckCircle2, ChevronLeft, Edit3, Info, Loader2 } from "lucide-react";
import type { AnalysisRecord, TrustResult } from "../../types";
import { adminAnnotationMediaUrl } from "../../services/api";
import type { WorkspaceTrustState } from "./useWorkspaceViewport";
import { useArchetypeAssets } from "./useArchetypeAssets";
import {
  getCropPreviewSize,
  getWorkspaceElementClassName,
  hasWorkspaceSubmittedAnnotation,
} from "./workspaceViewUtils";
import skeletonStyles from "../../components/LoadingSkeleton.module.css";
import sidebarStyles from "../../components/SidebarChrome.module.css";
import workspaceStyles from "./WorkspaceChrome.module.css";

export type WorkspaceStats = {
  total: number;
  rejectedCount: number;
  annotatedCount: number;
  submittedCount: number;
  topClass: string;
  topClasses: string[];
  imageSizeLabel: string;
};

type WorkspaceDetectedPanelLabels = {
  backToRegions: string;
  segmentPreview: string;
  trustSummary: string;
  recalculatedPrediction: string;
  initialProposal: string;
  ambiguousPrediction: string;
  ambiguousDetails: string;
  alternativesExist: string;
  lowConfidenceFlag: string;
  thresholdDetails: string;
  rank: string;
  margin: string;
  topPredictions: string;
  archetypeCoverage: string;
  correctElement: string;
  annotateRegion: string;
  proposalPanel: string;
  detectedElements: string;
  annotateRecord: string;
  allRejected: string;
  goToAnnotation: string;
  noElements: string;
};

type WorkspaceDetectedPanelProps = {
  record: AnalysisRecord;
  focusedIdx: number | null;
  hoveredIdx: number | null;
  stats: WorkspaceStats | null;
  trustState: WorkspaceTrustState;
  trustData: TrustResult | null;
  contextLoading: boolean;
  setCropCanvasRef: (idx: number, canvas: HTMLCanvasElement | null) => void;
  setDetailCanvasRef: (canvas: HTMLCanvasElement | null) => void;
  labels: WorkspaceDetectedPanelLabels;
  onBackToRegions: () => void;
  onEditorHandoff: () => void;
  onDetectedListKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  onFocusRegion: (idx: number) => void;
  onListRegionEnter: (idx: number) => void;
  onListRegionLeave: (idx: number) => void;
};

export default function WorkspaceDetectedPanel({
  record,
  focusedIdx,
  hoveredIdx,
  stats,
  trustState,
  trustData,
  contextLoading,
  setCropCanvasRef,
  setDetailCanvasRef,
  labels,
  onBackToRegions,
  onEditorHandoff,
  onDetectedListKeyDown,
  onFocusRegion,
  onListRegionEnter,
  onListRegionLeave,
}: WorkspaceDetectedPanelProps) {
  return (
    <section
      className={`${sidebarStyles.owner} ${workspaceStyles.owner} app-sidebar workspace-inspector-rail flex min-h-0 flex-col overflow-hidden rounded-none sidebar-shell`}
      data-testid="workspace-detected-panel"
    >
      {focusedIdx !== null ? (
        <WorkspaceFocusedRegionPanel
          record={record}
          focusedIdx={focusedIdx}
          trustState={trustState}
          trustData={trustData}
          contextLoading={contextLoading}
          setDetailCanvasRef={setDetailCanvasRef}
          labels={labels}
          onBackToRegions={onBackToRegions}
          onEditorHandoff={onEditorHandoff}
        />
      ) : (
        <WorkspaceDetectedList
          record={record}
          hoveredIdx={hoveredIdx}
          stats={stats}
          setCropCanvasRef={setCropCanvasRef}
          labels={labels}
          onEditorHandoff={onEditorHandoff}
          onDetectedListKeyDown={onDetectedListKeyDown}
          onFocusRegion={onFocusRegion}
          onListRegionEnter={onListRegionEnter}
          onListRegionLeave={onListRegionLeave}
        />
      )}
    </section>
  );
}

type FocusedPanelProps = Pick<
  WorkspaceDetectedPanelProps,
  | "record"
  | "focusedIdx"
  | "trustState"
  | "trustData"
  | "contextLoading"
  | "setDetailCanvasRef"
  | "labels"
  | "onBackToRegions"
  | "onEditorHandoff"
> & {
  focusedIdx: number;
};

function WorkspaceFocusedRegionPanel({
  record,
  focusedIdx,
  trustState,
  trustData,
  contextLoading,
  setDetailCanvasRef,
  labels,
  onBackToRegions,
  onEditorHandoff,
}: FocusedPanelProps) {
  const element = record.result.elements[focusedIdx];
  const trustMatches =
    trustState.recordId === record.id && trustState.focusedIdx === focusedIdx;
  const trustLoading = trustMatches && trustState.status === "loading";
  const trustError = trustMatches && trustState.status === "error";
  const trust = trustMatches && trustState.status === "ready" ? trustData?.trust : null;
  const summaryClass = trust?.top1_class ?? element.class_name;
  const topPrediction = trust?.top1_similarity ?? element.confidence;
  const runnerUp = trust?.top_k?.[1]?.confidence ?? element.top_k[1]?.confidence ?? 0;
  const margin = trust?.margin_to_second ?? topPrediction - runnerUp;
  const isRejected = trust ? !trust.above_rejection_threshold : element.rejected;
  const isAmbiguous = trust ? trust.ambiguous : false;
  const detailPreviewSize = getCropPreviewSize(element.bbox, 180);
  const initialPredictionDiffers = Boolean(trust && trust.top1_class !== element.class_name);
  const bboxSummary = element.bbox.map((value) => Math.round(value)).join(" · ");
  const topKItems = trust?.top_k ?? element.top_k;
  const archetypes = useArchetypeAssets(
    record.id,
    record.imageDataUrl,
    element.bbox,
    topKItems.map((item) => item.class_name),
  );

  return (
    <div className="workspace-focused-panel flex h-full flex-col">
      <div className="workspace-focused-header app-sidebar__header flex items-center justify-between sidebar-header">
        <button
          onClick={onBackToRegions}
          className="flex items-center gap-2 text-[color:var(--text-muted)] transition-colors hover:text-[var(--text-main)]"
        >
          <ChevronLeft size={16} />
          <span className="text-sm font-medium">{labels.backToRegions}</span>
        </button>
      </div>

      <div className="workspace-focused-body ui-scrollbar app-sidebar__body flex-1 overflow-y-auto sidebar-body">
        <div className="workspace-inspector-section workspace-inspector-section--preview flex flex-col gap-0" data-testid="workspace-focused-selected-card">
          <div className="workspace-inspector-section__header flex items-center justify-between">
            <div className="min-w-0">
              <h4 className="ui-title-md">{labels.segmentPreview}</h4>
              <p className="workspace-preview-meta ui-text-meta mt-1 truncate">
                #{focusedIdx} · {element.class_name || labels.noElements}
              </p>
            </div>
            <span className="workspace-preview-size ui-text-meta tabular-nums" title={`x · y · w · h: ${bboxSummary}`}>
              {Math.round(element.bbox[2])}×{Math.round(element.bbox[3])}
            </span>
          </div>
          <div className="workspace-inspector-preview flex min-h-[180px] items-center justify-center">
            <canvas
              ref={setDetailCanvasRef}
              data-testid="workspace-detail-crop-canvas"
              width={detailPreviewSize.width}
              height={detailPreviewSize.height}
              className="workspace-inspector-preview__canvas max-h-[180px] max-w-full rounded-none"
            />
          </div>
        </div>

        <div
          className={`${skeletonStyles.owner} workspace-inspector-section workspace-inspector-section--trust flex flex-col gap-3`}
          aria-busy={trustLoading}
          data-testid="workspace-trust-summary"
        >
          <div className="flex items-center justify-between gap-3">
            <h4 className="ui-title-md">{labels.trustSummary}</h4>
            {contextLoading && <Loader2 size={14} className="animate-spin text-[color:var(--text-muted)]" />}
          </div>
          {(trustLoading || trustError) && (
            <div
              className="workspace-trust-loading space-y-3"
              data-testid="workspace-trust-loading"
            >
              <div className="ui-text-eyebrow">
                {labels.initialProposal}
              </div>
              <div className="ui-title-sm truncate">
                {element.class_name}
              </div>
              <div className="workspace-trust-skeleton h-7 w-3/4 rounded-lg" />
              <div className="flex gap-2">
                <div className="workspace-trust-skeleton h-8 flex-1 rounded-lg" />
                <div className="workspace-trust-skeleton h-8 w-20 rounded-lg" />
              </div>
              <div className="space-y-2">
                <div className="workspace-trust-skeleton h-3 w-full rounded-full" />
                <div className="workspace-trust-skeleton h-3 w-5/6 rounded-full" />
                <div className="workspace-trust-skeleton h-3 w-2/3 rounded-full" />
              </div>
            </div>
          )}
          {!trustLoading && !trustError && (
            <>
              <div className="workspace-trust-hero flex items-center justify-between gap-3">
                <span className="min-w-0 truncate text-xl font-semibold text-[var(--text-heading)]">
                  {summaryClass}
                </span>
                <span
                  className={`workspace-trust-score shrink-0 px-2.5 py-1 text-sm font-semibold ${
                    isRejected
                      ? "ui-chip--danger"
                      : "border border-[color:var(--status-ready-border)] bg-[color:var(--status-ready-soft)] text-[color:var(--status-ready-text)]"
                  }`}
                >
                  {(topPrediction * 100).toFixed(1)}%
                </span>
              </div>
              {initialPredictionDiffers && trust && (
                <div className="ui-alert ui-alert--accent p-2 text-xs">
                  <div className="font-semibold">{labels.recalculatedPrediction}</div>
                  <div className="mt-1 opacity-85">
                    {labels.initialProposal}: <span className="font-medium">{element.class_name}</span>{" "}
                    #{trust.predicted_class_rank} · {(trust.predicted_class_similarity * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              {isAmbiguous && !isRejected && (
                <div className="ui-alert ui-alert--accent flex items-start gap-2 p-3">
                  <AlertCircle size={16} className="mt-0.5 shrink-0" />
                  <div className="text-sm">
                    <strong>{labels.ambiguousPrediction}</strong>
                    <p className="mt-1 text-xs opacity-80">
                      {labels.ambiguousDetails} {(margin * 100).toFixed(1)}%. {labels.alternativesExist}
                    </p>
                  </div>
                </div>
              )}
              {isRejected && (
                <div className="ui-alert ui-alert--danger flex items-start gap-2 p-3">
                  <AlertCircle size={16} className="mt-0.5 shrink-0" />
                  <div className="text-sm">
                    <strong>{labels.lowConfidenceFlag}</strong>
                    <p className="mt-1 text-xs opacity-80">
                      {labels.thresholdDetails}
                    </p>
                  </div>
                </div>
              )}
              {trust && (
                <div className="flex gap-2 text-sm">
                  <div className="workspace-trust-stat flex-1 p-2">
                    <span className="ui-text-meta block">{labels.rank}</span>
                    <span className="font-semibold text-[var(--text-body)]">#1</span>
                  </div>
                  <div className="workspace-trust-stat flex-1 p-2">
                    <span className="ui-text-meta block">{labels.margin}</span>
                    <span className={`font-semibold ${isAmbiguous ? "text-[var(--accent)]" : "text-[var(--text-body)]"}`}>
                      {(margin * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {!trustLoading && !trustError && (
          <div className="workspace-inspector-section flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h4 className="ui-title-md">{labels.topPredictions}</h4>
            </div>
            {archetypes.status === "ready" && (
              <div className="ui-text-meta" data-testid="archetype-coverage">
                {labels.archetypeCoverage
                  .replace("{covered}", String(archetypes.covered))
                  .replace("{total}", String(archetypes.total))}
              </div>
            )}
            <div className="space-y-2">
              {topKItems.map((item, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="ui-text-meta w-4 text-right">{i + 1}</span>
                  {archetypes.status === "ready" && (
                    <span
                      className="flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden border border-[color:var(--border-subtle)]"
                      data-archetype-thumb={item.class_name}
                      data-has-asset={archetypes.assets.get(item.class_name) ? "true" : "false"}
                    >
                      {archetypes.assets.get(item.class_name) ? (
                        <img
                          src={adminAnnotationMediaUrl(archetypes.assets.get(item.class_name) as string)}
                          alt=""
                          className="h-full w-full object-contain"
                          loading="lazy"
                        />
                      ) : (
                        <span className="ui-text-meta" aria-hidden="true">
                          —
                        </span>
                      )}
                    </span>
                  )}
                  <div className="flex-1">
                    <div className="mb-0.5 flex justify-between text-sm">
                      <span className={i === 0 ? "font-medium text-[var(--text-body)]" : "text-[color:var(--text-muted)]"}>
                        {item.class_name}
                      </span>
                      <span className={i === 0 ? "font-medium text-[var(--text-soft)]" : "text-[color:var(--text-muted)]"}>
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="ui-progress-track h-1 w-full">
                      <div
                        className={`h-full rounded-full ${i === 0 ? (isRejected ? "ui-progress-value--accent" : "ui-progress-value--ready") : "bg-[var(--surface-hover)]"}`}
                        style={{ width: `${item.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="workspace-inspector-footer app-sidebar__header border-t sidebar-header">
        <button
          type="button"
          onClick={onEditorHandoff}
          className="workspace-annotate-action workspace-annotate-action--full flex w-full items-center justify-center gap-2 px-4 py-2 text-sm transition-all active:scale-[0.99]"
        >
          <Edit3 size={16} />
          {element.rejected ? labels.correctElement : labels.annotateRegion}
        </button>
      </div>
    </div>
  );
}

type DetectedListProps = Pick<
  WorkspaceDetectedPanelProps,
  | "record"
  | "hoveredIdx"
  | "stats"
  | "setCropCanvasRef"
  | "labels"
  | "onEditorHandoff"
  | "onDetectedListKeyDown"
  | "onFocusRegion"
  | "onListRegionEnter"
  | "onListRegionLeave"
>;

function WorkspaceDetectedList({
  record,
  hoveredIdx,
  stats,
  setCropCanvasRef,
  labels,
  onEditorHandoff,
  onDetectedListKeyDown,
  onFocusRegion,
  onListRegionEnter,
  onListRegionLeave,
}: DetectedListProps) {
  return (
    <div className="flex h-full flex-col p-0">
      <div className="workspace-detected-header flex items-start justify-between gap-3 border-b border-[color:var(--border-subtle)]">
        <div>
          <p className="ui-text-eyebrow">
            {labels.proposalPanel}
          </p>
          <h2 className="ui-title-md mt-1">
            {labels.detectedElements}
          </h2>
        </div>
      </div>

      {stats && stats.rejectedCount === stats.total && stats.total > 0 && (
	        <div className="ui-alert ui-alert--accent mb-0 flex items-start gap-3 rounded-none p-3">
	          <Info className="mt-0.5 shrink-0" size={18} />
          <div>
            <p className="ui-text-body-sm">{labels.allRejected}</p>
            <button
              type="button"
              onClick={onEditorHandoff}
	              className="mt-1 text-xs font-medium text-[var(--accent)] hover:text-[var(--accent-hover)]"
            >
              {labels.goToAnnotation}
            </button>
          </div>
        </div>
      )}

      <div
        className="ui-scrollbar workspace-detected-grid grid flex-1 auto-rows-min grid-cols-1 gap-0 overflow-y-auto pr-0"
        data-testid="workspace-detected-list"
        tabIndex={0}
        onKeyDown={onDetectedListKeyDown}
      >
        {record.result.elements.length === 0 ? (
          <div className="ui-empty-state flex h-full flex-col items-center justify-center p-6 text-center">
            <Info className="mb-3 text-[color:var(--text-muted)]" size={32} />
            <p>{labels.noElements}</p>
          </div>
        ) : (
          record.result.elements.map((element, idx) => {
            const hasAnnotation = hasWorkspaceSubmittedAnnotation(record, idx);
            const displayClass = getWorkspaceElementClassName(record, idx);
            const isHovered = hoveredIdx === idx;
            const badgeClasses = element.rejected
              ? "ui-chip--danger"
              : "ui-chip--accent";
            const indexBadgeClasses = element.rejected
              ? "bg-[var(--danger)] text-[var(--danger-on-solid)]"
              : "bg-[var(--accent)] text-[var(--accent-text)]";
            const previewSize = getCropPreviewSize(element.bbox, 48);

            return (
              <div key={idx} className="flex flex-col">
                <div
                  role="button"
                  aria-label={`${displayClass} région ${idx}`}
                  className={`selection-card workspace-detected-row ui-row--hover flex cursor-pointer items-center gap-2.5 rounded-none text-left transition-colors ${isHovered ? "selection-card--active" : ""}`}
                  onClick={() => onFocusRegion(idx)}
                  onMouseEnter={() => onListRegionEnter(idx)}
                  onMouseLeave={() => onListRegionLeave(idx)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onFocusRegion(idx);
                    }
                  }}
                  tabIndex={0}
                >
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-none">
                    <canvas
                      ref={(canvas) => setCropCanvasRef(idx, canvas)}
                      data-testid={`workspace-list-crop-canvas-${idx}`}
                      width={previewSize.width}
                      height={previewSize.height}
                      className="block rounded-none bg-[var(--crop-bg)]"
                    />
                  </div>
                  <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded text-xs font-bold ${indexBadgeClasses}`}>
                    {idx}
                  </span>
                  <span className="ui-text-body-sm flex-1 truncate">
                    {displayClass}
                  </span>
                  {hasAnnotation && <CheckCircle2 size={12} className="shrink-0 text-[color:var(--status-ready-text)]" />}
                  <div className="flex shrink-0 flex-col items-end gap-0.5">
                    <span className={`ui-chip ${badgeClasses} rounded-none px-1.5 py-0.5 leading-none`}>
                      {(element.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
