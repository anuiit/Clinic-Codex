import type { KeyboardEvent, MutableRefObject, RefObject } from "react";
import { AlertCircle, CheckCircle2, ChevronLeft, Edit3, Info, Loader2 } from "lucide-react";
import type { AnalysisRecord, TrustResult } from "../../types";
import { getCropPreviewSize } from "./workspaceViewUtils";

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
  trustData: TrustResult | null;
  contextLoading: boolean;
  cropCanvasRefs: MutableRefObject<(HTMLCanvasElement | null)[]>;
  detailCanvasRef: RefObject<HTMLCanvasElement | null>;
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
  trustData,
  contextLoading,
  cropCanvasRefs,
  detailCanvasRef,
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
      className="flex min-h-0 flex-col overflow-hidden rounded-2xl bg-stone-900/35 sidebar-shell"
      data-testid="workspace-detected-panel"
    >
      {focusedIdx !== null ? (
        <WorkspaceFocusedRegionPanel
          record={record}
          focusedIdx={focusedIdx}
          trustData={trustData}
          contextLoading={contextLoading}
          detailCanvasRef={detailCanvasRef}
          labels={labels}
          onBackToRegions={onBackToRegions}
          onEditorHandoff={onEditorHandoff}
        />
      ) : (
        <WorkspaceDetectedList
          record={record}
          hoveredIdx={hoveredIdx}
          stats={stats}
          cropCanvasRefs={cropCanvasRefs}
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
  | "trustData"
  | "contextLoading"
  | "detailCanvasRef"
  | "labels"
  | "onBackToRegions"
  | "onEditorHandoff"
> & {
  focusedIdx: number;
};

function WorkspaceFocusedRegionPanel({
  record,
  focusedIdx,
  trustData,
  contextLoading,
  detailCanvasRef,
  labels,
  onBackToRegions,
  onEditorHandoff,
}: FocusedPanelProps) {
  const element = record.result.elements[focusedIdx];
  const trust = trustData?.trust;
  const summaryClass = trust?.top1_class ?? element.class_name;
  const topPrediction = trust?.top1_similarity ?? element.confidence;
  const runnerUp = trust?.top_k?.[1]?.confidence ?? element.top_k[1]?.confidence ?? 0;
  const margin = trust?.margin_to_second ?? topPrediction - runnerUp;
  const isRejected = trust ? !trust.above_rejection_threshold : element.rejected;
  const isAmbiguous = trust?.ambiguous ?? margin < 0.05;
  const detailPreviewSize = getCropPreviewSize(element.bbox, 180);
  const initialPredictionDiffers = Boolean(trust && trust.top1_class !== element.class_name);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between sidebar-header px-5 py-4 border-b border-stone-800/60">
        <button
          onClick={onBackToRegions}
          className="flex items-center gap-2 text-stone-400 hover:text-stone-100 transition-colors"
        >
          <ChevronLeft size={16} />
          <span className="text-sm font-medium">{labels.backToRegions}</span>
        </button>
      </div>

      <div className="annotation-scrollbar flex-1 overflow-y-auto sidebar-body p-5 space-y-6">
        <div className="flex flex-col gap-3 p-1">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold text-stone-300">
              {labels.segmentPreview}
            </h4>
          </div>
          <div className="flex min-h-[180px] items-center justify-center rounded-2xl bg-stone-950/55 p-3">
            <canvas
              ref={detailCanvasRef}
              width={detailPreviewSize.width}
              height={detailPreviewSize.height}
              className="max-h-[180px] max-w-full rounded-lg bg-stone-800"
            />
          </div>
        </div>

        <div className="flex flex-col gap-3 p-1">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold text-stone-300">
              {labels.trustSummary}
            </h4>
            {contextLoading && <Loader2 size={14} className="animate-spin text-stone-500" />}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold text-stone-100 truncate">
              {summaryClass}
            </span>
            <span
              className={`shrink-0 px-3 py-1.5 rounded-md text-base font-bold ${
                isRejected
                  ? "bg-red-500/10 text-red-400 border border-red-500/20"
                  : "bg-green-500/10 text-green-400 border border-green-500/20"
              }`}
            >
              {(topPrediction * 100).toFixed(1)}%
            </span>
          </div>
          {initialPredictionDiffers && trust && (
            <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-2 text-xs text-amber-300">
              <div className="font-semibold">{labels.recalculatedPrediction}</div>
              <div className="mt-1 text-amber-200/80">
                {labels.initialProposal}: <span className="font-medium text-amber-200">{element.class_name}</span>{" "}
                #{trust.predicted_class_rank} · {(trust.predicted_class_similarity * 100).toFixed(1)}%
              </div>
            </div>
          )}
          {isAmbiguous && !isRejected && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-amber-400">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <div className="text-sm">
                <strong>{labels.ambiguousPrediction}</strong>
                <p className="mt-1 text-xs opacity-80">
                  {labels.ambiguousDetails} {(margin * 100).toFixed(1)}%. {labels.alternativesExist}
                </p>
              </div>
            </div>
          )}
          {isRejected && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-red-400">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <div className="text-sm">
                <strong>{labels.lowConfidenceFlag}</strong>
                <p className="mt-1 text-xs opacity-80">{labels.thresholdDetails}</p>
              </div>
            </div>
          )}
          {trust && (
            <div className="flex gap-2 text-sm">
              <div className="flex-1 rounded-lg bg-stone-950/50 p-2">
                <span className="text-stone-500 block">{labels.rank}</span>
                <span className="text-stone-200 font-semibold">#1</span>
              </div>
              <div className="flex-1 rounded-lg bg-stone-950/50 p-2">
                <span className="text-stone-500 block">{labels.margin}</span>
                <span className={`font-semibold ${isAmbiguous ? "text-amber-400" : "text-stone-200"}`}>
                  {(margin * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3 p-1">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold text-stone-300">
              {labels.topPredictions}
            </h4>
          </div>
          <div className="space-y-2">
            {(trust?.top_k ?? element.top_k).map((item, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-4 text-[10px] text-stone-500 text-right">{i + 1}</span>
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-0.5">
                    <span className={i === 0 ? "text-stone-200 font-medium" : "text-stone-400"}>
                      {item.class_name}
                    </span>
                    <span className={i === 0 ? "text-stone-300 font-medium" : "text-stone-500"}>
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1 w-full overflow-hidden rounded-full bg-stone-800">
                    <div
                      className={`h-full rounded-full ${i === 0 ? (isRejected ? "bg-amber-500" : "bg-green-500") : "bg-stone-600"}`}
                      style={{ width: `${item.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="p-5 border-t border-stone-800 sidebar-header">
        <button
          type="button"
          onClick={onEditorHandoff}
          className="w-full flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-4 py-3 text-sm font-bold text-stone-950 transition-all hover:bg-amber-400 active:scale-[0.98]"
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
  | "cropCanvasRefs"
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
  cropCanvasRefs,
  labels,
  onEditorHandoff,
  onDetectedListKeyDown,
  onFocusRegion,
  onListRegionEnter,
  onListRegionLeave,
}: DetectedListProps) {
  return (
    <div className="flex h-full flex-col p-4 lg:p-5">
      <div className="mb-3 flex items-start justify-between gap-3 border-b border-stone-800 pb-3">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-stone-500">
            {labels.proposalPanel}
          </p>
          <h2 className="mt-1 text-base font-semibold text-stone-100">
            {labels.detectedElements}
          </h2>
        </div>
        <div className="text-right">
          <button
            type="button"
            onClick={onEditorHandoff}
            className="inline-flex items-center gap-2 rounded-xl bg-amber-500 px-3 py-2 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400"
          >
            <Edit3 size={16} /> {labels.annotateRecord}
          </button>
        </div>
      </div>

      {stats && stats.rejectedCount === stats.total && stats.total > 0 && (
        <div className="mb-4 flex items-start gap-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3">
          <Info className="mt-0.5 shrink-0 text-amber-500" size={18} />
          <div>
            <p className="text-sm text-stone-200">{labels.allRejected}</p>
            <button
              type="button"
              onClick={onEditorHandoff}
              className="mt-1 text-xs font-medium text-amber-400 hover:text-amber-300"
            >
              {labels.goToAnnotation}
            </button>
          </div>
        </div>
      )}

      {stats && (
        <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
          <StatCell label="Image" value={record.imageName} title={record.imageName} />
          <StatCell label="Dimensions" value={stats.imageSizeLabel} />
          <StatCell label="Annotés / rejetés" value={`${stats.annotatedCount}/${stats.total} · ${stats.rejectedCount}`} />
          <StatCell
            label="Classes"
            value={stats.topClasses.join(", ") || stats.topClass}
            title={stats.topClasses.join(", ")}
          />
        </div>
      )}

      <div
        className="annotation-scrollbar workspace-detected-grid grid flex-1 auto-rows-min grid-cols-1 gap-2 overflow-y-auto pr-2"
        data-testid="workspace-detected-list"
        tabIndex={0}
        onKeyDown={onDetectedListKeyDown}
      >
        {record.result.elements.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-dashed border-stone-800 bg-stone-950/50 p-6 text-center">
            <Info className="mb-3 text-stone-600" size={32} />
            <p className="text-stone-400">{labels.noElements}</p>
          </div>
        ) : (
          record.result.elements.map((element, idx) => {
            const hasAnnotation = (record.annotations ?? {})[idx] !== undefined;
            const displayClass = hasAnnotation
              ? (record.annotations ?? {})[idx]
              : element.class_name;
            const isHovered = hoveredIdx === idx;
            const badgeClasses = element.rejected
              ? "bg-red-400/10 text-red-400 border border-red-400/20"
              : "bg-amber-400/10 text-amber-400 border border-amber-400/20";
            const indexBadgeClasses = element.rejected
              ? "bg-red-500 text-stone-950"
              : "bg-amber-500 text-stone-950";
            const previewSize = getCropPreviewSize(element.bbox, 48);

            return (
              <div key={idx} className="flex flex-col">
                <div
                  role="button"
                  aria-label={`${displayClass} région ${idx}`}
                  className={`flex cursor-pointer items-center gap-2.5 rounded-xl border px-2.5 py-2 text-left transition-colors ${
                    isHovered
                      ? "border-stone-700 bg-stone-900"
                      : "border-stone-800 bg-stone-950"
                  }`}
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
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-stone-800/50 bg-stone-900">
                    <canvas
                      ref={(canvas) => {
                        cropCanvasRefs.current[idx] = canvas;
                      }}
                      width={previewSize.width}
                      height={previewSize.height}
                      className="block rounded bg-stone-800"
                    />
                  </div>
                  <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10px] font-bold ${indexBadgeClasses}`}>
                    {idx}
                  </span>
                  <span className="flex-1 truncate text-sm text-stone-100">
                    {displayClass}
                  </span>
                  {hasAnnotation && <CheckCircle2 size={12} className="text-green-400 shrink-0" />}
                  <div className="flex shrink-0 flex-col items-end gap-0.5">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none ${badgeClasses}`}>
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

function StatCell({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2">
      <span className="block text-stone-500">{label}</span>
      <span
        className="mt-1 block truncate font-semibold text-stone-200"
        title={title}
      >
        {value}
      </span>
    </div>
  );
}
