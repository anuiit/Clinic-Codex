import { AlertCircle, CheckCircle2, ChevronLeft, Edit3, Info, Loader2 } from "lucide-react";
import { useMemo } from "react";

type Element = {
  bbox: [number, number, number, number];
  class_name: string;
  confidence: number;
  rejected: boolean;
  top_k: Array<{ class_name: string; confidence: number }>;
};

export function WorkspaceDetectedPanel({
  currentRecord,
  focusedIdx,
  trustData,
  contextLoading,
  showLabelNames,
  onBack,
  onHandoff,
  onSelectRegion,
  onHoverRegion,
  onLeaveRegion,
  formatCropSize,
  labels,
}: {
  currentRecord: {
    id: string;
    imageName: string;
    imageDataUrl: string;
    result: { elements: Element[]; image_size: [number, number] };
    annotations?: Record<number, string>;
  };
  focusedIdx: number | null;
  trustData: any;
  contextLoading: boolean;
  showLabelNames: boolean;
  onBack: () => void;
  onHandoff: () => void;
  onSelectRegion: (idx: number) => void;
  onHoverRegion: (idx: number) => void;
  onLeaveRegion: (idx: number) => void;
  formatCropSize: (bbox: [number, number, number, number], size: number) => { width: number; height: number };
  labels: any;
}) {
  const stats = useMemo(() => {
    const elements = currentRecord.result.elements;
    const rejectedCount = elements.filter((element) => element.rejected).length;
    const annotatedCount = Object.keys(currentRecord.annotations ?? {}).length;
    const classCounts: Record<string, number> = {};
    elements.forEach((element, idx) => {
      const finalClass = (currentRecord.annotations ?? {})[idx] ?? element.class_name;
      classCounts[finalClass] = (classCounts[finalClass] || 0) + 1;
    });
    const topClasses = Object.entries(classCounts).sort(([, a], [, b]) => b - a).slice(0, 3).map(([className, count]) => `${className} ${count}`);
    const topClass = topClasses[0]?.split(" ")[0] ?? "None";
    return { total: elements.length, rejectedCount, annotatedCount, topClasses, topClass, imageSizeLabel: `${currentRecord.result.image_size[0]}×${currentRecord.result.image_size[1]}` };
  }, [currentRecord]);

  if (focusedIdx !== null) {
    const element = currentRecord.result.elements[focusedIdx];
    const trust = trustData?.trust;
    const summaryClass = trust?.top1_class ?? element.class_name;
    const topPrediction = trust?.top1_similarity ?? element.confidence;
    const runnerUp = trust?.top_k?.[1]?.confidence ?? element.top_k[1]?.confidence ?? 0;
    const margin = trust?.margin_to_second ?? topPrediction - runnerUp;
    const isRejected = trust ? !trust.above_rejection_threshold : element.rejected;
    const isAmbiguous = trust?.ambiguous ?? margin < 0.05;
    const detailPreviewSize = formatCropSize(element.bbox, 180);

    return (
      <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl bg-stone-900/35 sidebar-shell" data-testid="workspace-detected-panel">
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between sidebar-header px-5 py-4 border-b border-stone-800/60">
            <button onClick={onBack} className="flex items-center gap-2 text-stone-400 hover:text-stone-100 transition-colors">
              <ChevronLeft size={16} />
              <span className="text-sm font-medium">{labels.backToRegions}</span>
            </button>
          </div>
          <div className="annotation-scrollbar flex-1 overflow-y-auto sidebar-body p-5 space-y-6">
            <div className="flex flex-col gap-3 p-1">
              <div className="flex items-center justify-between"><h4 className="text-base font-semibold text-stone-300">{labels.segmentPreview}</h4></div>
              <div className="flex min-h-[180px] items-center justify-center rounded-2xl bg-stone-950/55 p-3"><canvas width={detailPreviewSize.width} height={detailPreviewSize.height} className="max-h-[180px] max-w-full rounded-lg bg-stone-800" /></div>
            </div>
            <div className="flex flex-col gap-3 p-1">
              <div className="flex items-center justify-between"><h4 className="text-base font-semibold text-stone-300">{labels.trustSummary}</h4>{contextLoading && <Loader2 size={14} className="animate-spin text-stone-500" />}</div>
              <div className="flex items-center justify-between"><span className="text-2xl font-bold text-stone-100 truncate">{summaryClass}</span><span className={`shrink-0 px-3 py-1.5 rounded-md text-base font-bold ${isRejected ? "bg-red-500/10 text-red-400 border border-red-500/20" : "bg-green-500/10 text-green-400 border border-green-500/20"}`}>{(topPrediction * 100).toFixed(1)}%</span></div>
              {isAmbiguous && !isRejected && <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-amber-400"><AlertCircle size={16} className="shrink-0 mt-0.5" /><div className="text-sm"><strong>{labels.ambiguousPrediction}</strong><p className="mt-1 text-xs opacity-80">{labels.ambiguousDetails} {(margin * 100).toFixed(1)}%. {labels.alternativesExist}</p></div></div>}
              {isRejected && <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-red-400"><AlertCircle size={16} className="shrink-0 mt-0.5" /><div className="text-sm"><strong>{labels.lowConfidenceFlag}</strong><p className="mt-1 text-xs opacity-80">{labels.thresholdDetails}</p></div></div>}
              {trust && <div className="flex gap-2 text-sm"><div className="flex-1 rounded-lg bg-stone-950/50 p-2"><span className="text-stone-500 block">{labels.rank}</span><span className="text-stone-200 font-semibold">#1</span></div><div className="flex-1 rounded-lg bg-stone-950/50 p-2"><span className="text-stone-500 block">{labels.margin}</span><span className={`font-semibold ${isAmbiguous ? "text-amber-400" : "text-stone-200"}`}>{(margin * 100).toFixed(1)}%</span></div></div>}
            </div>
            <div className="flex flex-col gap-3 p-1">
              <div className="flex items-center justify-between"><h4 className="text-base font-semibold text-stone-300">{labels.topPredictions}</h4></div>
              <div className="space-y-2">{(trust?.top_k ?? element.top_k).map((item, i) => <div key={i} className="flex items-center gap-2"><span className="w-4 text-[10px] text-stone-500 text-right">{i + 1}</span><div className="flex-1"><div className="flex justify-between text-sm mb-0.5"><span className={i === 0 ? "text-stone-200 font-medium" : "text-stone-400"}>{item.class_name}</span><span className={i === 0 ? "text-stone-300 font-medium" : "text-stone-500"}>{(item.confidence * 100).toFixed(1)}%</span></div><div className="h-1 w-full overflow-hidden rounded-full bg-stone-800"><div className={`h-full rounded-full ${i === 0 ? (isRejected ? "bg-amber-500" : "bg-green-500") : "bg-stone-600"}`} style={{ width: `${item.confidence * 100}%` }} /></div></div></div>)}</div>
            </div>
          </div>
          <div className="p-5 border-t border-stone-800 sidebar-header"><button type="button" onClick={onHandoff} className="w-full flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-4 py-3 text-sm font-bold text-stone-950 transition-all hover:bg-amber-400 active:scale-[0.98]"><Edit3 size={16} />{element.rejected ? labels.correctElement : labels.annotateRegion}</button></div>
        </div>
      </section>
    );
  }

  return (
    <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl bg-stone-900/35 sidebar-shell" data-testid="workspace-detected-panel">
      <div className="flex min-h-0 flex-col p-4 lg:p-5">
        <div className="mb-3 flex items-start justify-between gap-3 border-b border-stone-800 pb-3">
          <div><p className="text-xs uppercase tracking-[0.24em] text-stone-500">{labels.proposalPanel}</p><h2 className="mt-1 text-base font-semibold text-stone-100">{labels.detectedElements}</h2></div>
          <div className="text-right"><button type="button" onClick={onHandoff} className="inline-flex items-center gap-2 rounded-xl bg-amber-500 px-3 py-2 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400"><Edit3 size={16} /> {labels.annotateRecord}</button></div>
        </div>
        {stats && stats.rejectedCount === stats.total && stats.total > 0 && <div className="mb-4 flex items-start gap-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3"><Info className="mt-0.5 shrink-0 text-amber-500" size={18} /><div><p className="text-sm text-stone-200">{labels.allRejected}</p><button type="button" onClick={onHandoff} className="mt-1 text-xs font-medium text-amber-400 hover:text-amber-300">{labels.goToAnnotation}</button></div></div>}
        {stats && <div className="mb-3 grid grid-cols-2 gap-2 text-xs"><div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2"><span className="block text-stone-500">Image</span><span className="mt-1 block truncate font-semibold text-stone-200" title={currentRecord.imageName}>{currentRecord.imageName}</span></div><div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2"><span className="block text-stone-500">Dimensions</span><span className="mt-1 block font-semibold tabular-nums text-stone-200">{stats.imageSizeLabel}</span></div><div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2"><span className="block text-stone-500">Annotés / rejetés</span><span className="mt-1 block font-semibold tabular-nums text-stone-200">{stats.annotatedCount}/{stats.total} · {stats.rejectedCount}</span></div><div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2"><span className="block text-stone-500">Classes</span><span className="mt-1 block truncate font-semibold text-stone-200" title={stats.topClasses.join(", ")}>{stats.topClasses.join(", ") || stats.topClass}</span></div></div>}
        <div className="annotation-scrollbar workspace-detected-grid grid flex-1 auto-rows-min grid-cols-1 gap-2 overflow-y-auto pr-2" data-testid="workspace-detected-list" tabIndex={0}>
          {currentRecord.result.elements.length === 0 ? <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-dashed border-stone-800 bg-stone-950/50 p-6 text-center"><Info className="mb-3 text-stone-600" size={32} /><p className="text-stone-400">{labels.noElements}</p></div> : currentRecord.result.elements.map((element, idx) => {
            const hasAnnotation = (currentRecord.annotations ?? {})[idx] !== undefined;
            const displayClass = hasAnnotation ? (currentRecord.annotations ?? {})[idx] : element.class_name;
            const badgeClasses = element.rejected ? "bg-red-400/10 text-red-400 border border-red-400/20" : "bg-amber-400/10 text-amber-400 border border-amber-400/20";
            const indexBadgeClasses = element.rejected ? "bg-red-500 text-stone-950" : "bg-amber-500 text-stone-950";
            const [,, boxWidth, boxHeight] = element.bbox;
            let destinationWidth = 48; let destinationHeight = 48;
            if (boxWidth >= boxHeight) destinationHeight = Math.max(1, 48 * (boxHeight / boxWidth));
            else destinationWidth = Math.max(1, 48 * (boxWidth / boxHeight));
            return <div key={idx} className="flex flex-col"><div role="button" aria-label={`${displayClass} région ${idx}`} className="flex cursor-pointer items-center gap-2.5 rounded-xl border px-2.5 py-2 text-left transition-colors border-stone-800 bg-stone-950" onClick={() => onSelectRegion(idx)} onMouseEnter={() => onHoverRegion(idx)} onMouseLeave={onLeaveRegion} tabIndex={0}><div className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-stone-800/50 bg-stone-900"><canvas width={destinationWidth} height={destinationHeight} className="block rounded bg-stone-800" /></div><span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10px] font-bold ${indexBadgeClasses}`}>{idx}</span><span className="flex-1 truncate text-sm text-stone-100">{displayClass}</span>{hasAnnotation && <CheckCircle2 size={12} className="text-green-400 shrink-0" />}<div className="flex shrink-0 flex-col items-end gap-0.5"><span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none ${badgeClasses}`}>{(element.confidence * 100).toFixed(1)}%</span></div></div></div>;
          })}
        </div>
      </div>
    </section>
  );
}
