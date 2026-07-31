import { useCallback, useMemo, useState } from "react";
import {
  createImageBBoxZoomControls,
  ImageBBoxStage,
  useImageStageViewport,
  type ImageBBoxStageBox,
} from "../../components/ImageBBoxStage";
import { StatusPill } from "../../components/ui/AdminPrimitives";
import { adminAnnotationMediaUrl } from "../../services/api";
import type {
  AdminAnnotationAnalysis,
  AdminAnnotationElement,
} from "../../types";
import { DATASET_SPLIT_LABEL, formatBbox } from "./model";

function estimatedImageSize(
  analysis: AdminAnnotationAnalysis,
): [number, number] {
  return analysis.elements.reduce<[number, number]>(
    ([maxX, maxY], element) => [
      Math.max(maxX, element.bbox[0] + element.bbox[2]),
      Math.max(maxY, element.bbox[1] + element.bbox[3]),
    ],
    [1, 1],
  );
}

function AuditValue({ children, title }: { children: string; title?: string }) {
  return (
    <dd className="min-w-0 truncate text-[color:var(--text-main)]" title={title ?? children}>
      {children}
    </dd>
  );
}

export function AdminReviewStage({
  analysis,
  selectedElement,
  onSelectElement,
}: {
  analysis: AdminAnnotationAnalysis;
  selectedElement: AdminAnnotationElement;
  onSelectElement: (element: AdminAnnotationElement) => void;
}) {
  const imageUrl = adminAnnotationMediaUrl(analysis.image_url);
  const fallbackSize = useMemo(() => estimatedImageSize(analysis), [analysis]);
  const [loadedImage, setLoadedImage] = useState<{
    url: string;
    size: [number, number];
  } | null>(null);
  const imageSize = loadedImage?.url === imageUrl ? loadedImage.size : fallbackSize;
  const viewport = useImageStageViewport({
    imageSize,
    disabled: !analysis.image_exists,
    resetKey: analysis.analysis_id,
  });
  const boxes = useMemo<ImageBBoxStageBox[]>(
    () =>
      analysis.elements.map((element) => ({
        id: element.key,
        bbox: [
          element.bbox[0] ?? 0,
          element.bbox[1] ?? 0,
          element.bbox[2] ?? 1,
          element.bbox[3] ?? 1,
        ] as [number, number, number, number],
        label: `#${element.index} ${element.class_name || "Sans nom"}`,
        rejected: element.review_status === "rejected",
        status: element.review_status === "approved" ? "validated" : "draft",
      })),
    [analysis.elements],
  );
  const boxStateById = useMemo(
    () => ({ [selectedElement.key]: { focused: true } }),
    [selectedElement.key],
  );
  const zoomControls = useMemo(
    () =>
      createImageBBoxZoomControls({
        labels: {
          zoomOut: "Dézoomer",
          fitToView: "Ajuster à la vue",
          zoomIn: "Zoomer",
        },
        onZoomOut: viewport.zoomOut,
        onFitToView: viewport.resetView,
        onZoomIn: viewport.zoomIn,
      }),
    [viewport.resetView, viewport.zoomIn, viewport.zoomOut],
  );

  const recordNaturalSize = useCallback(
    (image: HTMLImageElement) => {
      if (image.naturalWidth <= 0 || image.naturalHeight <= 0) return;
      setLoadedImage({
        url: imageUrl,
        size: [image.naturalWidth, image.naturalHeight],
      });
    },
    [imageUrl],
  );

  if (!analysis.image_exists) {
    return <div className="ui-empty-state h-full">Image source manquante</div>;
  }

  return (
    <ImageBBoxStage
      tone="workspace"
      mode="inspect"
      className="h-full min-h-0 w-full"
      headerClassName="admin-review-image-header !gap-2 !px-3 !py-2"
      imageDataUrl={imageUrl}
      imageName={`Image complète ${analysis.analysis_id}`}
      imageSize={imageSize}
      imageFit="fill"
      boxes={boxes}
      selectedId={selectedElement.key}
      showLabelNames
      overlayMode="all"
      viewport={viewport}
      transformSize={viewport.transformSize}
      boxStateById={boxStateById}
      title={
        <h3 className="ui-title-sm max-w-[20rem] truncate" title={analysis.analysis_id}>
          Image complète · {analysis.analysis_id}
        </h3>
      }
      headerMeta={
        <div className="flex flex-wrap items-center gap-1.5">
          <StatusPill>{analysis.elements.length} éléments</StatusPill>
          <StatusPill tone="ready">{Math.round(viewport.zoom * 100)} %</StatusPill>
        </div>
      }
      headerActions={
        <details className="relative">
          <summary className="cursor-pointer list-none whitespace-nowrap border border-[color:var(--border-subtle)] bg-[color:var(--surface-muted)] px-2.5 py-1.5 text-xs font-semibold text-[color:var(--text-heading)] hover:bg-[color:var(--surface-hover)]">
            Détails techniques / audit
          </summary>
          <div className="absolute right-0 top-full z-30 mt-2 w-[min(32rem,calc(100vw-2rem))] border border-[color:var(--border-strong)] bg-[color:var(--surface-elevated)] p-3 shadow-xl">
            <dl className="grid min-w-0 gap-x-4 gap-y-2 text-xs sm:grid-cols-2">
              <div className="min-w-0">
                <dt className="ui-text-caption">Zone [x, y, l, h]</dt>
                <AuditValue>{`[${formatBbox(selectedElement.bbox)}]`}</AuditValue>
              </div>
              <div className="min-w-0">
                <dt className="ui-text-caption">Analyse</dt>
                <AuditValue>{analysis.analysis_id}</AuditValue>
              </div>
              <div className="min-w-0">
                <dt className="ui-text-caption">Split dataset</dt>
                <AuditValue>{`${DATASET_SPLIT_LABEL[selectedElement.dataset_split]} · ${selectedElement.split_reason}`}</AuditValue>
              </div>
              <div className="min-w-0">
                <dt className="ui-text-caption">Empreinte source</dt>
                <AuditValue>{selectedElement.source_fingerprint}</AuditValue>
              </div>
              <div className="min-w-0 sm:col-span-2">
                <dt className="ui-text-caption">Chemin de découpe</dt>
                <AuditValue>{selectedElement.crop_path}</AuditValue>
              </div>
            </dl>
          </div>
        </details>
      }
      panelControls={zoomControls}
      controlsPlacement="bottom-right"
      zoomLabel={`${Math.round(viewport.zoom * 100)} %`}
      onSelectBox={(id) => {
        const element = analysis.elements.find((candidate) => candidate.key === id);
        if (element) onSelectElement(element);
      }}
      renderLabel={(box) => box.label ?? "Sans nom"}
      stageClassName={viewport.zoom > 1 ? (viewport.isPanning ? "cursor-grabbing" : "cursor-grab") : ""}
      stageRef={viewport.containerRef}
      stageProps={{
        onPointerDown: (event) => {
          if (
            event.target instanceof Element &&
            event.target.closest('[data-stage-interactive="true"]')
          ) {
            return;
          }
          viewport.beginPan(event);
        },
        onPointerMove: viewport.movePan,
        onPointerUp: viewport.endPan,
        onPointerCancel: viewport.endPan,
        onWheel: viewport.handleStageWheel,
      }}
      imageProps={{
        alt: `Image complète ${analysis.analysis_id}`,
        onLoad: (event) => recordNaturalSize(event.currentTarget),
      }}
      testIds={{
        root: "admin-review-image-shell",
        header: "admin-review-image-header",
        stage: "admin-review-image-stage",
        transform: "admin-review-image-transform",
        overlay: "admin-review-image-overlay",
        controls: "admin-review-image-controls",
      }}
    />
  );
}

export default AdminReviewStage;
