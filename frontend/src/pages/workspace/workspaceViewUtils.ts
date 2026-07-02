export type WorkspaceOverlayMode = "all" | "focused" | "hidden";
export type WorkspaceHoverSource = "image" | "list" | null;

export function formatWorkspaceBboxLabel(
  idx: number,
  className: string,
  showName: boolean,
) {
  if (!showName) return `#${idx}`;
  return className.length > 18 ? `${className.slice(0, 17)}…` : className;
}

type WorkspaceRecordLabelSource = {
  result: {
    elements: Array<{
      class_name: string;
    }>;
  };
  annotations?: Record<number, string>;
  annotationStatus?: Record<number, "draft" | "validated">;
};

export function hasWorkspaceSubmittedAnnotation(
  record: WorkspaceRecordLabelSource,
  idx: number,
) {
  return (
    record.annotationStatus?.[idx] === "validated" ||
    (!(idx in (record.annotationStatus ?? {})) &&
      (record.annotations ?? {})[idx] !== undefined)
  );
}

export function getWorkspaceElementClassName(
  record: WorkspaceRecordLabelSource,
  idx: number,
) {
  const elementClassName = record.result.elements[idx]?.class_name ?? "";
  if (idx in (record.annotationStatus ?? {})) {
    return elementClassName;
  }

  return (record.annotations ?? {})[idx] ?? elementClassName;
}

export function getCropPreviewSize(
  bbox: [number, number, number, number],
  maxSize: number,
) {
  const [, , rawBoxWidth, rawBoxHeight] = bbox;
  const boxWidth = Number.isFinite(rawBoxWidth) ? Math.max(0, rawBoxWidth) : 0;
  const boxHeight = Number.isFinite(rawBoxHeight) ? Math.max(0, rawBoxHeight) : 0;
  const minVisibleSize = Math.min(maxSize, maxSize >= 120 ? 44 : 18);

  if (boxWidth <= 0 || boxHeight <= 0) {
    return { width: maxSize, height: maxSize };
  }

  let width = maxSize;
  let height = maxSize;

  if (boxWidth >= boxHeight) {
    height = Math.max(minVisibleSize, maxSize * (boxHeight / boxWidth));
  } else {
    width = Math.max(minVisibleSize, maxSize * (boxWidth / boxHeight));
  }

  return { width: Math.round(width), height: Math.round(height) };
}
