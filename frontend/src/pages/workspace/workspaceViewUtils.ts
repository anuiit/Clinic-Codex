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
  const [, , boxWidth, boxHeight] = bbox;
  let width = maxSize;
  let height = maxSize;

  if (boxWidth >= boxHeight) {
    height = Math.max(1, maxSize * (boxHeight / boxWidth));
  } else {
    width = Math.max(1, maxSize * (boxWidth / boxHeight));
  }

  return { width, height };
}
