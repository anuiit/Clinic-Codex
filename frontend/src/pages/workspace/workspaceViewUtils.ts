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
