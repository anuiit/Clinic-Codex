import { isUnnamedClass } from "../../utils/fuzzyClasses";

export function formatBboxLabel(
  idx: number,
  className: string,
  showName: boolean,
  unnamedLabel: string,
): string {
  if (!showName) return `#${idx}`;
  const displayName = isUnnamedClass(className) ? unnamedLabel : className;
  return displayName.length > 18 ? `${displayName.slice(0, 17)}…` : displayName;
}
