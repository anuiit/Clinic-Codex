import type { HTMLAttributes } from "react";

type ImageBBoxToolbarProps = HTMLAttributes<HTMLDivElement> & {
  label?: string;
};

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function ImageBBoxToolbar({
  children,
  className,
  label = "Image bounding-box controls",
  role = "toolbar",
  ...props
}: ImageBBoxToolbarProps) {
  return (
    <div
      aria-label={label}
      role={role}
      {...props}
      className={cx(
        "image-bbox-toolbar flex items-center gap-1 rounded-2xl p-1",
        className,
      )}
    >
      {children}
    </div>
  );
}

export default ImageBBoxToolbar;
