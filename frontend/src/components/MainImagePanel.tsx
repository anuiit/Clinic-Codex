import {
  forwardRef,
  type HTMLAttributes,
  type ReactNode,
} from "react";

type MainImagePanelTone = "workspace" | "annotation";

interface MainImagePanelProps extends HTMLAttributes<HTMLDivElement> {
  tone?: MainImagePanelTone;
  toolbar?: ReactNode;
}

const toneClasses: Record<MainImagePanelTone, string> = {
  workspace: "main-image-panel--workspace",
  annotation: "main-image-panel--annotation",
};

export const MainImagePanel = forwardRef<HTMLDivElement, MainImagePanelProps>(
  function MainImagePanel(
    { tone = "workspace", toolbar, className = "", children, ...stageProps },
    ref,
  ) {
    return (
      <div
        ref={ref}
        {...stageProps}
        data-main-image-panel-tone={tone}
        className={`main-image-panel ${toneClasses[tone]} image-stage-frame image-stage-grid relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-2xl ${className}`.trim()}
      >
        {toolbar}
        {children}
      </div>
    );
  },
);
