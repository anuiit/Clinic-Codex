import type {
  ButtonHTMLAttributes,
  CSSProperties,
  HTMLAttributes,
  ReactNode,
  Ref,
} from "react";
export { AnalyzerToolbar, AnalyzerToolbarButton } from "./AnalyzerToolbar";

export type MainImagePanelTone = "workspace" | "annotation";

export type MainImagePanelTestIds = {
  root?: string;
  header?: string;
  toolbar?: string;
  stage?: string;
  transform?: string;
  controls?: string;
};

export type MainImagePanelControl = {
  id: string;
  icon: ReactNode;
  label: string;
  onClick: NonNullable<ButtonHTMLAttributes<HTMLButtonElement>["onClick"]>;
  disabled?: boolean;
  title?: string;
  pressed?: boolean;
  className?: string;
};

type DivPropsWithTestId = HTMLAttributes<HTMLDivElement> & {
  "data-testid"?: string;
};

export type MainImagePanelToolbarPlacement = "top-left" | "bottom-center";
export type MainImagePanelControlsPlacement = "bottom-right" | "bottom-center";

export type MainImagePanelProps = {
  tone?: MainImagePanelTone;
  title?: ReactNode;
  eyebrow?: ReactNode;
  badges?: ReactNode;
  headerMeta?: ReactNode;
  /**
   * Reserved for non-analyzer metadata actions in the panel header.
   * Workflow/analyzer controls belong in the shared floating `toolbar` slot.
   */
  headerActions?: ReactNode;
  toolbar?: ReactNode;
  toolbarPlacement?: MainImagePanelToolbarPlacement;
  image: ReactNode;
  overlay?: ReactNode;
  controls?: MainImagePanelControl[];
  controlsPlacement?: MainImagePanelControlsPlacement;
  zoomLabel?: ReactNode;
  footer?: ReactNode;
  className?: string;
  headerClassName?: string;
  stageClassName?: string;
  transformClassName?: string;
  transformStyle?: CSSProperties;
  stageProps?: DivPropsWithTestId;
  transformProps?: DivPropsWithTestId;
  stageRef?: Ref<HTMLDivElement>;
  transformRef?: Ref<HTMLDivElement>;
  testIds?: MainImagePanelTestIds;
};

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function MainImagePanel({
  tone = "workspace",
  title,
  eyebrow,
  badges,
  headerMeta,
  headerActions,
  toolbar,
  toolbarPlacement = "top-left",
  image,
  overlay,
  controls,
  controlsPlacement = "bottom-right",
  zoomLabel,
  footer,
  className,
  headerClassName,
  stageClassName,
  transformClassName,
  transformStyle,
  stageProps,
  transformProps,
  stageRef,
  transformRef,
  testIds,
}: MainImagePanelProps) {
  const {
    className: stagePropsClassName,
    "data-testid": stageTestId,
    ...stageRest
  } = stageProps ?? {};
  const {
    className: transformPropsClassName,
    style: transformPropsStyle,
    "data-testid": transformTestId,
    ...transformRest
  } = transformProps ?? {};
  const hasHeader = Boolean(eyebrow || title || badges || headerMeta || headerActions);
  const toolbarPlacementClass =
    toolbarPlacement === "bottom-center"
      ? "bottom-4 left-1/2 -translate-x-1/2"
      : "left-4 top-4";
  const controlsPlacementClass =
    controlsPlacement === "bottom-center"
      ? "bottom-4 left-1/2 -translate-x-1/2"
      : "bottom-4 right-4";

  return (
    <section
      className={cx(
        "main-image-panel",
        `main-image-panel--${tone}`,
        "flex min-h-0 flex-col overflow-hidden rounded-2xl",
        className,
      )}
      data-testid={testIds?.root}
    >
      {hasHeader && (
        <div
          className={cx(
            "main-image-panel__header",
            "flex shrink-0 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between",
            headerClassName,
          )}
          data-testid={testIds?.header}
        >
          <div className="main-image-panel__header-main min-w-0">
            {eyebrow && (
              <div className="main-image-panel__eyebrow truncate text-xs font-semibold uppercase tracking-[0.28em]">
                {eyebrow}
              </div>
            )}
            {(title || badges || headerMeta) && (
              <div className="main-image-panel__title-row flex min-w-0 flex-wrap items-center gap-2 sm:flex-nowrap sm:gap-3">
                {title && (
                  <div className="main-image-panel__title flex min-w-0 items-center gap-3">
                    {title}
                    {badges}
                  </div>
                )}
                {headerMeta && (
                  <div className="main-image-panel__header-meta min-w-0">
                    {headerMeta}
                  </div>
                )}
              </div>
            )}
          </div>
          {headerActions && (
            <div className="main-image-panel__header-actions flex shrink-0 items-center justify-end gap-2">
              {headerActions}
            </div>
          )}
        </div>
      )}

      <div
        ref={stageRef}
        {...stageRest}
        className={cx(
          "main-image-panel__stage",
          "image-stage-frame image-stage-grid image-stage-scrollbar",
          "relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-2xl",
          stageClassName,
          stagePropsClassName,
        )}
        data-testid={testIds?.stage ?? stageTestId}
      >
        {toolbar && (
          <div
            className={cx(
              "main-image-panel__toolbar absolute z-10",
              `main-image-panel__toolbar--${toolbarPlacement}`,
              toolbarPlacementClass,
            )}
            data-testid={testIds?.toolbar}
          >
            {toolbar}
          </div>
        )}

        <div
          ref={transformRef}
          {...transformRest}
          style={{ ...transformStyle, ...transformPropsStyle }}
          className={cx(
            "main-image-panel__transform",
            "relative inline-block",
            transformClassName,
            transformPropsClassName,
          )}
          data-testid={testIds?.transform ?? transformTestId}
        >
          {image}
          {overlay}
        </div>

        {(controls?.length || zoomLabel) && (
          <div
            className={cx(
              "main-image-panel__controls absolute z-10 flex items-center gap-1 rounded-2xl p-1",
              `main-image-panel__controls--${controlsPlacement}`,
              controlsPlacementClass,
            )}
            data-testid={testIds?.controls}
          >
            {controls?.map((control) => (
              <button
                key={control.id}
                type="button"
                onClick={control.onClick}
                disabled={control.disabled}
                aria-label={control.label}
                aria-pressed={control.pressed}
                title={control.title ?? control.label}
                className={cx(
                  "main-image-panel__control rounded-xl p-2 text-[var(--text-soft)] transition-colors hover:bg-[var(--control-bg-hover)] hover:text-[var(--text-main)] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-[var(--text-soft)]",
                  control.className,
                )}
              >
                {control.icon}
              </button>
            ))}
            {zoomLabel && (
              <span className="main-image-panel__zoom-label ui-text-meta px-2 font-semibold tabular-nums">
                {zoomLabel}
              </span>
            )}
          </div>
        )}
      </div>

      {footer && (
        <div className="main-image-panel__footer shrink-0">{footer}</div>
      )}
    </section>
  );
}

export default MainImagePanel;
