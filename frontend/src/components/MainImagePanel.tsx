import type {
  ButtonHTMLAttributes,
  CSSProperties,
  HTMLAttributes,
  ReactNode,
  Ref,
} from "react";

export type MainImagePanelTone = "workspace" | "annotation";

export type MainImagePanelTestIds = {
  root?: string;
  header?: string;
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

export type MainImagePanelProps = {
  tone?: MainImagePanelTone;
  title?: ReactNode;
  eyebrow?: ReactNode;
  badges?: ReactNode;
  /**
   * Reserved for non-analyzer metadata actions in the panel header.
   * Workflow/analyzer controls belong in the shared top-left `toolbar` slot.
   */
  headerActions?: ReactNode;
  toolbar?: ReactNode;
  image: ReactNode;
  overlay?: ReactNode;
  controls?: MainImagePanelControl[];
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

export type AnalyzerToolbarProps = HTMLAttributes<HTMLDivElement>;

export function AnalyzerToolbar({
  className,
  children,
  ...props
}: AnalyzerToolbarProps) {
  return (
    <div
      {...props}
      className={cx("analyzer-toolbar flex items-center gap-1 rounded-2xl p-1", className)}
    >
      {children}
    </div>
  );
}

export type AnalyzerToolbarButtonProps =
  ButtonHTMLAttributes<HTMLButtonElement> & {
    active?: boolean;
  };

export function AnalyzerToolbarButton({
  active,
  className,
  type = "button",
  children,
  ...props
}: AnalyzerToolbarButtonProps) {
  return (
    <button
      {...props}
      type={type}
      className={cx(
        "analyzer-toolbar__button rounded-xl px-3 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        active
          ? "bg-amber-400 text-stone-950 shadow-lg shadow-amber-950/30"
          : "text-stone-300 hover:bg-stone-800 hover:text-stone-50 disabled:hover:bg-transparent disabled:hover:text-stone-300",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function MainImagePanel({
  tone = "workspace",
  title,
  eyebrow,
  badges,
  toolbar,
  image,
  overlay,
  controls,
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
  const hasHeader = Boolean(eyebrow || title || badges);

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
          <div className="min-w-0">
            {eyebrow && (
              <div className="main-image-panel__eyebrow truncate text-[10px] font-semibold uppercase tracking-[0.28em]">
                {eyebrow}
              </div>
            )}
            {title && (
              <div className="main-image-panel__title flex min-w-0 items-center gap-3">
                {title}
                {badges}
              </div>
            )}
          </div>
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
          <div className="main-image-panel__toolbar absolute left-4 top-4 z-10">
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
            className="main-image-panel__controls absolute bottom-4 right-4 z-10 flex items-center gap-1 rounded-2xl p-1"
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
                  "main-image-panel__control rounded-xl p-2 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-stone-300",
                  control.className,
                )}
              >
                {control.icon}
              </button>
            ))}
            {zoomLabel && (
              <span className="main-image-panel__zoom-label px-2 text-xs font-semibold tabular-nums text-stone-400">
                {zoomLabel}
              </span>
            )}
          </div>
        )}
      </div>

      {footer && (
        <div className="main-image-panel__footer shrink-0">
          {footer}
        </div>
      )}
    </section>
  );
}

export default MainImagePanel;
