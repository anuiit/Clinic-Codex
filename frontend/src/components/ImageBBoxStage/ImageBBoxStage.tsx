import { MainImagePanel } from "../MainImagePanel";
import { ImageBBoxOverlay } from "./ImageBBoxOverlay";
import type { ImageBBoxStageProps } from "./imageBBoxStage.types";

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function imageFitClassName(imageFit: ImageBBoxStageProps["imageFit"]) {
  return imageFit === "fill"
    ? "block h-full w-full object-fill"
    : "block max-h-full max-w-full object-contain";
}

export function ImageBBoxStage({
  imageDataUrl,
  imageName,
  imageSize,
  boxes,
  selectedId = null,
  hoveredId = null,
  showLabelNames = false,
  mode,
  title,
  eyebrow,
  badges,
  headerMeta,
  headerActions,
  toolbar,
  toolbarPlacement,
  controls,
  panelControls,
  controlsPlacement,
  zoomLabel,
  footer,
  onSelectBox,
  onHoverBox,
  overlayMode = "all",
  tone,
  viewport,
  transformSize,
  transformStyle,
  imageFit = "contain",
  displayBBoxById,
  boxStateById,
  renderLabel,
  renderBoxExtras,
  overlayChildren,
  boxTestIdPrefix,
  stageProps,
  svgProps,
  imageProps,
  className,
  headerClassName,
  stageClassName,
  transformClassName,
  stageRef,
  transformRef,
  testIds,
}: ImageBBoxStageProps) {
  const {
    className: imageClassName,
    alt,
    src,
    draggable,
    ...imageRest
  } = imageProps ?? {};
  const transformWidth = transformSize ? `${transformSize.width}px` : undefined;
  const transformHeight = transformSize ? `${transformSize.height}px` : undefined;

  const controlsSlot = controls ? (
    <div
      className="image-bbox-stage__controls-slot"
      data-testid={testIds?.controls}
    >
      {controls}
    </div>
  ) : null;

  return (
    <MainImagePanel
      tone={tone ?? (mode === "edit" ? "annotation" : "workspace")}
      title={title}
      eyebrow={eyebrow}
      badges={badges}
      headerMeta={headerMeta}
      headerActions={headerActions}
      toolbar={toolbar}
      toolbarPlacement={toolbarPlacement}
      controls={panelControls}
      controlsPlacement={controlsPlacement}
      zoomLabel={zoomLabel}
      footer={
        controlsSlot || footer ? (
          <>
            {controlsSlot}
            {footer}
          </>
        ) : undefined
      }
      className={cx("image-bbox-stage", className)}
      headerClassName={headerClassName}
      stageClassName={cx(
        "image-bbox-stage__stage",
        mode === "edit" ? "image-bbox-stage__stage--edit" : "image-bbox-stage__stage--inspect",
        stageClassName,
      )}
      transformClassName={cx(
        "image-bbox-stage__transform",
        transformSize ? "shrink-0 overflow-hidden rounded-lg" : undefined,
        transformClassName,
      )}
      transformStyle={{
        width: transformWidth,
        height: transformHeight,
        transform: `translate(${viewport.panOffset.x}px, ${viewport.panOffset.y}px) scale(${viewport.zoom})`,
        transformOrigin: "center center",
        transition: viewport.isPanning ? "none" : "transform 0.1s ease",
        willChange: "transform",
        ...transformStyle,
      }}
      stageProps={stageProps}
      stageRef={stageRef}
      transformRef={transformRef}
      testIds={{
        root: testIds?.root,
        header: testIds?.header,
        toolbar: testIds?.toolbar,
        stage: testIds?.stage,
        transform: testIds?.transform,
        controls: panelControls || zoomLabel ? testIds?.controls : undefined,
      }}
      image={
        <img
          {...imageRest}
          src={src ?? imageDataUrl}
          alt={alt ?? imageName}
          draggable={draggable ?? false}
          className={cx(
            "image-bbox-stage__image",
            imageFitClassName(imageFit),
            imageFit === "contain" ? "rounded-lg" : "pointer-events-none",
            imageClassName,
          )}
        />
      }
      overlay={
        <ImageBBoxOverlay
          imageSize={imageSize}
          boxes={boxes}
          selectedId={selectedId}
          hoveredId={hoveredId}
          showLabelNames={showLabelNames}
          overlayMode={overlayMode}
          onSelectBox={onSelectBox}
          onHoverBox={onHoverBox}
          displayBBoxById={displayBBoxById}
          boxStateById={boxStateById}
          renderLabel={renderLabel}
          renderBoxExtras={renderBoxExtras}
          overlayChildren={overlayChildren}
          boxTestIdPrefix={boxTestIdPrefix}
          svgProps={svgProps}
          testId={testIds?.overlay}
        />
      }
    />
  );
}

export default ImageBBoxStage;
