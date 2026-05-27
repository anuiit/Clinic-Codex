import type {
  CSSProperties,
  HTMLAttributes,
  ImgHTMLAttributes,
  ReactNode,
  Ref,
  SVGProps,
} from "react";
import type {
  MainImagePanelControlsPlacement,
  MainImagePanelControl,
  MainImagePanelTone,
  MainImagePanelToolbarPlacement,
} from "../MainImagePanel";

export type ImageBBoxStageBoxId = string | number;
export type ImageBBox = [number, number, number, number];

export type ImageBBoxStageBox = {
  id: ImageBBoxStageBoxId;
  bbox: ImageBBox;
  label?: string;
  confidence?: number;
  rejected?: boolean;
  status?: "draft" | "validated";
};

export type ImageBBoxBoxState = {
  focused?: boolean;
  imageHovered?: boolean;
  listHovered?: boolean;
  submitted?: boolean;
  rejected?: boolean;
};

export type ImageBBoxStageViewport = {
  zoom: number;
  panOffset: { x: number; y: number };
  isPanning?: boolean;
};

export type ImageBBoxStageTestIds = {
  root?: string;
  header?: string;
  toolbar?: string;
  stage?: string;
  transform?: string;
  overlay?: string;
  controls?: string;
};

type ImgPropsWithTestId = ImgHTMLAttributes<HTMLImageElement> & {
  "data-testid"?: string;
  ref?: Ref<HTMLImageElement>;
};

type DivPropsWithTestId = HTMLAttributes<HTMLDivElement> & {
  "data-testid"?: string;
};

export type ImageBBoxStageBoxRenderState = ImageBBoxBoxState & {
  focused: boolean;
  imageHovered: boolean;
  listHovered: boolean;
  submitted: boolean;
  rejected: boolean;
};

export type ImageBBoxOverlayMode = "all" | "focused" | "hidden";

export type ImageBBoxOverlayProps = {
  imageSize: [number, number];
  boxes: ImageBBoxStageBox[];
  selectedId?: ImageBBoxStageBoxId | null;
  hoveredId?: ImageBBoxStageBoxId | null;
  showLabelNames?: boolean;
  overlayMode?: ImageBBoxOverlayMode;
  onSelectBox?: (id: ImageBBoxStageBoxId | null) => void;
  onHoverBox?: (id: ImageBBoxStageBoxId | null) => void;
  displayBBoxById?: Partial<Record<ImageBBoxStageBoxId, ImageBBox>>;
  boxStateById?: Partial<Record<ImageBBoxStageBoxId, ImageBBoxBoxState>>;
  renderLabel?: (
    box: ImageBBoxStageBox,
    state: ImageBBoxStageBoxRenderState,
  ) => ReactNode;
  renderBoxExtras?: (
    box: ImageBBoxStageBox,
    state: ImageBBoxStageBoxRenderState,
    bbox: ImageBBox,
  ) => ReactNode;
  overlayChildren?: ReactNode;
  boxTestIdPrefix?: string;
  svgProps?: SVGProps<SVGSVGElement>;
  testId?: string;
};

export type ImageBBoxStageProps = {
  imageDataUrl: string;
  imageName: string;
  imageSize: [number, number];
  boxes: ImageBBoxStageBox[];
  selectedId?: ImageBBoxStageBoxId | null;
  hoveredId?: ImageBBoxStageBoxId | null;
  showLabelNames?: boolean;
  mode: "inspect" | "edit";

  title?: ReactNode;
  eyebrow?: ReactNode;
  badges?: ReactNode;
  headerMeta?: ReactNode;
  headerActions?: ReactNode;
  toolbar?: ReactNode;
  toolbarPlacement?: MainImagePanelToolbarPlacement;
  controls?: ReactNode;
  panelControls?: MainImagePanelControl[];
  controlsPlacement?: MainImagePanelControlsPlacement;
  zoomLabel?: ReactNode;
  footer?: ReactNode;

  onSelectBox?: (id: ImageBBoxStageBoxId | null) => void;
  onHoverBox?: (id: ImageBBoxStageBoxId | null) => void;

  overlayMode?: ImageBBoxOverlayMode;
  tone?: MainImagePanelTone;
  viewport: ImageBBoxStageViewport;
  transformSize?: { width: number; height: number } | null;
  transformStyle?: CSSProperties;
  imageFit?: "contain" | "fill";
  displayBBoxById?: Partial<Record<ImageBBoxStageBoxId, ImageBBox>>;
  boxStateById?: Partial<Record<ImageBBoxStageBoxId, ImageBBoxBoxState>>;
  renderLabel?: (
    box: ImageBBoxStageBox,
    state: ImageBBoxStageBoxRenderState,
  ) => ReactNode;
  renderBoxExtras?: (
    box: ImageBBoxStageBox,
    state: ImageBBoxStageBoxRenderState,
    bbox: ImageBBox,
  ) => ReactNode;
  overlayChildren?: ReactNode;
  boxTestIdPrefix?: string;
  stageProps?: DivPropsWithTestId;
  svgProps?: SVGProps<SVGSVGElement>;
  imageProps?: ImgPropsWithTestId;
  className?: string;
  headerClassName?: string;
  stageClassName?: string;
  transformClassName?: string;
  stageRef?: Ref<HTMLDivElement>;
  transformRef?: Ref<HTMLDivElement>;
  testIds?: ImageBBoxStageTestIds;
};
