import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import type { MainImagePanelControl } from "../MainImagePanel";

export type ImageBBoxZoomControlLabels = {
  zoomOut: string;
  fitToView: string;
  zoomIn: string;
};

export type CreateImageBBoxZoomControlsOptions = {
  labels: ImageBBoxZoomControlLabels;
  onZoomOut: () => void;
  onFitToView: () => void;
  onZoomIn: () => void;
};

export function createImageBBoxZoomControls({
  labels,
  onZoomOut,
  onFitToView,
  onZoomIn,
}: CreateImageBBoxZoomControlsOptions): MainImagePanelControl[] {
  return [
    {
      id: "zoom-out",
      label: labels.zoomOut,
      title: labels.zoomOut,
      onClick: onZoomOut,
      icon: <ZoomOut size={16} />,
    },
    {
      id: "fit-to-view",
      label: labels.fitToView,
      title: labels.fitToView,
      onClick: onFitToView,
      icon: <Maximize2 size={16} />,
    },
    {
      id: "zoom-in",
      label: labels.zoomIn,
      title: labels.zoomIn,
      onClick: onZoomIn,
      icon: <ZoomIn size={16} />,
    },
  ];
}
