import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RefCallback } from "react";

export type ImageBBoxStageDisplayRect = {
  left: number;
  top: number;
  width: number;
  height: number;
  scale: number;
};

export type ImageBBoxStageTransformSize = {
  width: number;
  height: number;
};

export type UseImageBBoxStageSizeOptions = {
  imageSize: [number, number] | null | undefined;
  disabled?: boolean;
  loading?: boolean;
};

export type UseImageBBoxStageSizeResult = {
  containerRef: RefCallback<HTMLDivElement>;
  getContainerElement: () => HTMLDivElement | null;
  stageWidth: number;
  stageHeight: number;
  transformSize: ImageBBoxStageTransformSize | null;
  displayRect: ImageBBoxStageDisplayRect | null;
  updateStageSize: () => void;
};

type StageMeasurement = {
  imageWidth: number;
  imageHeight: number;
  transformSize: ImageBBoxStageTransformSize;
  displayRect: ImageBBoxStageDisplayRect;
};

type BoxInsets = {
  paddingLeft: number;
  paddingRight: number;
  paddingTop: number;
  paddingBottom: number;
  borderLeft: number;
  borderRight: number;
  borderTop: number;
  borderBottom: number;
};

function toCssPixel(value: string | null | undefined) {
  const parsed = Number.parseFloat(value ?? "0");
  return Number.isFinite(parsed) ? parsed : 0;
}

function getBoxInsets(element: HTMLElement): BoxInsets {
  const style = window.getComputedStyle(element);

  return {
    paddingLeft: toCssPixel(style.paddingLeft),
    paddingRight: toCssPixel(style.paddingRight),
    paddingTop: toCssPixel(style.paddingTop),
    paddingBottom: toCssPixel(style.paddingBottom),
    borderLeft: toCssPixel(style.borderLeftWidth),
    borderRight: toCssPixel(style.borderRightWidth),
    borderTop: toCssPixel(style.borderTopWidth),
    borderBottom: toCssPixel(style.borderBottomWidth),
  };
}

function sizeForImage(imageSize: UseImageBBoxStageSizeOptions["imageSize"]) {
  const [imageWidth, imageHeight] = imageSize ?? [];
  if (!imageWidth || !imageHeight || imageWidth <= 0 || imageHeight <= 0) {
    return null;
  }

  return { imageWidth, imageHeight };
}

function sameMeasurement(
  current: StageMeasurement | null,
  next: StageMeasurement | null,
) {
  return (
    current?.imageWidth === next?.imageWidth &&
    current?.imageHeight === next?.imageHeight &&
    current?.transformSize.width === next?.transformSize.width &&
    current?.transformSize.height === next?.transformSize.height &&
    current?.displayRect.left === next?.displayRect.left &&
    current?.displayRect.top === next?.displayRect.top &&
    current?.displayRect.width === next?.displayRect.width &&
    current?.displayRect.height === next?.displayRect.height &&
    current?.displayRect.scale === next?.displayRect.scale
  );
}

function getMeasuredStage(
  element: HTMLDivElement | null,
  imageWidth: number,
  imageHeight: number,
): StageMeasurement | null {
  if (!element) {
    return null;
  }

  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }

  const insets = getBoxInsets(element);
  const availableWidth = Math.max(
    0,
    rect.width -
      insets.paddingLeft -
      insets.paddingRight -
      insets.borderLeft -
      insets.borderRight,
  );
  const availableHeight = Math.max(
    0,
    rect.height -
      insets.paddingTop -
      insets.paddingBottom -
      insets.borderTop -
      insets.borderBottom,
  );

  if (availableWidth <= 0 || availableHeight <= 0) {
    return null;
  }

  const scale = Math.min(1, availableWidth / imageWidth, availableHeight / imageHeight);
  const width = Math.max(1, Math.round(imageWidth * scale));
  const height = Math.max(1, Math.round(imageHeight * scale));
  const left = insets.borderLeft + insets.paddingLeft + (availableWidth - width) / 2;
  const top = insets.borderTop + insets.paddingTop + (availableHeight - height) / 2;

  return {
    imageWidth,
    imageHeight,
    transformSize: { width, height },
    displayRect: { left, top, width, height, scale },
  };
}

function naturalFallback(
  imageWidth: number,
  imageHeight: number,
): StageMeasurement {
  return {
    imageWidth,
    imageHeight,
    transformSize: { width: imageWidth, height: imageHeight },
    displayRect: {
      left: 0,
      top: 0,
      width: imageWidth,
      height: imageHeight,
      scale: 1,
    },
  };
}

export function useImageBBoxStageSize({
  imageSize,
  disabled = false,
  loading = false,
}: UseImageBBoxStageSizeOptions): UseImageBBoxStageSizeResult {
  const containerElementRef = useRef<HTMLDivElement | null>(null);
  const [measurement, setMeasurement] = useState<StageMeasurement | null>(null);
  const image = useMemo(
    () => (!disabled && !loading ? sizeForImage(imageSize) : null),
    [disabled, imageSize, loading],
  );
  const containerRef = useCallback<RefCallback<HTMLDivElement>>((node) => {
    containerElementRef.current = node;
  }, []);
  const getContainerElement = useCallback(() => containerElementRef.current, []);

  const updateStageSize = useCallback(() => {
    if (!image) {
      setMeasurement((current) => (current === null ? current : null));
      return;
    }

    const next = getMeasuredStage(
      containerElementRef.current,
      image.imageWidth,
      image.imageHeight,
    );

    setMeasurement((current) => (sameMeasurement(current, next) ? current : next));
  }, [image]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial DOM measurement must synchronize after the stage element is committed
    updateStageSize();

    if (!image) {
      return;
    }

    const element = containerElementRef.current;
    const handleResize = () => updateStageSize();
    window.addEventListener("resize", handleResize);

    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && element) {
      observer = new ResizeObserver(handleResize);
      observer.observe(element);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      observer?.disconnect();
    };
  }, [image, updateStageSize]);

  const activeMeasurement =
    image &&
    measurement?.imageWidth === image.imageWidth &&
    measurement?.imageHeight === image.imageHeight
      ? measurement
      : image
        ? naturalFallback(image.imageWidth, image.imageHeight)
        : null;
  const transformSize = activeMeasurement?.transformSize ?? null;

  return {
    containerRef,
    getContainerElement,
    stageWidth: transformSize?.width ?? 0,
    stageHeight: transformSize?.height ?? 0,
    transformSize,
    displayRect: activeMeasurement?.displayRect ?? null,
    updateStageSize,
  };
}

export default useImageBBoxStageSize;
