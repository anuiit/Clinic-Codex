import type { ReactNode } from "react";

// reference-art-exempt: fixed paper/ink colors intentionally match the supplied static mock
// and are allowed only in this page-local fallback-art component.
const paper = "bg-[#e4d5bd]";
const tilePaper = "bg-[#ddd0ba]";
const pagePaper = "bg-[#d9c9ad]";
const cropPaper = "bg-[#ded0b7]";
const ink = "text-[#18120d]";
const stage = "bg-[#050507]";
const pageNoise = {
  backgroundImage:
    "radial-gradient(circle at 25% 30%, rgb(0 0 0 / 0.08), transparent 18%), radial-gradient(circle at 65% 70%, rgb(0 0 0 / 0.06), transparent 20%), linear-gradient(90deg, rgb(0 0 0 / 0.045) 1px, transparent 1px), linear-gradient(rgb(0 0 0 / 0.035) 1px, transparent 1px)",
  backgroundSize: "auto, auto, 34px 34px, 34px 34px",
};

function ReferenceGlyph({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 100 180" aria-hidden="true" className={className}>
      <path
        fill="currentColor"
        d="M42 0 76 18 79 63 100 86 80 108 78 155 44 180 20 149 20 110 0 86 20 63 20 22Z"
      />
    </svg>
  );
}

export function ReferenceThumb({ children }: { children?: ReactNode }) {
  const hasRealMedia = Boolean(children);
  return (
    <span
      data-reference-art="thumb"
      data-real-media={hasRealMedia ? "true" : "false"}
      className={`relative flex h-9 w-9 items-center justify-center overflow-hidden border border-[color:var(--border-subtle)] ${hasRealMedia ? "bg-transparent" : `rounded-[0.3rem] ${paper}`}`}
    >
      {!hasRealMedia ? (
        <>
          <span className="sr-only">Aperçu mock de glyphe</span>
          <span className="absolute inset-[0.45rem] text-[#1a1410]" aria-hidden="true">
            <ReferenceGlyph className="h-full w-full" />
          </span>
        </>
      ) : null}
      <span className="relative z-10 h-full w-full [&>img]:h-full [&>img]:w-full [&>img]:object-contain">
        {children}
      </span>
    </span>
  );
}

export function ReferenceTileArt({ children }: { children?: ReactNode }) {
  const hasRealMedia = Boolean(children);
  return (
    <span
      data-reference-art="tile"
      data-real-media={hasRealMedia ? "true" : "false"}
      className={`absolute left-3 right-3 top-3 bottom-8 grid place-items-center overflow-hidden ${hasRealMedia ? "bg-transparent" : `rounded-[0.3rem] ${tilePaper}`}`}
    >
      {!hasRealMedia ? (
        <span className={`absolute inset-[24%_35%] ${ink}`} aria-hidden="true">
          <ReferenceGlyph className="h-full w-full" />
        </span>
      ) : null}
      <span className="relative z-10 h-full w-full [&>img]:h-full [&>img]:w-full [&>img]:object-contain">
        {children}
      </span>
    </span>
  );
}

export function ReferenceDecisionPane({
  type,
  children,
}: {
  type: "context" | "crop";
  children?: ReactNode;
}) {
  const isContext = type === "context";
  const hasRealMedia = Boolean(children);
  return (
    <div
      data-reference-art={type}
      data-real-media={hasRealMedia ? "true" : "false"}
      className={`ui-crop-shell admin-decision-media__${type} relative grid h-full place-items-center overflow-hidden ${hasRealMedia ? "bg-transparent" : stage}`}
    >
      {!hasRealMedia && isContext ? (
        <>
          <div
            aria-hidden="true"
            className={`col-start-1 row-start-1 h-[min(80%,32.5rem)] w-[min(80%,38.75rem)] border border-[rgb(255_255_255_/_0.13)] ${pagePaper} shadow-[0_30px_80px_rgb(0_0_0_/_0.35)]`}
            style={pageNoise}
          />
          <div
            aria-hidden="true"
            className={`col-start-1 row-start-1 h-[6.5rem] w-12 translate-x-32 -translate-y-14 ${ink} shadow-[0_0_0_0.8rem_#d9c9ad,0_0_0_0.95rem_var(--accent),0_0_0_999px_rgb(5_5_7_/_0.26)]`}
          >
            <ReferenceGlyph className="h-full w-full" />
          </div>
        </>
      ) : !hasRealMedia ? (
        <>
          <div
            aria-hidden="true"
            className={`col-start-1 row-start-1 h-[min(76%,26rem)] w-[min(76%,21rem)] rounded-lg ${cropPaper} shadow-[0_30px_80px_rgb(0_0_0_/_0.35)]`}
          />
          <div aria-hidden="true" className={`col-start-1 row-start-1 h-56 w-28 ${ink}`}>
            <ReferenceGlyph className="h-full w-full" />
          </div>
        </>
      ) : null}
      <span className="relative z-10 col-start-1 row-start-1 h-full w-full [&>img]:h-full [&>img]:w-full [&>img]:object-contain">
        {children}
      </span>
    </div>
  );
}
