import { useState } from "react";
import type { NavigateFunction } from "react-router-dom";
import { t as translate } from "../../i18n/annotation.fr";
import { saveAnnotation } from "../../services/api";
import { updateElements } from "../../services/storage";
import type {
  AnalysisRecord,
  AnnotationStatus,
  DetectedElement,
  SaveAnnotationResult,
} from "../../types";
import { isUnnamedClass } from "../../utils/fuzzyClasses";

type UseAnnotationSubmissionOptions = {
  id: string | undefined;
  record: AnalysisRecord | null;
  elements: DetectedElement[];
  annotationStatus: Record<number, AnnotationStatus>;
  navigate: NavigateFunction;
  labels: {
    submitBlockedUnnamed: string;
    submitBlockedNone: string;
  };
};

export function useAnnotationSubmission({
  id,
  record,
  elements,
  annotationStatus,
  navigate,
  labels,
}: UseAnnotationSubmissionOptions) {
  const [saving, setSaving] = useState(false);
  const [sending, setSending] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  const handleSave = async () => {
    if (!id) return;
    setSaving(true);
    const ok = await updateElements(id, elements, annotationStatus);
    if (!ok) {
      setToast({ msg: translate("save.networkError"), ok: false });
      setSaving(false);
      return;
    }
    setToast({ msg: translate("save.localSuccess"), ok: true });
    setSaving(false);
    setTimeout(() => {
      navigate("/");
    }, 300);
  };

  const handleSendSubmittedForReview = async () => {
    if (!record || !id) return;
    const submittedCandidates = elements
      .map((el, idx) => ({ el, idx }))
      .filter(({ idx }) => annotationStatus[idx] === "validated");
    const unnamedSubmittedIndexes = submittedCandidates
      .map(({ el, idx }) => (isUnnamedClass(el.class_name) ? idx : null))
      .filter((idx): idx is number => idx !== null);
    if (unnamedSubmittedIndexes.length > 0) {
      setToast({
        msg: `${labels.submitBlockedUnnamed} (${unnamedSubmittedIndexes.map((idx) => `#${idx}`).join(", ")})`,
        ok: false,
      });
      setTimeout(() => setToast(null), 4000);
      return;
    }
    const submittedElements = submittedCandidates.filter(
      ({ el }) => !isUnnamedClass(el.class_name),
    );
    if (submittedElements.length === 0) {
      const unnamedIndexes = elements
        .map((el, idx) => (isUnnamedClass(el.class_name) ? idx : null))
        .filter((idx): idx is number => idx !== null);
      setToast({
        msg:
          unnamedIndexes.length > 0
            ? `${labels.submitBlockedUnnamed} (${unnamedIndexes.map((idx) => `#${idx}`).join(", ")})`
            : labels.submitBlockedNone,
        ok: false,
      });
      setTimeout(() => setToast(null), 4000);
      return;
    }

    setSending(true);
    try {
      const persisted = await updateElements(id, elements, annotationStatus);
      if (!persisted) {
        setToast({ msg: translate("save.networkError"), ok: false });
        return;
      }

      const payload = {
        analysis_id: id,
        image_name: record.imageName,
        image_data_url: record.imageDataUrl,
        timestamp: record.timestamp,
        annotations: submittedElements.map(({ el, idx }) => ({
          index: idx,
          bbox: el.bbox,
          class_name: el.class_name,
        })),
      };
      const result: SaveAnnotationResult = await saveAnnotation(payload);

      if (result.ok) {
        setToast({ msg: translate("save.remoteSuccess"), ok: true });
      } else {
        let msg = translate("save.networkError");

        switch (result.error_code) {
          case "VALIDATION_ERROR":
            msg = result.message;
            break;
          case "PERMISSION_DENIED":
            msg = translate("save.permissionDenied");
            break;
          case "DISK_FULL":
            msg = translate("save.diskFull");
            break;
          case "STORAGE_ERROR":
            msg = translate("save.storageError", { message: result.message });
            break;
          case "INTERNAL_ERROR":
            msg = translate("save.internalError", {
              traceId: result.trace_id ?? "?",
            });
            break;
          default:
            msg = translate("save.networkError");
        }

        setToast({ msg, ok: false });
      }
    } finally {
      setSending(false);
      setTimeout(() => setToast(null), 4000);
    }
  };

  return {
    saving,
    sending,
    toast,
    handleSave,
    handleSendSubmittedForReview,
  };
}
