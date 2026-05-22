import { useCallback, useRef, useState, type DragEvent } from "react";
import { segmentGlyph } from "../../services/api";
import { saveAnalysis } from "../../services/storage";
import type { AnalysisRecord } from "../../types";

type UseWorkspaceUploadOptions = {
  apiErrorLabel: string;
  syncRecords: (preferredId?: string | null) => Promise<unknown>;
};

export function useWorkspaceUpload({
  apiErrorLabel,
  syncRecords,
}: UseWorkspaceUploadOptions) {
  const inputRef = useRef<HTMLInputElement>(null);
  const currentFileRef = useRef<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback((nextFile: File) => {
    currentFileRef.current = nextFile;
    setFile(nextFile);
    setPreview(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      if (currentFileRef.current === nextFile) {
        setPreview(event.target?.result as string);
      }
    };
    reader.readAsDataURL(nextFile);
  }, []);

  const clearPendingFile = useCallback(() => {
    currentFileRef.current = null;
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const nextFile = event.dataTransfer.files[0];
      if (nextFile) {
        handleFile(nextFile);
      }
    },
    [handleFile],
  );

  const analyze = async () => {
    if (!file || !preview) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await segmentGlyph(file);
      const record: AnalysisRecord = {
        id: crypto.randomUUID(),
        imageName: file.name,
        imageDataUrl: preview,
        timestamp: Date.now(),
        result,
        annotations: {},
      };

      await saveAnalysis(record);
      await syncRecords(record.id);
      clearPendingFile();
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : apiErrorLabel);
    } finally {
      setLoading(false);
    }
  };

  return {
    inputRef,
    dragging,
    file,
    preview,
    loading,
    error,
    setDragging,
    handleFile,
    clearPendingFile,
    onDrop,
    analyze,
  };
}
