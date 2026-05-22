import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { deleteAnalysis, getHistory } from "../../services/storage";
import type { AnalysisRecord } from "../../types";

export function resolveCurrentRecord(
  records: AnalysisRecord[],
  preferredId?: string | null,
) {
  if (preferredId) {
    return (
      records.find((record) => record.id === preferredId) ?? records[0] ?? null
    );
  }

  return records[0] ?? null;
}

export function useWorkspaceHistory() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const initialPreferredId = searchParams.get("analysis");
  const initializedRef = useRef(false);

  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [currentRecord, setCurrentRecord] = useState<AnalysisRecord | null>(null);
  const [filter, setFilter] = useState("");
  const [historyOpen, setHistoryOpen] = useState(true);
  const [storageLoading, setStorageLoading] = useState(true);

  useEffect(() => {
    if (location.pathname === "/" && searchParams.get("analysis")) {
      navigate("/", { replace: true });
    }
  }, [location.pathname, navigate, searchParams]);

  const selectRecord = (record: AnalysisRecord | null) => {
    setCurrentRecord(record);
  };

  const syncRecords = useCallback(async (preferredId?: string | null) => {
    const nextRecords = await getHistory();
    setRecords(nextRecords);
    selectRecord(resolveCurrentRecord(nextRecords, preferredId));
    return nextRecords;
  }, []);

  useEffect(() => {
    let active = true;

    getHistory()
      .then((nextRecords) => {
        if (!active) return;
        setRecords(nextRecords);
        setCurrentRecord(resolveCurrentRecord(nextRecords, initialPreferredId));
        if (!initializedRef.current) {
          setHistoryOpen(!resolveCurrentRecord(nextRecords, initialPreferredId));
          initializedRef.current = true;
        }
      })
      .finally(() => {
        if (active) setStorageLoading(false);
      });

    return () => {
      active = false;
    };
  }, [initialPreferredId]);

  const removeRecord = async (id: string) => {
    await deleteAnalysis(id);
    await syncRecords(currentRecord?.id === id ? null : currentRecord?.id);
  };

  const filteredRecords = useMemo(() => {
    if (!filter) {
      return records;
    }

    const query = filter.toLowerCase();
    return records.filter((record) => {
      const names = record.result.elements.map((element) =>
        element.class_name.toLowerCase(),
      );
      const annotated = Object.values(record.annotations ?? {}).map((value) =>
        value.toLowerCase(),
      );
      return (
        names.some((name) => name.includes(query)) ||
        annotated.some((annotation) => annotation.includes(query)) ||
        record.imageName.toLowerCase().includes(query)
      );
    });
  }, [filter, records]);

  const stats = useMemo(() => {
    if (!currentRecord) {
      return null;
    }

    const elements = currentRecord.result.elements;
    const rejectedCount = elements.filter((element) => element.rejected).length;
    const classCounts: Record<string, number> = {};

    elements.forEach((element, idx) => {
      const finalClass =
        (currentRecord.annotations ?? {})[idx] ?? element.class_name;
      classCounts[finalClass] = (classCounts[finalClass] || 0) + 1;
    });

    let topClass = "None";
    let maxCount = 0;
    Object.entries(classCounts).forEach(([className, count]) => {
      if (count > maxCount) {
        maxCount = count;
        topClass = className;
      }
    });

    const annotatedCount = Object.keys(currentRecord.annotations ?? {}).length;
    const topClasses = Object.entries(classCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([className, count]) => `${className} ${count}`);

    return {
      total: elements.length,
      rejectedCount,
      annotatedCount,
      submittedCount: annotatedCount,
      topClass,
      topClasses,
      imageSizeLabel: `${currentRecord.result.image_size[0]}×${currentRecord.result.image_size[1]}`,
    };
  }, [currentRecord]);

  return {
    records,
    currentRecord,
    filter,
    historyOpen,
    filteredRecords,
    stats,
    storageLoading,
    setFilter,
    setHistoryOpen,
    selectRecord,
    syncRecords,
    removeRecord,
  };
}
