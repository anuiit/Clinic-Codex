import { describe, expect, it } from "vitest";
import type { AdminAnnotationQueue, AdminAnnotationReviewStatus } from "../../types";
import { datasetRows, reviewRows } from "./model";

function element(
  index: number,
  review_status: AdminAnnotationReviewStatus,
  trainable: boolean,
  dataset_split: "train" | "val" | "test" | "excluded" = trainable
    ? "train"
    : "excluded",
) {
  return {
    key: `analysis-1:${index}`,
    analysis_id: "analysis-1",
    index,
    class_name: `class-${index}`,
    bbox: [0, 0, 10, 10],
    crop_path: `/tmp/${index}.png`,
    crop_url: `/crop/${index}`,
    crop_exists: trainable,
    review_status,
    trainable,
    source_fingerprint: `fingerprint-${index}`,
    stale_decision: false,
    dataset_split,
    split_reason: trainable ? "trainable_hash_80_10_10" : "not_trainable",
  };
}

function queue(): AdminAnnotationQueue {
  return {
    status: "ok",
    schema_version: 1,
    local_only: true,
    warning: "local",
    counts: {
      total: 4,
      pending: 1,
      approved: 2,
      rejected: 1,
      trainable: 1,
    },
    analyses: [
      {
        analysis_id: "analysis-1",
        uploaded_at: "2026-06-30T00:00:00Z",
        image_path: "/tmp/image.png",
        image_url: "/image",
        image_exists: true,
        elements: [
          element(0, "pending", false),
          element(1, "approved", true, "train"),
          element(2, "approved", false, "excluded"),
          element(3, "rejected", false),
        ],
      },
    ],
    diagnostics: [],
  };
}

describe("admin annotation model filters", () => {
  it("keeps already-approved elements out of the Trier queue", () => {
    expect(reviewRows(queue()).map((row) => row.element.index)).toEqual([0, 3]);
  });

  it("keeps Dataset limited to validated trainable crops", () => {
    expect(datasetRows(queue()).map((row) => row.element.index)).toEqual([1]);
  });
});
