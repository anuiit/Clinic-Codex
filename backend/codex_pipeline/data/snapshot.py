"""Resolve approved annotations to the deduplicated rows actually used for fitting."""


def live_annotation_usage(manifest: dict) -> dict:
    rows = manifest.get("rows", [])
    duplicates = manifest.get("duplicates", [])
    conflicts = manifest.get("conflicts", [])
    if not all(isinstance(value, list) for value in (rows, duplicates, conflicts)):
        raise ValueError("snapshot rows, duplicates and conflicts must be lists")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("row_id"), str):
            raise ValueError("snapshot row requires a row_id")
        if row["row_id"] in by_id or row.get("dataset_split") not in {"train", "dev", "locked_test"}:
            raise ValueError("snapshot row has duplicate identity or invalid split")
        by_id[row["row_id"]] = row
    counts = dict.fromkeys(("train", "dev", "locked_test", "excluded"), 0)
    training_ids = set()
    training_classes = set()
    for row in rows + duplicates:
        if not isinstance(row, dict):
            raise ValueError("snapshot row must be an object")
        if row.get("source_kind") != "live_annotation":
            continue
        representative = by_id.get(row.get("duplicate_of") or row.get("row_id"))
        if representative is None or representative.get("class_name") != row.get("class_name"):
            raise ValueError("live annotation has no same-class representative")
        split = representative["dataset_split"]
        counts[split] += 1
        if split == "train":
            training_ids.add(representative["row_id"])
            training_classes.add(representative["class_name"])
    for conflict in conflicts:
        if not isinstance(conflict, dict) or not isinstance(conflict.get("rows"), list):
            raise ValueError("snapshot conflict requires rows")
        counts["excluded"] += sum(row.get("source_kind") == "live_annotation" for row in conflict["rows"])
    return {
        "split_counts": counts,
        "training_row_ids": sorted(training_ids),
        "training_class_names": sorted(training_classes),
    }
