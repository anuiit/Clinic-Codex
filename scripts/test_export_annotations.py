from scripts.export_annotations import iter_export_annotations


def _record():
    return {
        "id": "analysis-id",
        "annotations": {},
        "result": {
            "elements": [
                {"bbox": [1, 2, 3, 4], "class_name": "atl"},
                {"bbox": [5, 6, 7, 8], "class_name": "beta"},
                {"bbox": [9, 10, 11, 12], "class_name": "unknown"},
            ]
        },
    }


def test_iter_export_annotations_exports_validated_only_by_default():
    record = _record()
    record["annotationStatus"] = {"0": "validated", "1": "draft", "2": "validated"}

    assert iter_export_annotations(record) == [
        {"index": 0, "bbox": [1, 2, 3, 4], "class_name": "atl"}
    ]


def test_iter_export_annotations_skips_legacy_records_by_default():
    record = _record()
    record["annotations"] = {"0": "corrected"}

    assert iter_export_annotations(record) == []


def test_iter_export_annotations_can_include_legacy_unvalidated_records():
    record = _record()
    record["annotations"] = {"0": "corrected"}

    assert iter_export_annotations(record, include_unvalidated=True) == [
        {"index": 0, "bbox": [1, 2, 3, 4], "class_name": "corrected"}
    ]
