import pandas as pd

from backend.codex_pipeline.scripts.precompute_embeddings import build_source_groups


def test_build_source_groups_uses_page_coordinates_when_available() -> None:
    metadata = pd.DataFrame(
        [
            {"image_path": "Elements/0001-atl/10_11_12-a.bmp", "codex": "10", "folio": "11", "page": "12"},
            {"image_path": "Elements/0002-cafe/loose.bmp", "codex": None, "folio": None, "page": None},
        ]
    )

    assert build_source_groups(metadata) == [
        "page:10:11:12",
        "image:Elements/0002-cafe/loose.bmp",
    ]
