# Model Card — 20260805T112535Z-vicreg-full-data-r2

- Model ID: `codex_classifier`
- Status: `candidate`
- Created: `2026-08-06T03:39:26.316854+00:00`
- Created by: `sina`

## Provenance

- Source: `{"checkpoint_path": "/home/sina/omxpro/clinic-codex/.omc/autoresearch/elements-baseline-replacement/runs/20260805-full-data-r2/results/replica-01/checkpoint.pt", "checkpoint_sha256": "c561dbeed33da1b6bae38ad485c3fb00941c64a5379b21f7d1b7704d56f899fc", "feature_cache_path": "/home/sina/omxpro/clinic-codex/.omc/autoresearch/elements-baseline-replacement/runs/20260805-full-data-r2/cache/full-data-extraction-01.pt", "feature_cache_sha256": "58486292a28b4e0687337d1cd47970256aff2517139bf51c9562ceb4aa7b12c9", "git_commit": "8a024497", "kind": "elements_refit_export", "runtime_config_path": "/home/sina/omxpro/clinic-codex/backend/codex_model/config.json", "runtime_config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b"}`
- Data: `{"class_count": 286, "class_labels_sha256": "63f163ccbfbd9ce2864900e126e442fb5a569dbf12ec52f88be408e3839d126a", "class_order_sha256": "47fe9ac228743f911d69220b2d2dc5e064d6e1941c6b5fda98def60af2ebeaeb", "feature_count": 9990, "runtime_class_labels_sha256": "a49dde1e25e60f12671f4018d8b2691184a59c7de2d2f928f91fe70fb5b3ddd3"}`
- Training: `{"checkpoint_type": "model_state_dict", "hidden_teacher_weight": 0.0, "initialization": "fresh_deterministic", "objective": "vicreg_then_supervised_episodic_full_data_r2_build", "seed": 1706651702, "teacher_assisted": false, "teacher_temperature": null, "teacher_weight": 0.0}`

## Artifacts

| Path | SHA-256 | Size |
| --- | --- | ---: |
| `provenance.json` | `51fb06b706e5bd48f34cdbd20a45b9e0157783ecef3ab9cf32dce848e30dc8fa` | 3102 |
| `runtime/config.json` | `54d5cb607f9a1fa7447ec2b6a929fe00b548341ca8745866b244240e94a11410` | 4921 |
| `runtime/weights/projection.pt` | `c03d85f2163259df86eee8615d27439e326c85a9d075e429a9630cc667288517` | 2758120 |
| `runtime/weights/prototypes.pt` | `e0a096667c5e73d3a003d7986e4d4d84d2728b4a8511defbe08796d2c8f16404` | 155306 |
