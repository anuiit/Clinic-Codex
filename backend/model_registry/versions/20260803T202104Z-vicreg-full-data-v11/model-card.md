# Model Card — 20260803T202104Z-vicreg-full-data-v11

- Model ID: `codex_classifier`
- Status: `candidate`
- Created: `2026-08-03T20:43:54.764582+00:00`
- Created by: `sina`

## Provenance

- Source: `{"checkpoint_path": "/home/sina/omxpro/clinic-codex/.omc/autoresearch/elements-baseline-replacement/runs/20260803-full-data-v11/results/replica-01/checkpoint.pt", "checkpoint_sha256": "acbf331bbdd1795eda301e1d20a7db546f4949d576e7d08fa6f1890d83dd946c", "feature_cache_path": "/home/sina/omxpro/clinic-codex/.omc/autoresearch/elements-baseline-replacement/runs/20260803-full-data-v11/cache/full-data-views08.pt", "feature_cache_sha256": "18371f2132ce227d3bcdfdb87fcb81e9bc476520078e44d53e4b65cae32d8022", "git_commit": "c402ee1e", "kind": "elements_refit_export", "runtime_config_path": "/home/sina/omxpro/clinic-codex/backend/codex_model/config.json", "runtime_config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b"}`
- Data: `{"class_count": 286, "class_labels_sha256": "63f163ccbfbd9ce2864900e126e442fb5a569dbf12ec52f88be408e3839d126a", "class_order_sha256": "47fe9ac228743f911d69220b2d2dc5e064d6e1941c6b5fda98def60af2ebeaeb", "feature_count": 9990, "runtime_class_labels_sha256": "a49dde1e25e60f12671f4018d8b2691184a59c7de2d2f928f91fe70fb5b3ddd3"}`
- Training: `{"checkpoint_type": "model_state_dict", "hidden_teacher_weight": 0.0, "initialization": "fresh_deterministic", "objective": "vicreg_then_supervised_episodic_full_data_build", "seed": 1706649342, "teacher_assisted": false, "teacher_temperature": null, "teacher_weight": 0.0}`

## Artifacts

| Path | SHA-256 | Size |
| --- | --- | ---: |
| `provenance.json` | `e520ae6ad84b630a755074d904b5eddb5fbf363fe46449a68a810ea53e42ff2f` | 2951 |
| `runtime/config.json` | `3df3ef5e1a32edf88bfc9899268dcc035a190788bbc4935a65733a078a8fdf60` | 4857 |
| `runtime/weights/projection.pt` | `7eec3979e66ec7d1a04db8f412b745ca0cf23ac33cc695b2c2ee298a6a48ff99` | 790504 |
| `runtime/weights/prototypes.pt` | `18b6be6f72e81aaee12c77ed36aff39269ba37fb4738345c6f5be377fb17b137` | 155306 |

## Evaluation boundary

This is a build-only candidate, not an efficacy result. It was reproduced by two complete sequential refits in the same process and on the same machine; independent-process and cross-machine determinism remain untested.

- Full-training-cache fit diagnostic: top-1 `0.96066`, macro top-1 `0.94716`, top-3 `0.99580`. Its metric context is `full_training_cache_fit_diagnostic_not_efficacy`.
- Strict OOF research reference from v10 iteration 7: top-1 `0.06330` versus B0 `0.05258` (`+0.01072`). It reused the same 653 OOF rows across iterations, covered only 25 independently evaluable provenance components and 11 classes, and does not evaluate this exact v11 artifact or the deployed runtime comparator.
- Full-data SSL final covariance was `107.5573`, versus `3.8541` to `7.2511` in the fold-local C1 replays. The full-data refit therefore reached a materially different operating point from the measured fold-local research candidate.
- Promotion is blocked until a new independent evaluation instrument compares this exact candidate against the deployed runtime artifacts under a frozen evaluator. B0 is not a valid promotion comparator.
