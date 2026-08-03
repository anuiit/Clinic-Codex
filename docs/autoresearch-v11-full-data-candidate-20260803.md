# Autoresearch v11 full-data candidate build

Date: 2026-08-03

## Result

V11 materialized the supported VICReg research recipe as one runtime-compatible registry candidate. The build passed every preregistered integrity and local-reproducibility gate. It did not read a final test, did not consume the historical v4/v5 holdout, and did not modify the deployed runtime.

This is a **build result, not an efficacy result**. No fresh independent evaluation instrument is operationally identifiable in the repository. The candidate remains `promotion.eligible=false` and requires manual review.

## Candidate

- Registry version: `20260803T202104Z-vicreg-full-data-v11`.
- Architecture: frozen DINOv2-S/14 plus runtime-compatible `ProjectionHead(384,384,128)` and 286 cosine prototypes.
- Authorized training corpus: 9,990 rows, 300 provenance components, 286 classes after the v8 quarantine of 49 rows on 22 conflicting RGB hashes.
- Fresh deterministic seed: `1706649342 = stable_seed("v11-full-data-refit")`.
- Full-data episodic eligibility: 253 classes; label-set hash `07a4f240abb61080afc7a9904861bf541350d1c92e4a08b3363a60ef4f859c0f`.
- State-dict semantic hash: `6a4dbdb04cd8f55ddb15871cb45a2f9468cf8f250d7414c6852a5ad52e8089fa`.
- Prototype-tensor semantic hash: `0aea4030dc78ded51e7f641025f20591fac372c7f1d1a8b92037de4c7f4027dd`.
- Exported projection artifact: `7eec3979e66ec7d1a04db8f412b745ca0cf23ac33cc695b2c2ee298a6a48ff99`.
- Exported prototype artifact: `18b6be6f72e81aaee12c77ed36aff39269ba37fb4738345c6f5be377fb17b137`.

## Reproducibility scope

Two complete refits were run sequentially in the same process on the same machine. They produced byte-identical checkpoint files with SHA-256 `acbf331bbdd1795eda301e1d20a7db546f4949d576e7d08fa6f1890d83dd946c`, identical state tensors, identical prototypes, identical training diagnostics, VICReg plan hash `065d1cbb545cd65060b070e57c5f1383dcfe3f7a639babe84f689afa4ec6fb1b`, and supervised episode-plan hash `ff866e07fc4d1e806abec5d915f1a60bd4556c080ee48412b9be4436e0630d1f`.

This proves deterministic replay within one process/machine configuration. It does not claim independent-process or cross-machine byte identity.

## Diagnostic boundary

The full training-cache fit is top-1 `0.96066`, macro top-1 `0.94716`, and top-3 `0.99580`. These values are labeled `full_training_cache_fit_diagnostic_not_efficacy`. They are training-fit diagnostics and must not be compared with production performance.

The qualified v10/i7 research reference was only top-1 `0.06330` on strict provenance OOF rows, versus B0 `0.05258` (`+0.01072`). That estimate reused the same 653 OOF rows across seven iterations and had only 25 independently evaluable components and 11 evaluable classes. It does not directly measure the v11 full-data artifact or the deployed runtime.

## Different full-data operating point

V11 applies the same C1 recipe to a different training sample and is not the fold-local C1 checkpoint that was measured in v9/v10. Its final VICReg covariance term was `107.56`, versus `3.85–7.25` across the 15 fold-local C1 replays, and its final VICReg loss was `117.11`. The SSL effective rank remained `54.14`, minimum batch standard deviation `0.0539`, final supervised effective rank `63.48`, and final episodic training accuracy `0.9912`.

The build therefore remained numerically healthy, while the order-of-magnitude covariance/loss shift confirms a materially different full-data optimization operating point. Independent evaluation must assess this exact frozen candidate, not infer its behavior from fold-local C1.

## Integrity and runtime boundary

- Spec SHA-256: `64e0e3422bf2504ed2cf59bacf06f4ef52213010ed22a8579f35b5cdd7531b16`.
- Evaluator SHA-256: `d39f0de910ebfd80dd1972d5a9ffb299c7657cb435bb806c84513635231460e5`.
- Full cache SHA-256: `18371f2132ce227d3bcdfdb87fcb81e9bc476520078e44d53e4b65cae32d8022` (local/ignored).
- Deployed runtime before/after: projection `0ad6ce348e…`, prototypes `ffc52f0369…`, config `f2150a97ec…`; all unchanged.
- Registry status: `candidate`; manual review required; promotion ineligible.
- Full v10/v11/registry/promotion verification: `60 passed`.
- Candidate package checksums: all four versioned artifacts passed `sha256sum -c`.

## Active blocker

The mission remains active with:

- phase: `candidate_built`;
- blocker: `independent_evaluation_instrument_absent`;
- promotion comparator: the deployed runtime (`0ad6ce34…` / `ffc52f03…`), not B0;
- promotion: blocked.

A future one-shot evaluation requires newly acquired post-freeze provenance components, disjointness against all 9,990 training rows, labels sealed until candidate/evaluator hashes are frozen, a pre-bound unlabeled manifest, and a definitive no-threshold-adjustment rule after reading. Without that instrument, the runtime must not be promoted.

Council validation synthesis: `584f727af59e759101cf368600062dd0975b40ab9d079ad10641c77971efac57`.
