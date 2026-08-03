# Autoresearch v10 model-improvement result

Date: 2026-08-03

## Result

The run produced a supported research reference under the preregistered iteration-7 evaluator. The reproducible mechanism is the C1 VICReg initialization on frozen DINOv2-S/14 features, not the iteration-7 covariance regularizer.

| Metric | Persisted B0 | i7 candidate | Delta |
| --- | ---: | ---: | ---: |
| Top-1 | 0.05258 | 0.06330 | +0.01072 |
| Top-3 | 0.11690 | 0.14038 | +0.02348 |
| Macro top-1 | 0.06939 | 0.07392 | +0.00453 |

The paired provenance-component bootstrap interval for top-1 is `[+0.00309, +0.02455]`. Two of three fixed seeds are positive. Row-level exact McNemar counts are 48 candidate-only correct and 27 baseline-only correct (`p = 0.02030`); this test is descriptive because it does not cluster dependent rows by provenance component. All preregistered iteration-7 efficacy, noninferiority, stability and integrity gates pass.

No final-test row was read, no runtime artifact changed and the result is not automatically promotable.

Iteration 7 is a prospective procedural replay under a new contract, not statistically independent evidence: it reuses the same 653 unique OOF rows, folds and seeds as v9, and the candidate differs from C1 by one prediction. It strengthens implementation and contract reproducibility but does not create a second sample against sampling noise.

All seven iterations reused the same 1,959 seed-expanded OOF rows. Iteration-level preregistration controls decision leakage, but no fresh independent evaluation artifact is operationally identifiable in the dossier. The historical 270-row v4/v5 holdout was already read once, covers only 80 classes, and has no audited overlap analysis against the v10 corpus; it is therefore neither fresh nor an available sealed final test for this branch.

The gain is concentrated rather than uniform: folds 1 and 4 are positive, fold 2 is negative by 0.03030 top-1, folds 3 and 5 are structurally near-dead, and seed 42 is negative. This heterogeneity is material to any deployment decision.

C1 remains rejected under its original v9 contract. Its slightly higher point estimate must not be retroactively evaluated under i7; only the i7 candidate was evaluated under the i7 contract.

## Covariance interpretation

Iteration 7 compared exact C1 with the same readout plus a post-L2 query covariance penalty of 0.04. The candidate differed from C1 by one top-1 prediction out of 1,959, in C1's favor:

- top-1 versus exact C1: `-0.00051`;
- top-3 versus exact C1: `0.00000`;
- macro top-1 versus exact C1: `-0.00076`;
- weighted covariance penalty / classification loss: `0.00006543` (0.00654%).

The scientifically accurate label is therefore: **prospective procedural replication of the VICReg reference on the same evaluation data; covariance intervention under-dimensioned and not demonstrated**. The coefficient-0.04 branch is closed without claiming that angular covariance regularization was refuted.

## Integrity

- 1,959 paired OOF prediction rows, 5 outer folds and seeds 17/42/73.
- 15 candidate checkpoints.
- Exact persisted-C1 state replay for all 15 pairs.
- Exact persisted-C1 top-k replay for all 1,959 rows.
- Exact historical episode-plan replay for all 15 pairs.
- All five strict cache hashes matched.
- Zero candidate-episode OOF exposure and no non-finite values.
- Runtime projection/prototype hashes unchanged.
- Final test not read and no automatic promotion.

Result hashes retained in the ignored local research artifacts:

- i7 summary: `5a2949262739d4a048cace3d344051084b03e1dccc632d59b309d41fcbc5a899`;
- i7 prediction rows: `7d932c02f166e11472c6825b3dbb0b954aa8c5740ff24f8a7f73f6b53fc691de`;
- i7 runner: `f8b51e9f5473a3a5306432809236680f778ecaaa6be316e17fcfae084d3bc453`;
- published post-i7 Council synthesis: `c95f624813c7da8b4ed55246fadcd4d95d9f7254a4dee8bcfd90deaab588ab00`.

These hashes provide provenance for the local ignored artifacts; a fresh clone cannot verify the uncommitted checkpoints and prediction data from the repository alone.

## Mandatory qualification

The `supported` verdict is relative to the iteration-7 evaluator. Two policies differ from the original v9 contract:

1. the 0.9 effective-rank gate became a descriptive/escalation diagnostic after the preregistered iteration-6 alignment audit;
2. macro top-1 uses a point-estimate noninferiority gate in i7, whereas v9 required a non-negative bootstrap lower bound.

Those policies were frozen before any i7 prediction, so the result is valid under its own contract. They also mean this is not universal promotion readiness. Only 11 classes and 25 provenance components are independently evaluable OOF, leaving macro performance measurement-limited.

## Closed branches

- Raw frozen S/14: inconclusive/neutral against B0.
- Raw B/14: small top-1 gain offset by top-3 and macro losses.
- Learned B/14 projection: +0.00306 top-1 with a bootstrap interval crossing zero.
- Support-aware episodic readout: -0.01174 top-1 against exact C1.
- Effective-rank 0.9 proxy: not aligned with measured harm in its worst fold.
- SSL-update interpolation at alpha 0.5: closed without execution because gain and rank compression were coupled along the same axis and the expected effect was below the current measurement resolution.
- Post-L2 query covariance at 0.04: effectively inert; the Council unanimously rejected a stronger follow-up in this run.

## Promotion boundary

The research reference is better than B0 on the strict OOF top-1/top-3 evidence, but the runtime model has not changed. V11 subsequently materialized the exact full-data VICReg recipe as a build-only candidate; it did not evaluate or promote it. Promotion first requires acquisition and binding of a new independent instrument, followed by a separate preregistered experiment that freezes:

- the exact candidate artifact hashes and evaluator implementation;
- the deployed runtime artifacts as the comparator, not research baseline B0;
- disjointness and provenance coverage relative to every development and historical holdout corpus;
- final-test top-1, top-3, macro, rank and regression thresholds;
- the evaluator generation, especially v9 macro-bootstrap versus i7 macro point-estimate policy, before any final-test observation;
- rollback rules and runtime artifact hashes.
- whether a metadata-only count of independent components can be inspected before committing the one-shot read, without exposing rows or labels;
- an explicit rule that a failed one-shot final evaluation closes the VICReg branch without threshold adjustment or a second read.

Acquiring independently evaluable provenance components is therefore mandatory rather than optional. The coverage audit estimated about 42 components for a 0.01 top-1 interval half-width and about 166 for 0.005 under its descriptive `1/sqrt(n)` approximation.

## Verification

- Full autoresearch v10 suite after consolidation: `40 passed`.
- Python compilation passed for the new runners and tests.
- Iteration 7 completed all 15 pairs and every recorded integrity gate passed.
