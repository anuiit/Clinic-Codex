# Decision log — v19 VICReg backbone renomination

## 2026-08-04 — run opened after Council turn 43

- v18.3 was ratified as `retired_infrastructure_not_supported` after the sole recovery-02 failed during a 28.7 GiB drvfs mmap flush. No model metric, candidate state, optimizer step, prediction, final-test read, or runtime write occurred.
- Council synthesis SHA-256 `0379abee5893d619069eae42753dfae80d59682cf66517c7cc1d23452b5f56c0` authorizes exactly one v19-A experiment and no v20.
- The v10 history was corrected prospectively: frozen B/14 capacity was already tested and closed. v19-A is an explicit renomination under the distinct v9 VICReg plus episodic-readout mechanism, not a virgin backbone-scaling branch.
- The only dimension-forced architecture is the shared `ProjectionHead(input_dim=768)`, or `768→768→128`. The old supervised v10 `768→384→128` head is outside scope.
- The feasibility phase is preregistered with zero candidate information. It may verify pins, calculate cache geometry, run deterministic synthetic CUDA forwards at batch sizes 8/16/32, and seal a synthetic mini-cache on ext4. It may not read corpus images, access OOF rows, train, predict, score, read the final test, or write runtime artifacts.
- Passing feasibility returns to Council before candidate contract freeze. Failing feasibility retires v19-A without retry or replacement factor.

## 2026-08-04 — iteration 1 feasibility passed

- Audit SHA-256: `defa12f5cae3662fd12eb75bf75ba580b4d200b97b5b351261aea3b9a206a0e4`; all nine evaluator gates passed.
- The exact cache estimate is 552,407,040 bytes, below the 2 GiB ceiling and structurally free of drvfs and large memmaps.
- CUDA forwards at batch sizes 8, 16, and 32 all passed. Batch 32 retained 77.74% free device memory and is the selected extraction batch.
- The 12,416-byte synthetic mini-cache replayed byte-identically with SHA-256 `60268f1e8398d0fbda5a746026e9d2d88944b9f5da09b923da2c90e87ca9f14f`.
- All candidate-information counters remained zero; final test unread and runtime hashes unchanged.
- The phase stops here for mandatory Council review. No real-source extraction, training, prediction, or score is authorized.


## 2026-08-04 - iteration 2 phase-1 contract frozen

- The thin v19 runner is frozen at SHA-256 `14066b79538a5d3176bdf76f3a6475235da91480faf86a758c328234c6b60219`; it imports the recovered canonical v9 runner unchanged at `e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9`.
- The runner enforces phase-specific Council authorizations bound to the freeze audit, WSL ext4 caches only, no overwrite, exact S/14<->B/14 row order, no OOF views, five cache readback hashes, and exact v9 SSL/episode plan hashes for all 15 foldxseed states.
- The C1 control is deliberately double-gated: a sealed phase-2 replay audit is required, then all 15 C1 checkpoints and 1,959 top-k rows are replayed again in the training process before any candidate gradient.
- Candidate RNG is restored to the canonical v9 post-`ProjectionHead(384,128)` state after the dimension-forced B/14 head initialization. The only allowed initialization divergence is the 768-input head itself.
- Deterministic replay reuses sealed caches without extraction and requires byte-identical prediction rows, checkpoints, candidate states, and normalized audits.
- Eleven runner tests pass; the combined feasibility plus runner suite reports 16/16 passing. The tests are static or synthetic and execute zero image reads, real feature extractions, candidate optimizer steps, predictions, scores, final-test reads, or runtime writes.
- Static validation seals all v9/C1/B/14/runtime inputs. Freeze-audit SHA-256: `46c9ba183af997f4922574701b488961d71b50051a3a6a0d24f45da8a74f1dab`.
- Phase 1 stops here. Phase 2 extraction and C1 control remain forbidden until a new Argos Council synthesis explicitly authorizes them.

## 2026-08-04 - Council turn 45 authorizes iteration 2 phase 2

- Published Argos source turn 42 / local turn 45 synthesis SHA-256: `d14ffe0db4c29c6128b4678106961dc7e2ddf3a6de321136a6e98bb7bec4f9f2`.
- Kimi ratified the frozen contract after reading the full runner and authorized exactly one phase-2 execution.
- Fable failed explicitly before inference because its OAuth token had expired; no retry was attempted and no Fable opinion was fabricated.
- The authorization permits exactly five B/14 caches on WSL ext4, 40,613 source-image reads, a sealed manifest, and exact replay of 15 C1 checkpoints plus 1,959 top-k rows.
- Candidate gradients, predictions, and scores remain forbidden. Recovery or a second extraction attempt is forbidden.
- Phase-2 authorization SHA-256: `ffb8c398cdcf68be118ed369adb6c8f1741eb1cd10e4cdeba4f39f9d9a62f3bf`.
- Phase 2 must return to Council before any candidate gradient.


## 2026-08-04 - iteration 2 phase 2 failed terminally

- The sole authorized `precompute` execution exited 1 during the fold-01 training-row DataLoader pin-memory handoff.
- The exception was `RuntimeError: CUDA error: out of memory` in the pin-memory thread before the first completed feature batch.
- Worker prefetch proves that training source images were decoded, but the exact read count is not recoverable after the exception. OOF extraction never started.
- The cache directory is empty: zero fold caches, zero sidecars, no cache manifest, and no C1 replay audit.
- Candidate optimizer steps, predictions, and scores remain zero. The final test remains unread and runtime hashes are unchanged.
- No retry was performed. The Council-published one-execution/no-recovery rule makes this failure terminal.
- Failure-audit SHA-256: `6cfff7acdce3af4b8ef080bdf21dcf54788379c479a9f0f4322283d77f27a47b`.
- v19-A is retired, phase 3 is forbidden, and the explicit no-v20 terminal condition remains binding.

## 2026-08-04 - Council turn 46 ratifies terminal closure

- Published Argos source turn 43 / local turn 46 synthesis SHA-256: `de277fc45d15e72f25dce6e6ee757bb904011f4a22486d0deab7a51943a77f2a`.
- Kimi ratified the failure audit and the `retired_infrastructure_not_supported` classification after reading all ten untruncated inputs.
- Fable failed before inference with an explicit provider `api_error`; no retry was attempted and the missing voice was not converted into a vote.
- B/14 under v9 VICReg remains unmeasured. No positive, negative, or neutral metric claim is licensed from v19-A.
- Retry, phase 3, v20, runtime promotion, and final-test access are definitively forbidden.
- C1 remains the best validated research reference under `supported_reference`; v11 remains registered but non-promotable.
- The mission closes as `candidate_built_awaiting_independent_instrument`. Further model iteration requires a genuinely independent expert-segmented or institutional evaluation instrument.
