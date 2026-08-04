# Decision log — discriminative readout v18

## Iteration 0001 — class-balanced cosine head

- Factor: prototype-initialized, class-balanced normalized cosine head on the exact C1 embeddings.
- Integrity and engagement gates passed; canonical replay was byte-identical.
- Result: `not_supported`, with top-1 delta versus C1 of `-0.007146503318019394`.
- Decision: do not promote; close the head-only branch.

## Iteration 0002 — component-aware contrastive projection

- Factor: component-aware supervised-contrastive adaptation of the exact C1 `net.3` weight and bias.
- Integrity and engagement gates passed; all 19 replay artifacts were byte-identical.
- Result: `not_supported`, with top-1 delta versus C1 of `-0.003062787136294022`.
- Decision: do not promote; close the projection-only branch.
- Council: Fable recommended stopping; Codex authorized one final, conditional backbone-participation factor because it uses all fold-train rows. Kimi returned `outcome_unknown` and was not retried.

## Iteration 0003 — LoRA-VICReg feasibility gate

- Scope: feasibility only. Candidate feature extraction, training, prediction, runtime writes, and final-test reads remained forbidden.
- Candidate contract under audit: rank-8, alpha-8, dropout-0 LoRA on all four linear layers of the final DINOv2-S/14 block, with the base backbone frozen.
- Result: `pass=true`.
- Exact LoRA trainable parameter count: `49152`.
- Synthetic CUDA batch-256 forward/backward peak: `5444311040` bytes on an RTX 3070.
- Estimated float32 pre-final-block token cache: `142226382336` bytes; available disk exceeds the required 25% margin.
- All 9,990 retained source paths and decoded RGB hashes matched the frozen corpus; all five canonical cache metadata and fold-isolation checks passed.
- Zero-LoRA output was byte-identical; base parameters had no gradients and remained byte-identical after the synthetic optimizer step.
- Candidate operation counts stayed zero; runtime hashes stayed unchanged; final test remained unread.
- Audit SHA-256: `519d3f78c66373a24c3d89e645e207f843c519fc17ef02ba8f6e85e036b410a6`.
- Decision: submit the feasibility milestone to Argos Council before freezing any v18.3 spec, evaluator, or candidate cache manifest.

### Council after the feasibility gate

- Argos source turn 35 completed with both Fable and Kimi available; all seven selected files were included and none skipped.
- Both partners consent to the single v18.3 execution while retaining their historical preference for stopping.
- The Council requires a re-executed C1 arm on the shared token path, isolated LoRA RNG, explicit AdamW/clipping groups, train/OOF physical separation, no batch-dependent running state, and cache reuse for replay.
- Codex rejects Fable's proposed float32 end-to-end policy because the pinned v9 code proves C1 uses a mixed path: backbone under CUDA autocast float16, block-boundary/output tensors float32, persisted features float16, then VICReg/readout float32.
- A synthetic probe confirmed the pre-last-block tokens and outputs are float32 and that `block 11 → norm → CLS` reproduces the full backbone output byte-for-byte under the same autocast and batch shape.
- Published Council synthesis SHA-256: `1c64340186e3e21a3b448fec60958528479c494d8959ed4977fc047992599b52`.


## Iteration 0003 — executable contract freeze

- The spec, evaluator, and shared-token-cache schema remain frozen at SHA-256 `775f809a8b3b641bc1058be34e54be0416306962f3bd956165cc105a0f2d4e87`, `761d14fd26be86fcd9cee24e50a3897fb3b552ec27b9b6615e7d695655dca180`, and `f687656cc15d8e6a0bebff7cc8e704c77f6279d85137e5f3196e38fe74cdbffc`.
- The executable runner and replay auditor are frozen at SHA-256 `557a99dc8a7c8f402a4d9644c0c3d76c5fefeefa8ed9094865508986274121e1` and `135952e53b783f5aaed7c6fdb3d9cbb8840072af901e0dfd2e7cd06e848f78d5`.
- Twelve targeted tests pass. A real CUDA preflight validated all 15 fold-seed LoRA boundaries with one optimizer-manifest hash, one unchanged base-backbone hash, zero optimizer steps, and zero OOF access.
- All 15 exact C1 state/plan replays and all 15 LoRA preflights are ordered before the first candidate optimizer step. All 15 candidate trainings complete before any OOF cache is created; all five OOF entries are then hash-sealed before candidate interpretation.
- Deterministic replay reuses the sealed caches and compares 15 candidate-state binaries plus summary, evaluation, prediction rows, and audit byte-for-byte under an identity normalization contract.
- Candidate extraction, training, prediction, final-test reads, and runtime writes remain zero. Runtime hashes remain unchanged.
- Contract-freeze audit SHA-256: `c905b09a3ab4d98f3734448341a1e6407d2fa2d9456ac1caa7ac6277e022c26a`.
- Decision: submit this executable contract to Argos Council before the 132.5 GiB train-token extraction.

### Council after the executable contract freeze

- Argos source turn 36 completed with Fable and Kimi both `ok`; all nine selected context files were included, none skipped, and no prompt was truncated.
- Both partners validated the causal factor, the 14,220-step arithmetic, the control/preflight ordering, physical OOF separation, RNG isolation, backbone freeze, and byte-identical replay design.
- Both partners found one blocking evidence defect: three manually transcribed contract hashes in the local Council evidence did not match the authoritative freeze audit, and one was not a valid-length SHA-256.
- The evidence was regenerated from the final artifact octets. The three artifact hashes, the freeze audit, and the runner constants now match exactly; `validate-contract` passes again with all candidate counters at zero, the final test unread, and no runtime write.
- The frozen spec, evaluator, cache schema, runner, and replay auditor were not modified by this evidence correction.
- Published Council synthesis SHA-256: `d60c3d7ed7b7fe84e076e5fb3b020fd2160b58285ba19afc1e0431c8f70964f4`.
- Decision: authorize `extract-train` only. Candidate training, OOF extraction, prediction, final-test access, and runtime writes remain forbidden until the next explicit gate.

## Iteration 0003 — failed train-token extraction

- The authorized `extract-train` command exited nonzero after 1,344.6 seconds with two DataLoader worker bus errors.
- WSL then failed even `execvpe(/bin/true)` with an I/O error and required a full WSL shutdown/restart; ext4 recovery was observed at remount.
- Folds 1–3 completed and were re-read and SHA-256 hashed after recovery. Their feature-equivalence counts are 72,819, 73,251, and 72,189.
- Fold 4 contains only an untrusted preallocated `.partial` file; fold 5 was not started. No train manifest or final token manifest exists.
- No OOF token, candidate state, candidate optimizer step, candidate prediction, final-test read, or runtime write occurred. Runtime hashes remain unchanged.
- Post-recovery `/dev/shm` is 8,354,869,248 bytes, so the generic PyTorch message does not by itself establish shared-memory exhaustion; the subsequent WSL-wide I/O failure is the stronger infrastructure signal.
- Failure audit SHA-256: `83dd5848dc57ebedbd62cc541da03f75893267a54c6054664969dc115540e028`.
- Decision: freeze all deletion/retry/resume and submit the infrastructure failure to Argos Council. The strict no-retry contract remains in force unless a new prospective synthesis explicitly permits recovery.

### Council after the extraction infrastructure failure

- Argos source turn 37 completed with Fable and Kimi both `ok`; all seven selected context files were included, none skipped, and no prompt was truncated.
- All three voices classify the attempt as scientifically null because no score, candidate state, OOF feature, optimizer step, or prediction was observed.
- The Council authorizes one full re-extraction with the unchanged runner into a new empty `iteration-0003-token-caches-recovery-01` directory. The failed directory remains preserved and is never reused.
- Recovery folds 1–3 will be compared against all preserved token, metadata, control-feature, and equivalence hashes. A mismatch pauses acceptance and returns to Council.
- Codex would retire v18.3 after another identical failure. Fable and Kimi would permit one formally re-frozen `num_workers=0` infrastructure amendment; this synthesis does not authorize that amendment automatically.
- Published Council synthesis SHA-256: `040212f449a5bc8e449131d8f1f719f0ac9e6bf708a554422f33fe9c689bf219`.
- Recovery preregistration SHA-256: `5ac2303a1bcf67309910bec5edb534da213576df8cb5ac8471fc4a897a06a978`.
- Decision: run recovery-01 once, stop before candidate training, and submit either its sealed manifest or its frozen failure to the next Council gate.

## Iteration 0003 — recovery-01 host-disk exhaustion

- The preregistered unchanged-runner recovery exited nonzero after 36.3 seconds, before any fold completed.
- A DataLoader worker failed in `Image.open` with `OSError: [Errno 5]`; Python cleanup then raised `OSError: [Errno 30] Read-only file system`. WSL became unresponsive and required shutdown/restart.
- The definitive infrastructure boundary was the Windows C: volume hosting `ext4.vhdx`: only 39,317,504 physical bytes were free at diagnosis, despite roughly 820 GB of sparse virtual ext4 capacity reported inside Linux.
- The prior `df /` feasibility check was therefore invalid for the actual backing-store capacity. The evidence does not support a `num_workers` or `/dev/shm` cause.
- Only an untrusted 570,425,344-byte fold-1 `.partial` exists in recovery-01. No sealed manifest, OOF token, candidate state, optimizer step, prediction, final-test read, or runtime write exists.
- Regenerable NVIDIA DXCache and pip-cache entries were cleared to restore 6.8 GB on C:; ext4 recovery completed and `/` is read-write again.
- F: has 215,109,963,776 bytes available versus 177,782,977,920 required by the preregistered 25% cache margin, but no F: path was created and no retry was started.
- Recovery failure audit SHA-256: `4be32360b48579062c02b8f79586f1647755937feff3b3403801d2b31308fc41`.
- Decision: freeze all model operations and submit this new root-cause evidence to Argos Council before deciding whether a storage-only recovery on F: is permissible.

### Council after recovery-01 root-cause analysis

- Argos source turn 38 completed with Fable and Kimi both `ok`; all seven selected files were included, none skipped, and no prompt was truncated.
- All voices now treat both failures as manifestations of the same unmeasured physical C: exhaustion and authorize one final storage-only recovery on F:.
- The runner, contracts, model factor, seeds, GPU, data and `num_workers=2` remain frozen; both partners explicitly close the former `num_workers=0` option.
- Recovery-02 requires at least 177,782,977,920 host bytes free on F:, 10 GiB host bytes free on C:, a read-write WSL root, an absent output path and unchanged scientific counters/hashes.
- A separate 2 GiB non-scientific drvfs probe must verify physical allocation, byte-identical mmap reread, sequential write >= 50 MiB/s and pseudo-random 512 KiB mmap reads >= 50 MiB/s.
- Any probe failure, new infrastructure failure, internal equivalence failure or corruption of the new cache retires v18.3; there is no recovery-03.
- A mismatch only against the post-crash folds 1–3 suspends acceptance and returns to Council without candidate operations because Fable disputes whether recovered old files are a valid terminal reference.
- Published Council synthesis SHA-256: `c0c4f86c8cbd2a44a92246c522760e35629a427aa3aef0cb10802c1eceb1e10f`.
- Decision: satisfy and preregister the storage gates, run the probe, and only then permit the single recovery-02 extraction. Candidate training remains forbidden.

## Iteration 0003 — recovery-02 storage probe

- The official `uv cache clean` removed only regenerable package cache entries and raised physical C: free space above the Council floor; 31,094,767,616 bytes remained after the probe.
- Recovery-02 was preregistered before either F: path existed; preregistration SHA-256: `ba8f1e8e7782ebe02e6324a5cfc4eaeb1423731bf89345754e8bc4d615a7b4ec`.
- The 2 GiB probe passed all five frozen gates with 100% physical allocation and a byte-identical sequential mmap hash.
- Observed write throughput was 222,342,752 bytes/s and pseudo-random 512 KiB mmap throughput was 1,640,791,262 bytes/s, both above 50 MiB/s.
- The probe file was removed and the recovery-02 output path remains absent. F: free space remains 215,109,963,776 bytes.
- Dataset/model access, candidate steps/predictions, final-test reads and runtime writes remained zero. Probe audit SHA-256: `8d0cfa4857a98b45d6b8bc1889cde1f89ebe694b160c0132e6cc6f9541cef5cd`.
- Limitation: the random timing followed a full read of the same 2 GiB file and may therefore measure warm page cache rather than a 142 GB working set.
- Decision: keep `extract-train` forbidden and submit the passing audit plus this limitation to Council before creating the recovery output path.

### Council after the recovery-02 storage probe

- Argos source turn 39 completed with Fable and Kimi both `ok`; all seven selected files were included, none skipped, and no prompt was truncated.
- All voices authorize the single recovery-02 extraction now. The probe validated the allocation, mmap integrity and sequential-write profile needed by this phase.
- No additional cold-cache probe is allowed before extraction; adding one after observing the passing probe would reopen the prospective contract without causal need.
- Immediately before launch, C:/F: host floors, WSL `rw`, path absence, contract/runtime hashes and zero scientific counters must be revalidated.
- The exact preregistered command, unchanged runner and `num_workers=2` remain mandatory. Any new extraction failure retires v18.3; no recovery-03 exists.
- Success still stops before candidate training after five folds, 359,640 equivalences and fold 1–3 hash comparison.
- The warm-cache limitation moves to a read-only benchmark on the real fold-1 token cache before candidate training, with 512 KiB blocks, seed 20260804 and a 50 MiB/s reference.
- Fable treats sub-threshold speed as operational; Kimi treats it as terminal. The synthesis precommits to suspension and Council review without candidate operations if it is below threshold.
- Published Council synthesis SHA-256: `dc6ffc3ae71edeabdfe4e34f9e5a9531d028427bd841ad729362ae74cdb50407`.

## Iteration 0003 — terminal recovery-02 extraction failure

- The single authorized command exited 1 after 344.9 seconds during fold-1 `numpy.memmap.flush()` with `OSError: [Errno 12] Cannot allocate memory`.
- Recovery-02 contains only an untrusted 28,745,446,016-byte fold-1 `.partial`; no fold, train manifest or final token manifest was sealed.
- Post-failure WSL still had 15,778,222,080 memory bytes available and 4,294,758,400 swap bytes free, supporting a large drvfs mmap/flush limitation rather than ordinary residual RAM exhaustion.
- No OOF token, candidate state, optimizer step, prediction, model metric, final-test read or runtime write occurred. Runtime hashes remain unchanged.
- The passing 2 GiB probe did not expose flush behavior for the 28.7 GB mapping; this realizes its documented smaller-than-RAM limitation.
- The prospectively published policy retires v18.3 after any new extraction failure, with no recovery-03 and no `num_workers=0` amendment.
- Failure audit SHA-256: `fbf611de7f821f54c49517588df3f9434b8d42ce4cd64dccaf6b01b136d0c808`.
- Decision: retire v18.3 without a scientific result, freeze all caches, and return to Council to choose a materially different cache-light model branch while keeping the overall mission active.
