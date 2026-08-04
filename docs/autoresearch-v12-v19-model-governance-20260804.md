# Autoresearch v12-v19: model governance and terminal result

Date: 2026-08-04
Mission: `elements-baseline-replacement`
Final state: `candidate_built_awaiting_independent_instrument`

## Outcome

The search did not validate a model better than C1 on the available instrument. C1 remains the best replicated research reference under `supported_reference`: fold-local VICReg on a frozen DINOv2-S/14 representation with the canonical episodic readout. Its pooled OOF top-1 improved from 0.05258 to 0.06381 against B0 (delta +0.01123), with favorable McNemar evidence, while remaining ineligible for automatic runtime promotion.

The deployed runtime was not changed. Its projection, prototypes, and config hashes remain:

- `0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210`
- `ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54`
- `f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b`

The final test was not read.

## What v12-v19 established

| Run | Question | Disposition |
| --- | --- | --- |
| v12 | Is an independent evaluation instrument already available? | No qualifying independent instrument was available. |
| v13-v15 | Can archive/member attribution or a shadow source establish independence? | Provenance was made explicit, but no clean independent comparison was established. |
| v16 | Can institutional inventory, endpoint discovery, or metadata provide a usable independent endpoint? | No eligible endpoint or metadata path passed the preregistered gates. |
| v17 | Do decision audit, supervised multi-view training, or hierarchical shrinkage improve C1? | The audit preserved C1 as the supported reference; the tested readouts did not produce a promotable improvement. |
| v18.1 | Does a discriminative cosine head improve C1? | Rejected; pooled top-1 delta was approximately -0.0071. |
| v18.2 | Does a component-contrastive projection improve C1? | Rejected; pooled top-1 delta was approximately -0.0031. |
| v18.3 | Can token-level LoRA plus VICReg be evaluated on the current host? | Retired as infrastructure-not-supported after the sole governed recovery failed during a large drvfs mmap flush. No candidate metric was produced. |
| v19-A | Does frozen DINOv2-B/14 improve the exact v9 VICReg plus episodic-readout mechanism? | Retired as infrastructure-not-supported. The sole authorized extraction failed in the first fold's CUDA pin-memory handoff before a completed feature batch. No candidate metric was produced. |

v11 remains a registered full-data candidate, but it is not promotable: it consumed all 9,990 development rows, the deployed runtime lineage is not independently identified, and no clean local comparison against that runtime is licensed.

## v19 terminal boundary

The v19 contract allowed one phase-2 extraction and explicitly prohibited recovery. The command failed with `CUDA error: out of memory` in the DataLoader pin-memory thread.

The sealed boundary is:

- training images were decoded by worker prefetch, but the exact read count cannot be reconstructed;
- OOF extraction did not start;
- zero feature caches, sidecars, or manifests were produced;
- zero C1 control replays were run;
- zero candidate optimizer steps, predictions, or scores occurred;
- zero runtime writes occurred;
- no retry, batch change, pin-memory change, runner edit, phase 3, or v20 is allowed.

The failure audit is `6cfff7acdce3af4b8ef080bdf21dcf54788379c479a9f0f4322283d77f27a47b`.

Argos Council source turn 43 ratified the terminal classification `retired_infrastructure_not_supported`. The published synthesis is `de277fc45d15e72f25dce6e6ee757bb904011f4a22486d0deab7a51943a77f2a`. Kimi completed the review; Fable failed before inference with an explicit provider API error, so its missing voice was recorded without being treated as a vote.

## Reproducibility

The v12-v19 source and test suite contains prospectively gated runners for instrument discovery, provenance attribution, endpoint validation, model-decision audits, supervised readouts, LoRA feasibility, storage feasibility, and B/14 renomination.

Verification at closure:

- 168 targeted tests passed across 24 new test files;
- 26 new Python runners/auditors compiled successfully;
- v19 runner hash remained `14066b79538a5d3176bdf76f3a6475235da91480faf86a758c328234c6b60219`;
- runtime hashes remained unchanged;
- final test access remained false.

## Next valid step

Further tuning on the same 653-row OOF instrument would increase model-selection overfitting without resolving the evaluation uncertainty. The next valid investment is a genuinely independent instrument, such as expert-segmented cases or an institutional partner dataset. C1 and the registered v11 candidate can then be evaluated against the deployed runtime under a new, prospectively frozen protocol.
