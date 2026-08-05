# Decision log — v19-R2

| Time (Europe/Paris) | Phase | Decision | Evidence |
|---|---|---|---|
| 2026-08-05T01:24:46+02:00 | Council gate | Create a new terminal R2 instrument; preserve v19-R1 and its consumed claim unchanged. | Published Council SHA-256 e6e262815bd2d52e4a01f760f0d0ded2851cad021420aee7fa1fbe106be8461b. |
| 2026-08-05T01:24:46+02:00 | Root-cause fix | Move model.to(device) immediately before inference_mode in embedding_snapshot; no other scientific change. | Root-cause audit SHA-256 78575d86db4eca59a140c4ee48cddda03c936cf7d4894dd7c8f135a40acc5ff5. |
| 2026-08-05T01:24:46+02:00 | Verification | Require CUDA value-equivalence, zero inference parameters, successful backward, finite gradients, and all historical immutability pins. | 66 combined tests passed; runner 38c0879a9d58424e1ef5d2a139653edfd3bdf09ae1c89f31acfdd800cd8ec90d; test 72622791d719782ec2de70543e079d6ae5411eb88ca36e6b8c0bbd1ad668896f. |
| 2026-08-05T01:24:46+02:00 | Freeze | Authorize exactly one R2-3a followed by one R2-3b, no adaptation, no further instrument after failure. | Freezes 9952d9c0f9380885046a9b86d57fc596ceddb342a912c425dfac942436d5eb79 and e583d3562be4bfb2295b46c13d3c0bd18db52b159bfe65d82856a4d0c035be97. |

| 2026-08-05T01:42:54+02:00 | Phase 3a | Complete the one-shot candidate evaluation without the R1 inference-tensor failure. | 59,220 optimizer steps; 1,959 predictions; candidate-minus-C1 top-1 +0.0112302195; integrity, engagement, B0 anchor, and C1 non-inferiority gates passed; summary 5efcf3b24c6bb633c54a6743bec03adfc43488829451a0a87b17d18535af53b2. |
| 2026-08-05T01:56:19+02:00 | Phase 3b | Reject the official candidate verdict because the preregistered byte-identical checkpoint gate failed. | Prediction rows and 15 model-state hashes were identical, but checkpoint file SHA-256 values, diagnostics, and the normalized audit were not byte-identical; replay audit 17354fa0acaccae432edae3485f9f336edb2cb828fdb984619d0686edfba096c. |
| 2026-08-05T01:59:48+02:00 | Semantic audit | Preserve the distinction between binary serialization failure and scientific reproduction. | All 15 loaded checkpoint payloads matched metadata and tensors exactly; all 30 normalized differences were checkpoint binary-hash fields; terminal audit 3ceba1be75ccf31ee0749e2f5b1cc6304e82c2f731e43be79cbce2924bd12f3b. |
| 2026-08-05T01:59:48+02:00 | Terminal decision | Do not retry, amend, promote, or create another v19 instrument. | Both one-shot claims are consumed; the final test remains unread; runtime artifacts are unchanged; terminal Argos Council is required. |

| 2026-08-05T02:11:51+02:00 | Terminal Council | Ratify `not_issued`, forbid R3 and close the local mission as `candidate_built_awaiting_independent_instrument`. | Published synthesis ae0d75b4bd2834dff677c93081f4375447e613d2edf41275073b0a6e52b48a8c; C1 remains the best validated reference; R2 remains descriptive and non-promotable. |

Both claims have been consumed. R2 is terminal-failed under its frozen replay contract, promotion is forbidden, the final test remains unread, and runtime artifacts are unchanged.
