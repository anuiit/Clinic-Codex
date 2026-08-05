# Mission — authority-amended B/14 retry v19-R1

Continue the single elements-baseline-replacement mission after v19 failed for infrastructure reasons before producing one feature batch or consuming any OOF prediction, score, or learned candidate information.

The retry is a distinct, one-shot run. It may change only the extraction execution geometry: fixed row batch 4, num_workers=0, unchanged pin-memory behavior, unchanged DINOv2-B/14 weights, unchanged eight-view transform plan, unchanged VICReg/readout/evaluator, and unchanged leakage controls.

Iteration 1 is limited to an isomorphic train-only pipeline smoke: exactly 12 fold-1 training rows, three complete batches, 36 images per forward. It may not read outer-fold source images, persist features, optimize a candidate, predict, score, write runtime weights, or read the final test.

Any smoke failure retires v19-R1 without fallback. A passing smoke still authorizes no extraction: the persistent Argos Council must review its sealed audit before phase 2 can be frozen.

Even if phase 2 is later authorized, v19-R1 has one real extraction/evaluation life only, cannot be promoted automatically, and retires the local 653-row OOF instrument when complete.
