Exploratory instruction-conditioned action-effect kernel experiment

Both 16-base-taskvar and the previously frozen four-coarse-group protocols are
reported for all four methods. This follows prior inspection of all 140 rows.
It is not independent confirmation, a trajectory benchmark, or a new VLM.

plan.json was saved before fitting. runtime/ preserves the actual source.
cv-*.json include every inner split, validation prediction, chosen C, kernel
bandwidth and outer prediction. evaluation.json retains every fixed comparison
and the same-observation/different-instruction diagnostic. Legacy reference
predictions are copied byte-for-byte under references/.

No images or feature matrix are duplicated. Source inputs, labels and features
are the exact files from artifacts/robot-benchmark/source-inputs.jsonl,
source-labels.jsonl and features.json; expected SHA256 values are in plan.json
and the manifest. Source dataset/encoder licenses remain in the main bundle.

Repository reproduction (with the declared packages and prepared source data):
python -m robotbench.relational_run --output data/robot-relational-new plan
python -m robotbench.relational_run --output data/robot-relational-new run
python -m robotbench.relational_run --output data/robot-relational-new --bundle artifacts/robot-relational-new export

The frozen runtime snapshot must be used to reproduce this exact code revision;
current code changes are rejected when resuming an existing plan. Inspect and
trust Python code before executing it. Checksums are not authenticity signatures.

Bootstrap intervals resample fixed OOF outputs and do not include overlapping
fold training variance. Four unequal coarse groups permit only exploratory
intervals. Identical-image counterfactual pairs are dependent observations.
