VisionForge: post-hoc four-group sensitivity experiment

This bundle preserves plan.json, cv.json, evaluation.json, and the source files
used during the actual fit. verifier/ is a later, separately
hashed export/verification implementation; runtime/ is the original snapshot.

It depends on the primary robot-benchmark bundle for exact copies of:
source-inputs.jsonl, source-labels.jsonl, features.json, and primary cv.json.
No 510-image duplicate is included, and verification does not require images.

Standalone standard-library verification (Python 3.12+):
python artifacts/robot-sensitivity/verify.py --main-bundle artifacts/robot-benchmark

Repository CLI equivalent:
python -m robotbench.sensitivity --bundle artifacts/robot-sensitivity --main-bundle artifacts/robot-benchmark verify

Verification checks payload and source hashes, planned assignments, exact fold
membership and purging, full prediction coverage, and recalculates every saved
metric and paired interval using the trusted verifier's metrics.py, whose SHA
must match the frozen planned version. It never executes runtime/ source. It does not refit a
classifier or invoke any encoder, model, network, or Ollama service.

This is one post-hoc stress test, not a replacement for the primary result.
Only four unequal clusters exist. Intervals resample fixed out-of-fold outputs;
training folds overlap and model-training variance is not covered. Do not use
these intervals for significance or deployment claims.

bundle-manifest.json hashes every payload file. Its own SHA256 is stored in
bundle-manifest.sha256; those two envelope files are excluded from the payload
map to avoid circular self-hashes. A checksum is not an authenticity signature.
Trust/review the verifier code before executing a downloaded standalone tool;
the repository CLI uses its already installed code and checks the data bundle.
