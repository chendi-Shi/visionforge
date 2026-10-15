Fixed two-model temporal study comparison

This metadata-only bundle requires both sibling model evidence bundles listed
in comparison.json. Open report.html; inspect raw comparisons and study-plan.json.
Run python -B verify.py to verify both complete model bundles, enforce the study
membership/protocol/model-manifest pins, and recompute all paired comparisons.
No network, inference, HDF5, numeric package or untrusted prediction-source
execution is required. The explicitly shipped verifier code must be reviewed
and trusted before execution. Checksums are consistency checks, not signatures.

Qwen's local manifest bytes must hash to its full frozen model digest and match
the registry config and ordered layer identities pinned before any test call.
MiniCPM's digest is compared directly to the study plan. All model inputs, frame
selections, panels, labels, server version and evaluator source are held fixed.
No methods are chosen or retuned from the comparison. Small-subset task-cluster
intervals are descriptive; conditional MAE must be read with coverage.
