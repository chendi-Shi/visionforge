VisionForge temporal failure evidence bundle

Open report.html for offline review. Panel cells seek the corresponding video
frame. Videos are compressed review media; panels are exact model input bytes.
Measured timestamps are used for every prediction and metric. Video playback
coordinates use source frame index / encoded video fps only for navigation.

Run: python -B verify.py
Repository equivalent: python -B -m robotbench.temporal_report verify --bundle PATH

The verifier requires only Python's standard library. It checks every manifest
file, frozen source/annotation snapshots, all 16 x 2 raw records and input-panel
hashes, reconstructs author labels and model verdicts, and recomputes the full
evaluation and both label-free controls. It does not execute prediction_sources,
refit anything, decode HDF5, call Ollama, or access the network. input-evidence.json
contains the full numeric model input evidence needed to recompute its frozen
identity. Original large HDF5 and original MP4 files are not duplicated; their
hashes and upstream source references are retained. Verification is consistency
checking, not independent rerunning of inference or authentication of authorship.
It recomputes controls and metrics from the saved frame selections/state scores;
it does not rerun Gaussian smoothing, peak detection or the sampling algorithm.

verifier/ contains explicitly shipped executable verification code. Inspect and
trust it before execution. The repository command uses already installed code
and requires version-matched evaluator/report SHA values. Prediction snapshots
are preserved as data and never imported. A checksum is not a signature.

All selected records remain in denominators. Missing or failed predictions
abstain. Conditional MAE is meaningful only with valid-onset coverage. Sampler
oracle coverage measures time resolution, not visible semantic failure evidence.
The clock and motion-peak controls never see labels and always predict failure.
Task-cluster intervals on this small stratified subset are descriptive. This is
offline retrospective diagnosis, not online failure detection or robot control.
