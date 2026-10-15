Post hoc local VLM protocol diagnostic: four fixed development samples, sixteen calls.

Run offline verification with Python's standard library:
  python verify.py verify .
The verifier does not execute runtime/ or require Ollama/ML packages. Inspect code before running it.

plan.json binds exact inputs, author labels, image/panel hashes, code, model digests and server version.
records/ preserves every complete raw response including thinking-only/truncated results.
report.json is independently recomputed from those responses, with all four samples per condition.
panels/ contains exact-byte reconstructions verified against the pre-inference plan; images/ contains source PNGs.
Full source input/label rows and the old pilot plan permit reproducing the stable label-stratified selection.
Source dataset files are hash-pinned copies, not a claim of authentic independent authorship.
Checksums detect corruption/mixing, not an attacker rewriting and re-signing the entire bundle.

baseline_short differs from the original 48-call evidence schema and token budget.
This is not a generalization estimate or a model ranking. No confidence/significance claim is made.
All runtime files are retained for audit only. No old benchmark evidence was edited.
