# UR5 VLM protocol audit and fixed development diagnostic

Audit date: 2026-10-15. This audit investigates the completed 16-sample × 3-condition local experiment. All 48 saved predictions are `success`; that is a collapse of the predicted class, not 100% task success or proof of useful visual understanding. The original inputs, protocol snapshots, and raw responses remain unchanged.

## What the local evidence establishes

- The running server reports **Ollama 0.40.4** at `GET http://127.0.0.1:11434/api/version`; the installed CLI agrees. The current server log records its startup at **2026-10-15 10:07:14 +08:00**. The earlier rotated log records 0.34.4 at 10:02:23. The 48 experiment completions range from **11:48:48 to 13:00:24 +08:00**, after the 0.40.4 startup. The previous 0.34.4 summary described an earlier process. The old prediction records did not independently save server version, so this version attribution uses the local server logs and timestamps.
- The pinned model is `qwen3-vl:2b`, digest `0635d9d857d497aeadba3d7d27485746c50554446f9f6ec01ef39788221adbe8`, a Q4_K_M model. Its local metadata declares the `qwen3-vl-thinking` renderer and parser, with thinking enabled by default. It must not be described as the separately tagged `2b-instruct` model; that tag has a different digest. See the official [2b listing](https://ollama.com/library/qwen3-vl:2b) and [2b-instruct listing](https://ollama.com/library/qwen3-vl:2b-instruct).
- Images reached the visual encoding path. The installed Ollama Python SDK 0.6.3 converts the supplied PNG bytes to base64. An offline serialization check recovered identical bytes and SHA256. Matching server request blocks show explicit image encoding for 47 of 48 calls. The remaining reversed call, `ur5-8e2e7ec301a2829cacbc`, reused the prompt/image cache: its before and after source pixels are identical. The server reported 1,251 of 1,252 cached tokens and approximately 88 ms prompt evaluation. This is consistent with cache reuse, not an omitted image. The official [vision API documentation](https://docs.ollama.com/capabilities/vision) describes the same bytes/base64 transport.
- Each original camera image is 256 × 256. The after-only panel is 768 × 280; ordered and reversed panels are 768 × 560, including labels. Runtime logs show a roughly one-megapixel minimum vision area, giving about 1,060 image tokens for after-only and 1,064 for the paired panels. Upscaling does not recover absent source detail; six views share roughly the same visual-token budget as three. Total prompt lengths of 1,243–1,270 tokens do not approach the configured 4,096-token context. This establishes an input-resolution limitation, not that it caused the classification collapse.

Image encoding, fluent visual descriptions, and successful JSON validation are insufficient evidence that the model used the relevant visual details correctly. This audit does not claim visual grounding from those observations.

## Confirmed output-quality defect

**11 of 48 saved evidence strings have exactly 280 characters and lack sentence-ending punctuation.** The original schema caps `evidence` at 280 characters. One after-only response for `ur5-0e266205e2f66630b273` ends mid-phrase at “confirming the sub”. All 48 API responses report `done_reason="stop"`; the existing token-truncation check consequently does not identify these schema-bound evidence cutoffs.

These 11 strings must be marked as likely schema-constrained evidence truncation when interpreting the old run. They are retained verbatim. The class predictions remain the originally recorded `success` values; the cutoff is not itself an explanation of the class collapse. A schema can constrain generation before a server reports a length stop; see Ollama's [structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs).

## Protocol hypotheses that remain unproved

1. **Thinking/prefill interaction.** The original call supplies `think=False` and a final assistant prefix containing an empty, closed `<think>` block. Ollama 0.40.4's [Qwen3-VL renderer](https://raw.githubusercontent.com/ollama/ollama/v0.40.4/model/renderers/qwen3vl.go) preserves image-bearing messages and supports assistant continuation. Its [parser](https://raw.githubusercontent.com/ollama/ollama/v0.40.4/model/parsers/qwen3vl.go) uses a nonempty assistant prefix to start collecting final content; without a prefix it can begin in the thinking state. The prefix therefore has a compatibility rationale. It may change model behavior, but source inspection does not prove that it forces success. A no-prefill response can put JSON into `message.thinking`; silently parsing that as a final answer would hide the protocol problem. The [thinking API documentation](https://docs.ollama.com/capabilities/thinking) distinguishes these fields.
2. **Enum/order prior.** The original schema places `prediction` before `evidence`, with the enum ordered `success`, `failure`, `abstain`. The model commits to a class before producing its evidence. Grammar constraints limit valid outputs; they do not mechanically select the first enum value. A permutation experiment is necessary before attributing collapse to enum order.
3. **Goal-completion prior and small-model capacity.** The prompt states success, failure, and abstention conditions and warns against assuming a closed gripper holds the intended object. Nevertheless, an imperative subtask such as “grasp banana” can plausibly prime a completion narrative. The responses sometimes describe approach/contact as achievement. This is a hypothesis about task verification ability and prompt behavior, not a demonstrated transport bug.
4. **Resolution and temporal relation.** A six-tile panel contains small objects and occluded grippers. The model may miss the intended physical relation or favor a single apparent end state. The reversed condition is a corruption diagnostic scored against the original author label, not a labeled reverse-action trajectory. The pixel-identical sample mentioned above has the author's **success** label; identical pixels must not be relabeled as failure by this audit.

## Installed model control

`minicpm-v4.6:1b` is already installed, digest `e95583acac773b45d95469c069db44808c87295f924183f4c942d52616b2d132`. Local metadata identifies the `qwen35` language family, about 752 million language parameters, and a `minicpmv4_6` visual projector with approximately 548 million parameters. Its native template defaults to disabled thinking and inserts its own closed thinking block. It should receive the same image/question and `think=False`, **without copying the Qwen3-VL assistant prefix**.

This provides a different model/visual-encoder control, not a claim of independent pretraining or a guarantee of better accuracy. The official [model listing](https://ollama.com/library/minicpm-v4.6:1b) and [deployment recipe](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_6_ollama.md) describe its Ollama support. No speed or robot-task performance is inferred from its model name. It is not described as the newest MiniCPM release.

## Frozen four-sample diagnostic

The new `robotbench.vlm_diagnostic` command implements a **post hoc development diagnostic**, not a rerun or a new held-out benchmark. Selection uses only the original 16 pilot IDs and author labels: sort by SHA256 of `["visionforge-vlm-protocol-diagnostic-v1", sample_id]` and take the first two examples of each label. Previous model predictions are not a selection input.

| Author label | Fixed sample ID | Subtask |
|---|---|---|
| success | `ur5-a17751fac2eb11b5cc8d` | grasp banana |
| success | `ur5-b3156884e38e22385224` | grasp banana |
| failure | `ur5-b2b82180904118385ce8` | grasp scissor |
| failure | `ur5-7fa4b21d4f6445cf774c` | grasp white plate |

All four conditions use identical ordered, three-camera panels and task text, deterministic decoding, `num_ctx=4096`, and `num_predict=48`:

| Condition | Model | Enum | Explicit assistant prefix |
|---|---|---|---|
| `baseline_short` | Qwen3-VL 2B | success, failure, abstain | empty closed thinking block |
| `enum_reversed` | same Qwen | abstain, failure, success | same prefix |
| `no_prefill` | same Qwen | success, failure, abstain | none; `think=False` retained |
| `minicpm` | installed MiniCPM 1B | success, failure, abstain | none; native template and `think=False` |

Every condition returns only `{"prediction": ...}`. The original evidence-sentence instruction is replaced by a prediction-only instruction, and the original 160-token cap is reduced to 48. **Thus `baseline_short` is not the original schema or an exact reproduction of the 48 calls.** The short format saves CPU time and permits controlled within-diagnostic comparisons. An enum effect here would not, by itself, establish the cause of the old run.

`freeze` performs metadata reads and image validation, with no inference. The plan binds source code, source input/label files, original pilot plan, all image hashes and composed-panel hashes, both model digests/configurations, server version, schema, prompt, settings, and execution order. `run` shares the existing cross-process inference lock, validates identities before sending images, preserves raw `content` and `thinking`, and resumes only matching saved requests. Existing failed attempts remain recorded instead of being silently replaced. `report` works offline, validates raw-response/derived-prediction consistency, and retains missing or invalid responses in the fixed four-sample denominators. Checksums support consistency auditing, not independent proof of authenticity.

From the repository root:

```console
python -m robotbench.vlm_diagnostic --data data/robotbench --output data/vlm-diagnostic freeze --pilot-plan data/robot-run/vlm-plan.json
python -m robotbench.vlm_diagnostic --data data/robotbench --output data/vlm-diagnostic run
python -m robotbench.vlm_diagnostic --output data/vlm-diagnostic report
```

The 16-cell result should be interpreted as a protocol troubleshooting table. Four label-balanced, previously examined samples cannot support a generalization claim, confidence interval, or model ranking. A shorter response, changed verdict, or a second model's agreement is not a substitute for a visual-grounding test.

## Completed diagnostic result

All **16/16 calls** completed under plan SHA256 `bd10d092050a7a6bdfaca43fb54e30752aa13811f73df0a474f9e5bc73d79627`. The fixed author-label denominator is two successes and two failures in every condition.

| Condition | Valid final JSON | Final predictions | Correct / fixed 4 | Protocol observation |
|---|---:|---|---:|---|
| `baseline_short` | 4/4 | 4 success | 2/4 | No explicit thinking in responses |
| `enum_reversed` | 4/4 | 4 success | 2/4 | 0/4 matched final-verdict flips |
| `no_prefill` | 0/4 | 4 operational abstentions | 0/4 | All four exhausted 48 tokens, contained thinking, and had no final content |
| `minicpm` | 4/4 | 4 success | 2/4 | 0/4 matched final-verdict flips versus the short Qwen baseline |

Reversing the enum did **not** remove success collapse on these four inputs. Removing the explicit Qwen prefix exposed a concrete protocol/budget failure: despite `think=False`, the server returned thinking-only, token-limited responses. The four operational abstentions represent failed final outputs, not four model judgments of uncertainty. This does not show that a longer no-prefill run would classify correctly. The installed MiniCPM control also predicted success throughout this tiny diagnostic; it provides no improvement claim.

The complete, independently verifiable bundle is in [`artifacts/robot-vlm-diagnostic`](../artifacts/robot-vlm-diagnostic/). It retains original raw responses, the frozen plan and source snapshots, exact-byte panel reconstructions checked against pre-inference SHA256, source images, and author labels. From that directory, `python verify.py verify .` recomputes the selection rule, each raw-output verdict and failure status, every fixed-denominator count, and matched comparisons using only the Python standard library. It also checks the complete payload inventory and hashes; it never executes the retained inference source.
