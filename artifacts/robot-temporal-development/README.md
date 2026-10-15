# Development protocol evidence — excluded from test scores

These are the same three development episodes (two failures, one success),
from two task groups excluded entirely from the 16-episode test subset.
Each completed version has 3 episodes × 2 samplers, not six independent episodes.
All records, exact model-input JPEG panels, plans, source snapshots, annotation
snapshots and completed evaluations are copied byte for byte. samples.json is
rebuilt as metadata only: local file URLs and machine paths are omitted.
input-evidence.json preserves numeric inputs needed to verify frozen identities.
Original HDF5 files and full original videos are not duplicated.

- v1: six raw responses rejected by the strict independent class/frame parser.
  Failed records remain abstentions in the saved evaluation. Raw responses and
  errors remain available under predictions/; none was silently repaired.
- v2-blocked: the first joint-enum plan was stopped by source-identity checks
  before any inference. Its frozen plan/data/source snapshots are retained.
  There are zero prediction records and no invented evaluation for this plan.
- v2-ready: after the adapter source was stable, six responses were structurally
  valid and all predicted success. Each sampler is correct on 1/3 episodes,
  detects 0/2 failures and has Hit@2s 0/2. Format repair did not establish
  failure-detection ability. These findings refer to this model and protocol.

The v1 independent fields were replaced before test inference by one joint
decision enum. Its 12 failure-location choices and one success/abstain choice
are not generation-symmetric; changes in class distribution cannot be
attributed solely to visual reasoning. No test outputs were used for this change.

Verify from the version-matched repository (standard library only):

    python -B tools/export_temporal_development.py --verify --out artifacts/robot-temporal-development

The verifier checks file inventories/hashes, plan and source hashes, numeric
input identities, author-label reconstruction, raw output parsing, panel hashes,
and completed evaluations through the existing report validators. It does not
execute prediction_sources, call a model, decode videos or contact the network.
evaluation_sources are the evaluator/report versions used to validate this
export, separate from the original inference snapshots. Checksums establish
consistency, not authorship or independent reproduction of model inference.
