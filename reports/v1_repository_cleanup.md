# ApexDecision V1 cleanup audit

## Before

- Inventory scope: 881 files and directories; 709 files reported by the initial file scan.
- Approximate workspace size: 4.37 GB.
- Model artifacts: V8 50M production checkpoint, V7.7 checkpoint, four 24M checkpoints, teacher weights, and generated adapters/ONNX outputs.
- Tests: three legacy `src` tests.
- Runtime dependencies: the former packaging file listed training, conversion, GPU, and benchmark-era dependencies together.

## Deleted

- `laya_weights/` (846 MB) and `laya_repo/` (5.3 MB): teacher-only assets, not used by V1 runtime.
- `apex_decision/v3/` through `v6/`, `apex_decision/src/`, and the empty nested package directory: superseded implementations.
- Four 24M checkpoints and `student-50m-v77.pt`: non-production models.
- `apex_decision/build/`, caches, compiled Python files, the root object file, V8 baseline manifest, and debug trace script: generated or local-development artifacts.

## Preserved

- Production runtime, typed API, tools, verification, context, reasoning, multilingual support, and the V8 `student-50m-v8.pt` checkpoint with tokenizer assets.
- User-requested research, benchmark, evaluation, and report material.
- A seven-module V1 test suite, examples, package metadata, and open-source documentation.

## Security

- No hard-coded `C:\\Users\\...` or `/home/...` Python paths remain.
- Model resolution is package-relative and supports `APEX_MODEL_PATH`; server host, port, device, logging, API key, and tool-call limit use environment configuration.
- The public audit-trace endpoint was removed; unexpected API failures return RFC 7807-style generic errors.
- No credential files were found in the retained production surface. Historical research scripts are retained by request and are excluded from the V1 public API.

## Verification

- `python -m pytest tests -q`: **10 passed**.
- `python -m build`: source distribution and wheel built successfully; build products were removed afterwards.
- `python -m pip install -e . --no-deps`: successful.
- Real model load and real `POST /v1/systemone` Noul request: successful.
- `/health` and `/version`: successful.

## Final repository

- 220 files after this audit report is included; approximately 302 MB, including the 186 MB production checkpoint.
- Production entry points: `ApexClient`, `apex_decision.api.server:app`, and the `apex-decision` console command.
- Start command: `apex-decision`.

## Remaining limitation

The workspace is not a Git repository, so an actual clone-based clean-checkout validation could not be performed. In addition, retained historical evaluation/research scripts expect the removed teacher assets and legacy `src` package; they are archival research material rather than runnable V1 components.
