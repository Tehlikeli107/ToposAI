# Release Guide

This guide keeps ToposAI releases reproducible and reviewable. It is written
for maintainers preparing a public tag or a clean pull request.

## Clean Workspace Checks

Run these before opening a release PR:

```bash
git status --short
git ls-files | grep -E '(__pycache__|\.pyc$)'
ruff check topos_ai/
python -m compileall -q topos_ai tests applications benchmarks experiments
python -m pytest -q
python -m build
```

The second command should print nothing. Generated caches, checkpoints, logs,
and local benchmark artifacts should stay ignored.

## Suggested Commit Groups

Keep changes reviewable by splitting large cleanups into focused commits:

1. Core math and runtime fixes.
2. Tests and claim-hygiene coverage.
3. Documentation and release metadata.
4. CI, packaging, and wheel-install verification.
5. Benchmark records or result updates.

Avoid mixing generated files with source edits. If benchmark output is useful,
record the raw command, hardware, and summary in `docs/BENCHMARKS.md` instead of
committing transient caches.

## Release Validation

After building the package, install the wheel in a fresh environment and run a
minimal import/model smoke test:

```bash
python -m venv .venv-wheel
. .venv-wheel/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
python - <<'PY'
import torch
import topos_ai
from topos_ai.models import ToposTransformer

model = ToposTransformer(vocab_size=32, d_model=16, num_layers=1, num_universes=2)
idx = torch.randint(0, 32, (1, 8))
logits, _ = model(idx)
assert logits.shape == (1, 8, 32)
print("wheel smoke passed", topos_ai.__version__)
PY
```

On Windows PowerShell, activate with `.venv-wheel\Scripts\Activate.ps1`.
