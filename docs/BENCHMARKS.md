# Benchmark Reporting

ToposAI benchmark numbers are only meaningful when they are tied to a concrete
environment. Use this page as the canonical place to record reproducible
results before mentioning them in the README, papers, or release notes.

## Current Status

No benchmark table in this repository should be treated as a universal result.
The benchmark scripts are useful for comparing local behavior, but headline
claims need fresh measurements on clean hardware.

## Required Metadata

Each recorded run should include:

- Date and git commit.
- CPU, GPU, RAM/VRAM, operating system, Python version, PyTorch version.
- Exact command and arguments.
- Random seed if applicable.
- Raw output or a link to raw logs.
- A short interpretation that distinguishes measured behavior from hypotheses.

## Result Template

```markdown
### YYYY-MM-DD - Benchmark Name

- Commit: `<git-sha>`
- Hardware: `<CPU/GPU/RAM/OS>`
- Environment: `Python <version>, PyTorch <version>`
- Command: `<exact command>`
- Result:

| Metric | Baseline | ToposAI | Notes |
|--------|----------|---------|-------|
| latency_ms | TBD | TBD | median of N runs |
| peak_vram_mb | TBD | TBD | measured with tool/version |

Interpretation:
Measured on this machine only. Do not generalize until reproduced.
```

## Recommended Commands

```bash
python benchmarks/scaling_laws_benchmark.py
python benchmarks/mamba_vs_attention_benchmark.py
python benchmarks/real_world_needle_in_haystack.py
```

If a benchmark requires CUDA or external data, document that clearly and keep it
out of the default CPU CI path.
