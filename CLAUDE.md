# CLAUDE.md

This file is the authoritative project contract for CASCADE. It is read at the
start of every Claude Code session. Every future change MUST conform to the
rules below. If a request conflicts with this file, stop and surface the
conflict to the user before acting.

## Project

CASCADE is an interpretable GNN for siRNA seed-mediated off-target prediction.
It extends a prior model (CCMP: FiLM-conditioned per-concept heterogeneous
message passing, K=9 concept channels) and preserves CCMP's by-construction
zero cross-concept leakage property (L_k = 0).

Target venue: ICLR. Secondary venue: AAAI.

## Contributions (priority order)

1. **Per-concept compositional PAC-Bayes generalization bound.** A bound that
   is channel-separated across the K concept channels, plus a theorem that
   L_k = 0 isolation strictly tightens the bound by zeroing the cross-channel
   covariance term.
2. **Certified off-target containment.** The architecture constrains GRN
   propagation to a bounded-alpha personalized-PageRank operator, so predicted
   off-target signal at graph distance r decays at a certified rate. The
   upgrade target for this contribution is an effective-resistance /
   spectral-gap distance bound.
3. **Causal-abstraction consistency corollary.** Under L_k = 0, the seed
   concept channel is a consistent causal abstraction of the propagation
   submodule.

## Hard engineering rules

These rules are non-negotiable. Violating any of them is grounds to reject a
change.

1. **Theorem / verification parity.** Every theorem MUST have a matching
   numerical verification script under `verify/` (top level) that empirically
   confirms the bound on real data. A theorem without a passing verification
   script does not ship.
2. **By-construction containment.** Containment is enforced architecturally
   (bounded teleport alpha, spectral-norm-clipped diffusion). It is NEVER
   enforced via a loss penalty or a soft regularizer.
3. **Seeded determinism.** Every script accepts `--seed` and logs the seed it
   ran with. Default seed values are not silently chosen.
4. **No em-dashes.** Anywhere. Not in code comments, docstrings, README, the
   paper, commit messages, or any other artifact in this repo. Use a comma,
   a colon, parentheses, or split the sentence.
5. **Reproducibility.** A single `make repro` reproduces all headline numbers
   from cached data without network access.
6. **Commit cadence.** Commit after each completed unit of work. Use
   Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
   `chore:`, `build:`, `ci:`).

## Stack

Python 3.11. Pinned in `pyproject.toml`.

- PyTorch, PyTorch Geometric
- numpy, scipy, pandas, scikit-learn
- ViennaRNA (Python bindings)
- hydra-core for configuration
- pytest for tests
- ruff for lint and format

## Repository layout

```
src/cascade/data/        loaders + caching
src/cascade/features/    seed-match + thermodynamics + per-concept features
src/cascade/models/      ccmp_backbone, propagation_head, cascade
src/cascade/verify/      pac_bayes, containment, abstraction audits (library)
src/cascade/baselines/   seedmatch, appnp, sequence_only, linear
src/cascade/eval/        metrics + figures
configs/                 hydra configs
tests/                   pytest
paper/                   LaTeX
scripts/                 data download
verify/                  runnable numerical verification scripts (rule 1)
```

The split between `src/cascade/verify/` and top-level `verify/` is
deliberate: the former holds the library code (audits, estimators), the
latter holds the runnable scripts that each correspond to a theorem.

## Workflow

- Use hydra for all configuration. No hard-coded paths.
- Tests live next to behavior, not next to code. Put unit tests in `tests/`.
- Lint and tests run in CI on every push (see `.github/workflows/ci.yml`).
- Pre-commit hooks run ruff. Install with `pre-commit install`.
