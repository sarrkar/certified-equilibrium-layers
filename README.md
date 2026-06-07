# CASCADE

Interpretable GNN for siRNA seed-mediated off-target prediction.

CASCADE extends CCMP (FiLM-conditioned per-concept heterogeneous message
passing, K=9 concept channels) and preserves its by-construction zero
cross-concept leakage property (L_k = 0). The full project contract,
including hard engineering rules, lives in [OUTLINE.md](OUTLINE.md).

## Quickstart

```bash
make dev     # install with dev extras + pre-commit hooks
make lint    # ruff check + format check + em-dash scan
make test    # pytest
make repro   # reproduce all headline numbers from cached data
```

## Layout

See [CLAUDE.md](CLAUDE.md) for the canonical repository layout.

## Reproducibility

Every theorem has a matching verification script under [verify/](verify/).
Every script accepts `--seed` and logs it. A single `make repro` reproduces
the headline numbers from cached data without network access.
