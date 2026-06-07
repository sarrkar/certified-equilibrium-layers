# verify/

Runnable numerical verification scripts. Each theorem in the paper has a
matching script here that empirically confirms the bound on real data.

Per the project contract (see [OUTLINE.md](../OUTLINE.md)), every script in
this directory:

- accepts `--seed` and logs the seed it ran with,
- writes its result artifact to `verify/_out/<script>/`,
- is invoked by `make repro` and must exit non-zero on bound violation.

Planned scripts:

- `pac_bayes_bound.py`     verifies Contribution 1 (per-concept PAC-Bayes).
- `containment_decay.py`   verifies Contribution 2 (PPR distance decay).
- `abstraction_consistency.py` verifies Contribution 3 (causal abstraction).
