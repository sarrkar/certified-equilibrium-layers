# Project 2 (AAAI): certification results

Generated artifacts (JSON + figures only, no prose paper; see OUTLINE.md rule 7).

Per the hard data policy (OUTLINE.md), the real-data experiments under
`experiments/aaai/` (`exp_realdata.py`, `exp_backbone_efficacy.py`) use ONLY real
measured data and raise until that data is wired, so the application and backbone
sections are reported as PENDING. The certificate-soundness experiments are
random-graph math verification (no biological claim) and live under
`experiments/soundness/`.

## How to reproduce

```bash
# Hard data policy gate, then aggregate the AAAI paper numbers (real-data only):
make repro-aaai

# Certificate soundness (random-graph verification, no biological claim):
python experiments/soundness/run_certification.py --seed 0
python experiments/soundness/exp_verify.py --seed 0
python experiments/soundness/exp_tightness.py --seed 0
python experiments/soundness/exp_normalized.py --seed 0
```

`run_certification.py` writes `certified_region.json` and `certified_region.png`.

## What it measures

For y = P_alpha(S) x0 (the CASCADE propagation, reused from
`cascade.verify.containment`) under the confidence-calibrated random-edge noise
model on S, it certifies a radius R with

    P( || y' - y0 ||_2 <= R ) >= 1 - delta

at finite-sample confidence 1 - gamma. Two radii are reported per delta:

- `radius_mc`: distribution-free Monte-Carlo radius (binomial order-statistic
  upper bound on the (1-delta) quantile);
- `radius_analytic`: the resolvent perturbation bound
  ((1-alpha)/alpha) || S' - S ||_2 || x0 ||, which dominates the Monte-Carlo
  radius.

`coverage_out_of_sample` checks the certificate on fresh perturbations;
`all_calibrated` is true when realized coverage meets the nominal level at every
delta.
