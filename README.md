# ADB Quantum Challenge — Joint-Loss Modelling for Natural Catastrophe Risk

Prototype for the Phase 1 submission. The technical report, with results and an honest
assessment, is in **[REPORT.md](REPORT.md)**.

## Layout

```
src/model/       Phase 0  exact reproduction (compound Poisson fit, 25-state CTMC, rain, 3-year convolution, targets)
src/baselines/   Phase 1  MC, scrambled Sobol, stratified, PDF importance sampling, zero-variance Doob (exact)
src/learned/     Phase 2  learned Doob h-transform: log-linear Ṽ_θ fitted by multilevel cross-entropy
src/quantum/     Phase 3  QCBM proposal (exact statevector sim), IQAE emulation, IS→QAE cascade, resource estimates
src/tensor/      Phase 4  n-country synthetic model, MPS in a tilted gauge, exact targets, bond-dimension scaling
src/causal/      Phase 5  classical (nonnegative rank) vs quantum (PSD rank) latent dimension across cuts
src/figures.py   report figures
tests/           28 tests (pytest)
results/         JSON outputs of every phase (+ logs)
figures/         PNG figures used in REPORT.md
```

## Reproduce

Environment: Python 3.13, numpy 2.5, scipy 1.18, matplotlib, pytest. CPU only (4 cores);
no quantum hardware or quantum SDK was used. Quantum circuits are simulated with an exact
real-amplitude statevector simulator written here (`src/quantum/qcbm.py`); IQAE is emulated
exactly from the circuit's ideal measurement statistics (`src/quantum/iqae.py`).

```bash
pip install numpy scipy matplotlib pytest
```

```bash
python -m pytest -q tests/
```

```bash
PYTHONPATH=. python -m src.model.exact            # Phase 0  (< 1 s)
PYTHONPATH=. python -m src.baselines.run_baselines # Phase 1  (~ 3 min)
PYTHONPATH=. python -m src.learned.run_learned     # Phase 2  (~ 40 min)
PYTHONPATH=. python -m src.quantum.run_quantum     # Phase 3  (~ 15 min)
PYTHONPATH=. python -m src.tensor.scaling          # Phase 4  (~ 20 min)
PYTHONPATH=. python -m src.causal.latent_dimension # Phase 5  (~ 20 min)
PYTHONPATH=. python -m src.figures
```

All randomness uses fixed seeds, which are recorded in each `results/*.json`.

## AI use

The code and report were written with Claude Code (Anthropic), following the team's plan
document. All numbers in the report come from the scripts in this repository.
