# Soup Kernel Challenge — Submission (Decentrathon 6.0)

- **Repository:** https://github.com/adilhan385/soup-kernel-challenge
- **Final commit:** [УТОЧНИТЬ: хеш, указанный в COMMIT публичного ноутбука]
- **Measured solution code:** `cd1b98d` (same `solution/` code, only a rename and an unused import removed since)
- **Public Kaggle notebook (Save & Run All):** [ССЫЛКА НА НОУТБУК]
- **Demo video:** [УТОЧНИТЬ]
- **Presentation:** [УТОЧНИТЬ]

## Measured results (Kaggle Tesla T4, `python harness/run_task.py`, 1 repeat)

| Metric | Base (Soup 0.77.0) | Solution | Ratio |
|---|---:|---:|---:|
| Full `soup train` wall time | 84.8 s | 62.9 s | speedup 1.348x |
| Peak GPU memory (nvidia-smi) | 14345 MiB | 2779 MiB | memory 5.162x |
| Eval loss | 2.04967 | 2.04977 | +0.005 % |
| Own Triton kernels share (steps 3+) | — | 10.6 % | ≥ 5 % required |

`harness/check_rules.py`: ALL CHECKS PASSED (10/10, 79 Soup tests passed). `pytest solution -q`: 5 passed.

See `README.md` for the kernels, how to run, and limitations.
