# Soup Kernel Challenge — Submission (Decentrathon 6.0)

- **Repository:** https://github.com/adilhan385/soup-kernel-challenge
- **Final commit:** `26512f3` (or latest `main`)
- **Measured solution code:** `26512f3`
- **Public Kaggle notebook (Save & Run All):** https://www.kaggle.com/code/lolkek4ik/notebook90b0a93b32
- **Demo video script / structure:** [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) (запись видео по сценарию)
- **Presentation (4 slides):** [PRESENTATION.md](PRESENTATION.md)

## Measured results (Kaggle Tesla T4, `python harness/run_task.py`, 1 repeat)

| Metric | Base (Soup 0.77.0) | Solution | Ratio |
|---|---:|---:|---:|
| Full `soup train` wall time | 79.1 s | 58.4 s | speedup 1.355x |
| Peak GPU memory (nvidia-smi) | 14345 MiB | 2759 MiB | memory 5.199x |
| Eval loss | 2.04966 | 2.04980 | +0.007 % |
| Own Triton kernels share (steps 3+) | — | 10.6 % | ≥ 5 % required |

`harness/check_rules.py`: ALL CHECKS PASSED (10/10, 79 Soup tests passed). `pytest solution -q`: 5 passed.

See `README.md` for the kernels, how to run, and limitations.
