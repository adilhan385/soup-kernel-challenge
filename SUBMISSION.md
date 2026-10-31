# Soup Kernel Challenge — Submission (Decentrathon 6.0)

- **Repository:** https://github.com/adilhan385/soup-kernel-challenge
- **Final commit:** `2be96d7` (latest `main`)
- **Measured solution code:** `2be96d7`
- **Public Kaggle notebook (Save & Run All):** https://www.kaggle.com/code/lolkek4ik/notebook90b0a93b32
- **Demo video script / structure:** [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) (запись видео по сценарию)
- **Presentation (4 slides):** [PRESENTATION.md](PRESENTATION.md)

## Measured results (Kaggle Tesla T4, `python harness/run_task.py`, 1 repeat)

| Metric | Base (Soup 0.77.0) | Solution | Ratio |
|---|---:|---:|---:|
| Full `soup train` wall time | 84.6 s | 62.5 s | speedup 1.352x |
| Peak GPU memory (nvidia-smi) | 14345 MiB | 2759 MiB | memory 5.199x |
| Eval loss | 2.04968 | 2.04976 | +0.004 % |
| Own Triton kernels share (steps 3+) | — | 10.6 % | ≥ 5 % required |

`harness/check_rules.py`: ALL CHECKS PASSED (10/10, 79 Soup tests passed). `pytest solution -q`: 5 passed.

See `README.md` for the kernels, how to run, and limitations.
