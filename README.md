# Filtered GMM vs Gaussian Synthetic Data (Breast-Cancer Stand-in)

This experiment is motivated by "Real-like synthetic sperm video generation from learned behaviors" (S. Hernández-García et al., 2025, Applied Intelligence), which argues that synthetic biological motion data must look realistic and carry the behaviour of real samples to be useful. We test a simple, generic version of that idea on tabular features: can a class-conditional Gaussian-mixture generator whose samples are filtered by a real-vs-synthetic discriminator produce data that transfers better to real held-out data than a standard class-conditional Gaussian generator?

**Question:** Can a filtered class-conditional Gaussian-mixture generator produce synthetic samples that transfer better to real held-out data, and show a smaller real-vs-synthetic gap, than a standard class-conditional Gaussian generator?

**Dataset:** scikit-learn's built-in breast-cancer dataset (569 rows, 30 continuous features, binary label), loaded without download. It is a stand-in for sperm-motion features and is not the paper's own dataset. A stratified 70/30 split is used; the test set is held out from all generator fitting.

**Baseline:** class-conditional single-Gaussian generator fitted on the training split, sampling as many synthetic rows as real ones; a logistic regression is trained only on synthetic data and tested on real data.

**Improved model:** class-conditional 3-component Gaussian mixture generator whose samples are kept only if a gradient-boosting real-vs-synthetic discriminator (fitted on training data) scores them as realistic. Everything else (data, classifier, budget, seeds) is identical.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the public breast_cancer dataset (stand-in for sperm-motion data), a standard class-conditional Gaussian generator reached 96.1% test accuracy when training on synthetic data, while the filtered 3-component GMM reached 96.5%, a small gain within seed noise (std about 1 point).

| Metric | Value |
|---|---|
| baseline_score | 0.961 |
| baseline_score_std | 0.01202 |
| baseline_f1 | 0.9693 |
| baseline_f1_std | 0.009214 |
| baseline_disc_auc | 0.827 |
| baseline_disc_auc_std | 0.004705 |
| baseline_mean_ks | 0.08311 |
| baseline_mean_ks_std | 0.001486 |
| improved_score | 0.9649 |
| improved_score_std | 0.00827 |
| improved_f1 | 0.9721 |
| improved_f1_std | 0.006772 |
| improved_disc_auc | 0.7204 |
| improved_disc_auc_std | 0.01156 |
| improved_mean_ks | 0.07504 |
| improved_mean_ks_std | 0.008109 |
| improvement | 0.003899 |
| disc_auc_improvement | 0.1066 |
| mean_ks_improvement | 0.008068 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Breast-cancer data is a stand-in, not the paper's sperm-motion data. Single stratified split; no hyper-parameter search was run (fixed defaults), so nothing was selected on test data. Filter fallback fills with top-scored rejects when too few candidates pass. Std is over 3 seeds only; differences may be within noise. Discriminator AUC uses fresh synthetic test-size samples.

## Reference

This project is motivated by the following paper:

> S Hernández-García, A Cuesta-Infante, D Makris (2025). *Real-like synthetic sperm video generation from learned behaviors: S. Hernández-García et al.*. Applied. https://link.springer.com/article/10.1007/s10489-025-06407-3

Paper author: Prof. Dr. Antonio Sanz Montemayor, Universidad Rey Juan Carlos (antonio.sanz@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
