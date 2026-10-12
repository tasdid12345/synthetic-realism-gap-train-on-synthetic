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

## Reference

This project is motivated by the following paper:

> S Hernández-García, A Cuesta-Infante, D Makris (2025). *Real-like synthetic sperm video generation from learned behaviors: S. Hernández-García et al.*. Applied. https://link.springer.com/article/10.1007/s10489-025-06407-3

Paper author: Prof. Dr. Antonio Sanz Montemayor, Universidad Rey Juan Carlos (antonio.sanz@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
