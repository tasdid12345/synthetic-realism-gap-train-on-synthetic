"""Filtered class-conditional GMM generator vs. standard class-conditional Gaussian generator.

Stand-in for sperm-motion features: sklearn breast-cancer data (569 rows, 30 features).
Protocol: train-on-synthetic, test-on-real with logistic regression.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # used for every randomized model
N_COMP = 3                    # GMM components per class (improved model)
OVERSAMPLE = 4                # candidates drawn per kept sample (improved model)
ACCEPT = 0.5                  # discriminator P(real) threshold for keeping a sample


def make_baseline(X, y):
    """Standard class-conditional single-Gaussian generator (fitted on training data only)."""
    params = {}
    for c in (0, 1):
        Xc = X[y == c]
        params[c] = (Xc.mean(0), np.cov(Xc, rowvar=False) + 1e-6 * np.eye(X.shape[1]))

    def sample(counts, rng):
        Xs, ys = [], []
        for c in (0, 1):
            mu, cov = params[c]
            Xs.append(rng.multivariate_normal(mu, cov, size=counts[c]))
            ys.append(np.full(counts[c], c))
        return np.vstack(Xs), np.concatenate(ys)
    return sample


def make_improved(X, y, seed):
    """Class-conditional 3-component GMM; samples kept only if a real-vs-synthetic GBM scores them real."""
    gmms = {c: GaussianMixture(N_COMP, covariance_type="full", reg_covar=1e-2,
                               random_state=seed).fit(X[y == c]) for c in (0, 1)}
    # Discriminator fitted on TRAINING data: real train rows (label 1) vs unfiltered GMM draws (label 0).
    # Class label is appended as a feature so the discriminator also checks class consistency.
    Xg = np.vstack([gmms[c].sample(int(np.sum(y == c)))[0] for c in (0, 1)])
    yg = np.concatenate([np.full(int(np.sum(y == c)), c) for c in (0, 1)])
    Xd = np.vstack([np.column_stack([X, y]), np.column_stack([Xg, yg])])
    ld = np.r_[np.ones(len(X)), np.zeros(len(Xg))]
    disc = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed).fit(Xd, ld)

    def sample(counts, rng):  # rng unused: GMM draws use each fitted GMM's seeded random state
        Xs, ys = [], []
        for c in (0, 1):
            n = counts[c]
            cand = gmms[c].sample(OVERSAMPLE * n)[0]
            p = disc.predict_proba(np.column_stack([cand, np.full(len(cand), c)]))[:, 1]
            order = np.argsort(-p)                       # most "real" first
            keep = order[p[order] >= ACCEPT][:n]
            if len(keep) < n:                            # fallback keeps the sample budget equal to baseline
                print(f"  class {c}: only {len(keep)}/{n} passed filter; filling with top-scored rejects")
                keep_set = set(keep.tolist())
                rest = [i for i in order if i not in keep_set]
                keep = np.r_[keep, rest[:n - len(keep)]].astype(int)
            Xs.append(cand[keep])
            ys.append(np.full(n, c))
        return np.vstack(Xs), np.concatenate(ys)
    return sample


def run(build, Xtr, ytr, Xte, yte, seed):
    """Fit generator on train, train classifier on synthetic, score on real test + secondary metrics."""
    sample = build(Xtr, ytr, seed)
    rng = np.random.default_rng(seed)
    n_tr = {c: int(np.sum(ytr == c)) for c in (0, 1)}
    n_te = {c: int(np.sum(yte == c)) for c in (0, 1)}

    # Main metric: train on synthetic (same budget as real train), test on real held-out data
    Xs, ys = sample(n_tr, rng)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xs, ys)
    pred = clf.predict(Xte)
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred)

    # Secondary: mean per-feature KS between real train and synthetic train
    ks = np.mean([ks_2samp(Xtr[:, j], Xs[:, j]).statistic for j in range(Xtr.shape[1])])

    # Secondary: real-vs-synthetic discriminator AUC on the test split (0.5 = indistinguishable)
    Xe, _ = sample(n_te, rng)                            # fresh synthetic rows for the test split
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)
    gbm.fit(np.vstack([Xtr, Xs]), np.r_[np.ones(len(Xtr)), np.zeros(len(Xs))])
    auc = roc_auc_score(np.r_[np.ones(len(Xte)), np.zeros(len(Xe))],
                        gbm.predict_proba(np.vstack([Xte, Xe]))[:, 1])
    return dict(acc=acc, f1=f1, ks=ks, auc=auc)


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Single stratified 70/30 split; test set is never used for fitting or selection
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)                       # shared preprocessing, fitted on train only
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    print(f"Data: {len(X)} rows, train {len(Xtr)}, test {len(Xte)}")

    builders = {"baseline": lambda A, b, s: make_baseline(A, b),
                "improved": make_improved}
    res = {k: [] for k in builders}
    for s in SEEDS:
        for name, build in builders.items():
            print(f"seed {s}: fitting/evaluating {name}")
            res[name].append(run(build, Xtr, ytr, Xte, yte, s))

    def stat(name, key):
        v = np.array([r[key] for r in res[name]], dtype=float)
        if not np.all(np.isfinite(v)):
            print("Non-finite metric encountered; aborting.")
            sys.exit(1)
        return float(v.mean()), float(v.std())

    m = {}
    for name in builders:
        m[f"{name}_score"], m[f"{name}_score_std"] = stat(name, "acc")
        m[f"{name}_f1"], m[f"{name}_f1_std"] = stat(name, "f1")
        m[f"{name}_disc_auc"], m[f"{name}_disc_auc_std"] = stat(name, "auc")
        m[f"{name}_mean_ks"], m[f"{name}_mean_ks_std"] = stat(name, "ks")
    m["improvement"] = m["improved_score"] - m["baseline_score"]   # accuracy: higher is better
    m["disc_auc_improvement"] = m["baseline_disc_auc"] - m["improved_disc_auc"]  # closer to 0.5 better
    m["mean_ks_improvement"] = m["baseline_mean_ks"] - m["improved_mean_ks"]     # lower is better

    results = {
        "question": "Can a filtered class-conditional GMM generator produce synthetic data that transfers "
                    "better to real held-out data (smaller real-vs-synthetic gap) than a standard "
                    "class-conditional Gaussian generator?",
        "method": "Train-on-synthetic, test-on-real logistic regression; baseline = class-conditional single "
                  "Gaussian; improved = class-conditional 3-component GMM filtered by a real-vs-synthetic "
                  "gradient-boosting discriminator. Seeds 42/43/44.",
        "dataset": {
            "name": "sklearn breast_cancer (Wisconsin diagnostic), stand-in for sperm-motion features",
            "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
            "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "test accuracy on real held-out data (train on synthetic)",
        "higher_is_better": True,
        "baseline": "Class-conditional single multivariate Gaussian generator fitted on training split; "
                    "samples as many rows as real training data.",
        "improved": "Class-conditional 3-component Gaussian mixture generator; oversampled candidates are kept "
                    "only if a gradient-boosting real-vs-synthetic discriminator (fitted on training data) "
                    "scores them P(real) >= 0.5.",
        "metrics": m,
        "notes": "Breast-cancer data is a stand-in, not the paper's sperm-motion data. Single stratified split; "
                 "no hyper-parameter search was run (fixed defaults), so nothing was selected on test data. "
                 "Filter fallback fills with top-scored rejects when too few candidates pass. Std is over 3 seeds "
                 "only; differences may be within noise. Discriminator AUC uses fresh synthetic test-size samples.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")

    # Optional chart: baseline vs improved for main metric and discriminator AUC
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, key, title in [(ax[0], "score", "Test accuracy (real)"),
                          (ax[1], "disc_auc", "Real-vs-synthetic AUC (0.5 ideal)")]:
        vals = [m[f"baseline_{key}"], m[f"improved_{key}"]]
        errs = [m[f"baseline_{key}_std"], m[f"improved_{key}_std"]]
        a.bar(["baseline", "improved"], vals, yerr=errs, capsize=4, color=["gray", "steelblue"])
        a.set_title(title)
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    print("Saved figures/baseline_vs_improved.png")


if __name__ == "__main__":
    main()
