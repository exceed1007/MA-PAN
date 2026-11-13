"""Classical baselines for incomplete fingerprint classification."""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Imputation helpers
# --------------------------------------------------------------------------- #
def fit_fill(X, mask, kind):
    """Per-column fill values estimated on the training split only."""
    fill = np.zeros(X.shape[1])
    for j, k in enumerate(kind):
        col = X[mask[:, j], j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            fill[j] = 0.0
        elif k == "num":
            fill[j] = float(np.median(col))
        else:
            vals, cnt = np.unique(np.round(col), return_counts=True)
            fill[j] = float(vals[cnt.argmax()])
    return fill


def _masked(X, mask):
    """Set unobserved entries to NaN so that a model can route them itself."""
    return np.where(np.asarray(mask, dtype=bool), np.asarray(X, dtype=float), np.nan)


def densify(X, mask, fill, indicators=True):
    Z = np.where(mask, np.nan_to_num(X, nan=0.0), fill[None, :])
    if indicators:
        Z = np.concatenate([Z, (~mask).astype(float)], axis=1)
    return Z


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def run_baseline(name, Xtr, obstr, ytr, Xte, obste, kind, seed=0):
    t0 = time.time()
    fill = fit_fill(Xtr, obstr, kind)
    Ztr = densify(Xtr, obstr, fill)
    Zte = densify(Xte, obste, fill)

    if name == "svm":
        clf = Pipeline([("sc", StandardScaler()), ("m", SVC(C=10.0, gamma="scale", probability=True,
                                                           random_state=seed))])
        clf.fit(Ztr, ytr)
        proba = clf.predict_proba(Zte)

    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, n_jobs=1,
                                     random_state=seed)
        clf.fit(Ztr, ytr)
        proba = clf.predict_proba(Zte)

    elif name == "hgb":
        # native support for missing values, no explicit imputation; the mask is
        # applied to the design matrix here because the observation mask, not the
        # value matrix, is the authoritative description of what was observed
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             early_stopping=True, random_state=seed)
        clf.fit(_masked(Xtr, obstr), ytr)
        proba = clf.predict_proba(_masked(Xte, obste))

    elif name == "mlp":
        clf = Pipeline([("sc", StandardScaler()),
                        ("m", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                            batch_size=128, learning_rate_init=3e-3,
                                            max_iter=200, early_stopping=True,
                                            n_iter_no_change=10, random_state=seed))])
        clf.fit(Ztr, ytr)
        proba = clf.predict_proba(Zte)

    elif name == "knn_imp":
        # k-nearest-neighbour imputation followed by a random forest
        imp = KNNImputer(n_neighbors=10, weights="distance")
        Ztr_i = imp.fit_transform(np.where(obstr, np.nan_to_num(Xtr, nan=0.0), np.nan))
        Zte_i = imp.transform(np.where(obste, np.nan_to_num(Xte, nan=0.0), np.nan))
        clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=seed)
        clf.fit(Ztr_i, ytr)
        proba = clf.predict_proba(Zte_i)

    elif name == "iter_imp":
        # iterative (MICE-style) imputation with ridge regressors, then RF
        imp = IterativeImputer(max_iter=10, random_state=seed, initial_strategy="median")
        Ztr_i = imp.fit_transform(np.where(obstr, np.nan_to_num(Xtr, nan=0.0), np.nan))
        Zte_i = imp.transform(np.where(obste, np.nan_to_num(Xte, nan=0.0), np.nan))
        clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=seed)
        clf.fit(Ztr_i, ytr)
        proba = clf.predict_proba(Zte_i)

    else:
        raise ValueError(name)

    return proba, time.time() - t0
