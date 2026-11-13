"""Probabilities of the two reference methods at the calibration operating point.

The calibration comparison of MA-PAN with the boosted trees and with the
imputation-plus-MLP pipeline needs the reliability curves, not only the expected
calibration error.  The main experiment stores only scores, so the two baselines
are refitted here for seed 0 at 30% missingness and their test-set probabilities
are written to ``results/calib_baselines.npz`` for plotting.
"""
import os

import numpy as np

import experiments as E
from baselines import run_baseline

RATE = 0.3
SEED = 0


def main():
    d, (tr, va, te) = E.data()
    p = E.prepared(RATE, "mcar")
    out = {"y": d["y"][te]}
    for m in ("hgb", "mlp", "knn_imp"):
        proba, _ = run_baseline(m, d["X"][tr], p["otr"], d["y"][tr],
                                d["X"][te], p["ote"], d["kind"], seed=SEED)
        out[m] = np.asarray(proba, dtype=np.float32)
        acc = float((out[m].argmax(1) == out["y"]).mean())
        print(f"{m}: test accuracy {acc*100:.1f}", flush=True)
    path = os.path.join(E.RESULTS, "calib_baselines.npz")
    np.savez_compressed(path, **out)
    print("wrote", path)


if __name__ == "__main__":
    main()
