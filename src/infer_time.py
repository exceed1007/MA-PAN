"""Measure per-entity inference time for every method in the cost comparison.

A model is trained briefly (`--epochs`, default 12) because inference cost does
not depend on how long the model was trained; the numbers are then written to
``results/infer_time.json`` in seconds per entity, consumed by the
evaluation scripts.
"""
import argparse
import json
import os
import time

import numpy as np

import experiments as E
import train as T


def bench(fn, n, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d, (tr, va, te) = E.data()
    p = E.prepared(args.rate, "mcar")
    n_te = p["bte"].shape[0]
    out = {}

    # ---- MA-PAN (and the prototype-free variant)
    for tag, variant in (("mapan", "full"), ("ft_lite", "ft_lite")):
        model, _, _ = E.train_one(args.rate, "mcar", args.seed, variant=variant,
                                  cfg=dict(epochs=args.epochs, patience=99))
        bt, ot = p["bte"], p["ote"]
        out[tag] = bench(lambda: model.predict_proba(bt, ot, batch=64), n_te)
        print(f"{tag:<9} {out[tag]*1000:.3f} ms/entity", flush=True)

    # ---- classical baselines
    for m in ["svm", "rf", "hgb", "mlp", "knn_imp"]:
        t0 = time.perf_counter()
        proba, _ = E.run_baseline(m, d["X"][tr], p["otr"], d["y"][tr],
                                  d["X"][te], p["ote"], d["kind"], seed=args.seed)
        fit = time.perf_counter() - t0
        # a second call re-does imputation + prediction, which is what an
        # online deployment pays per batch
        out[m] = bench(lambda: E.run_baseline(m, d["X"][tr], p["otr"], d["y"][tr],
                                              d["X"][te], p["ote"], d["kind"], seed=args.seed),
                       n_te, repeat=2)
        print(f"{m:<9} {out[m]*1000:.3f} ms/entity (fit+predict {fit:.1f} s)", flush=True)

    path = os.path.join(E.RESULTS, "infer_time.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
