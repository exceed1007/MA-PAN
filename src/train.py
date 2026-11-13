"""Data handling, training loop and metrics shared by all experiments."""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

import autograd as ag
from autograd import Tensor
from benchmark import impose_missing
from model import MAPAN, BinMapper, AdamW


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_bench(path="benchmark.npz"):
    d = np.load(path, allow_pickle=True)
    return dict(X=d["X"], y=d["y"], y_coarse=d["y_coarse"], mask=d["mask"],
                names=[str(s) for s in d["names"]], kind=d["kind"], ncat=d["ncat"],
                difficulty_idx=int(d["difficulty_idx"]))


def stratified_split(y, seed=0, frac=(0.7, 0.1, 0.2)):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n1 = int(round(frac[0] * idx.size))
        n2 = int(round((frac[0] + frac[1]) * idx.size))
        tr.append(idx[:n1])
        va.append(idx[n1:n2])
        te.append(idx[n2:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)), np.sort(np.concatenate(te))


def corrupt(d, idx, rate, protocol="mcar", seed=0):
    """Observation mask for the given subset after controlled incompleteness."""
    code = {"mcar": 1, "mar": 2, "block": 3, "mnar": 4}[protocol]
    rng = np.random.default_rng(seed * 100003 + int(rate * 100) * 17 + code)
    groups = [[j for j, n in enumerate(d["names"]) if _layer_of(n) == g]
              for g in ("L3", "L4", "L7", "DV")]
    meta = dict(difficulty_idx=d["difficulty_idx"], groups=groups)
    return impose_missing(rng, d["X"][idx], d["mask"][idx], rate, protocol, meta)


_LAYERS = None


def _layer_of(name):
    global _LAYERS
    if _LAYERS is None:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "data", "feature_spec.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        _LAYERS = {f["name"]: f["layer"] for f in spec["features"]}
    return _LAYERS[name]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, proba, n_classes=None):
    n_classes = n_classes or proba.shape[1]
    pred = proba.argmax(axis=1)
    acc = float((pred == y_true).mean())
    f1s, ps, rs = [], [], []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y_true == c))
        fp = np.sum((pred == c) & (y_true != c))
        fn = np.sum((pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        ps.append(p)
        rs.append(r)
    sup = np.bincount(y_true, minlength=n_classes).astype(float)
    f1_w = float(np.average(f1s, weights=sup)) if sup.sum() else 0.0
    # expected calibration error (15 equal-frequency bins)
    conf = proba.max(axis=1)
    correct = (pred == y_true).astype(float)
    order = np.argsort(conf)
    n = len(order)
    ece = 0.0
    for chunk in np.array_split(order, 15):
        if chunk.size:
            ece += chunk.size / n * abs(correct[chunk].mean() - conf[chunk].mean())
    return dict(acc=acc, f1_macro=float(np.mean(f1s)), f1_weighted=f1_w,
                precision=float(np.mean(ps)), recall=float(np.mean(rs)), ece=float(ece))


def prf_per_class(y_true, proba, names):
    pred = proba.argmax(axis=1)
    out = []
    for c, nm in enumerate(names):
        tp = np.sum((pred == c) & (y_true == c))
        fp = np.sum((pred == c) & (y_true != c))
        fn = np.sum((pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.append(dict(cls=nm, precision=float(p), recall=float(r), f1=float(f),
                        support=int(tp + fn)))
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(d=32, heads=4, layers=2, K=3, p_drop=0.15, lr=5e-3, wd=2e-2,
                   epochs=40, batch=64, patience=12, l2=0.0, coarse=0.2, proto=0.2,
                   cons=0.3, temp=2.0, extra_hide=0.20, warmup=3, rand_miss=None)


def train_map(bins_tr, obs_tr, ytr, yctr, bins_va, obs_va, yva, ycva,
              mapper, n_classes, n_super, cfg=None, seed=0, verbose=False,
              use_proto=True, use_gate=True, use_miss_bias=True, use_coarse=True,
              use_cons=True, use_proto_loss=True, log=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rng = np.random.default_rng(seed)
    model = MAPAN(mapper, n_classes, n_super, d=cfg["d"], heads=cfg["heads"],
                  layers=cfg["layers"], K=cfg["K"], p_drop=cfg["p_drop"], seed=seed,
                  use_proto=use_proto, use_gate=use_gate, use_miss_bias=use_miss_bias,
                  use_coarse=use_coarse)
    opt = AdamW([p.t for p in model.params], lr=cfg["lr"], wd=cfg["wd"])
    wts = dict(coarse=cfg["coarse"] if use_coarse else 0.0,
               proto=cfg["proto"] if (use_proto and use_proto_loss) else 0.0,
               cons=cfg["cons"] if use_cons else 0.0, l2=cfg["l2"])

    n = bins_tr.shape[0]
    steps_per_epoch = max(1, n // cfg["batch"])
    total_steps = cfg["epochs"] * steps_per_epoch
    best = (-1.0, None, 0)
    hist = []

    for ep in range(cfg["epochs"]):
        perm = rng.permutation(n)
        run = 0.0
        for s in range(steps_per_epoch):
            sl = perm[s * cfg["batch"]:(s + 1) * cfg["batch"]]
            b_batch, o_batch = bins_tr[sl], obs_tr[sl]
            if cfg["rand_miss"]:
                # missingness-consistent augmentation: every sample draws its own rate
                rate_i = rng.random((b_batch.shape[0], 1)) * float(cfg["rand_miss"])
                hide = rng.random(o_batch.shape) < rate_i
                o_batch = o_batch & ~hide
                if hide.any():
                    b_batch = b_batch.copy()
                    mask = ~o_batch
                    b_batch[mask] = np.broadcast_to(model.absent_idx, b_batch.shape)[mask]
            out = model.forward(b_batch, o_batch, training=True, rng=rng)
            cons_out = None
            if use_cons:
                cons_out = model.forward(b_batch, o_batch, training=True,
                                         extra_hide=cfg["extra_hide"], rng=rng)
            loss, comp = model.compute_loss(out, ytr[sl], yctr[sl], wts, cons_out, cfg["temp"])
            for p in model.params:
                p.t.grad = None
            loss.backward()
            # cosine schedule with linear warm-up
            gstep = ep * steps_per_epoch + s
            if gstep < cfg["warmup"] * steps_per_epoch:
                lr = cfg["lr"] * (gstep + 1) / (cfg["warmup"] * steps_per_epoch)
            else:
                prog = (gstep - cfg["warmup"] * steps_per_epoch) / max(
                    1, total_steps - cfg["warmup"] * steps_per_epoch)
                lr = 0.5 * cfg["lr"] * (1 + np.cos(np.pi * prog)) + 1e-5
            opt.step(lr)
            run += float(loss.data)
            if (s + 1) % 100 == 0:
                # a node and its backward closure form a reference cycle, so the
                # cycle collector - not reference counting - frees a graph; it is
                # nudged here because a long run otherwise accumulates tensors
                gc.collect()

        proba = model.predict_proba(bins_va, obs_va)
        m = metrics(yva, proba, n_classes)
        sub = rng.choice(bins_tr.shape[0], size=min(800, bins_tr.shape[0]), replace=False)
        mt = metrics(ytr[sub], model.predict_proba(bins_tr[sub], obs_tr[sub], batch=256), n_classes)
        hist.append(dict(epoch=ep, loss=run / steps_per_epoch, val_f1=m["f1_macro"], val_acc=m["acc"],
                         train_f1=mt["f1_macro"]))
        if verbose:
            print(f"  ep{ep:02d} loss={run/steps_per_epoch:.3f} trainF1={mt['f1_macro']:.4f} "
                  f"valF1={m['f1_macro']:.4f} valAcc={m['acc']:.4f} miss={1-obs_va.mean():.2f}",
                  flush=True)
        if m["f1_macro"] > best[0] + 1e-5:
            best = (m["f1_macro"], [p.data.copy() for p in model.params], ep)
        elif ep - best[2] >= cfg["patience"]:
            break

    for p, b in zip(model.params, best[1]):
        p.t.data = b.copy()
    return model, hist, best[2]
