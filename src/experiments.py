"""
Experiment driver.

Every experiment is expressed as a job dictionary and dispatched to a pool of
worker processes (the numpy implementation is CPU bound and independent across
jobs).  Results are appended to results/results.jsonl so that an interrupted
run can be resumed without recomputing finished jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("MAPAN_DTYPE", "f32")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train as T                       # noqa: E402
from baselines import run_baseline      # noqa: E402
from model import BinMapper             # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data", "benchmark.npz")

_CACHE = {}


def data():
    if "d" not in _CACHE:
        _CACHE["d"] = T.load_bench(DATA)
        d = _CACHE["d"]
        _CACHE["split"] = T.stratified_split(d["y"], seed=0)
    return _CACHE["d"], _CACHE["split"]


def prepared(rate, protocol, seed=0, extra_hide=0.0):
    """Binned tensors + observation masks for one incompleteness setting."""
    d, (tr, va, te) = data()
    key = (rate, protocol, extra_hide)
    if key not in _CACHE:
        otr = T.corrupt(d, tr, rate, protocol, seed=0)
        ova = T.corrupt(d, va, rate, protocol, seed=0)
        ote = T.corrupt(d, te, rate, protocol, seed=0)
        mapper = BinMapper(d["kind"], d["ncat"],
                           num_bins=int(os.environ.get("MAPAN_BINS", 16))).fit(d["X"][tr], otr)
        _CACHE[key] = dict(
            otr=otr, ova=ova, ote=ote, mapper=mapper,
            btr=mapper.transform(d["X"][tr], otr),
            bva=mapper.transform(d["X"][va], ova),
            bte=mapper.transform(d["X"][te], ote))
    return _CACHE[key]


VARIANTS = {
    "full":      dict(use_proto=True, use_gate=True, use_miss_bias=True, use_coarse=True,
                      use_cons=True, use_proto_loss=True),
    "no_proto":  dict(use_proto=False, use_gate=False, use_miss_bias=True, use_coarse=True,
                      use_cons=True, use_proto_loss=False),
    "no_gate":   dict(use_proto=True, use_gate=False, use_miss_bias=True, use_coarse=True,
                      use_cons=True, use_proto_loss=True),
    "no_missb":  dict(use_proto=True, use_gate=True, use_miss_bias=False, use_coarse=True,
                      use_cons=True, use_proto_loss=True),
    "no_coarse": dict(use_proto=True, use_gate=True, use_miss_bias=True, use_coarse=False,
                      use_cons=True, use_proto_loss=True),
    "no_cons":   dict(use_proto=True, use_gate=True, use_miss_bias=True, use_coarse=True,
                      use_cons=False, use_proto_loss=True),
    "no_protl":  dict(use_proto=True, use_gate=True, use_miss_bias=True, use_coarse=True,
                      use_cons=True, use_proto_loss=False),
    "ft_lite":   dict(use_proto=False, use_gate=False, use_miss_bias=False, use_coarse=False,
                      use_cons=False, use_proto_loss=False),
}


def subsample(y, frac, seed):
    """Stratified index subset keeping ``frac`` of every class.

    Used by the label-efficiency experiment: the point is to shrink the number
    of *labels*, not to change their distribution, so the subset is drawn class
    by class rather than uniformly over the training pool.
    """
    if frac is None or frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.RandomState(1000 + seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = max(2, int(round(frac * len(idx))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def train_one(rate, protocol, seed, variant="full", cfg=None, keep_model=False, frac=None):
    d, (tr, va, te) = data()
    p = prepared(rate, protocol)
    sel = subsample(d["y"][tr], frac, seed)
    t0 = time.time()
    cfg = dict(cfg or {})
    cfg.setdefault("rand_miss", rate)      # missingness-consistent augmentation
    model, hist, best = T.train_map(
        p["btr"][sel], p["otr"][sel], d["y"][tr][sel], d["y_coarse"][tr][sel],
        p["bva"], p["ova"], d["y"][va], d["y_coarse"][va],
        p["mapper"], 11, 4, cfg=cfg, seed=seed, verbose=False, **VARIANTS[variant])
    fit_time = time.time() - t0
    t1 = time.time()
    proba = model.predict_proba(p["bte"], p["ote"])
    infer_time = time.time() - t1
    m = T.metrics(d["y"][te], proba)
    m.update(model="MA-PAN" if variant == "full" else variant, fit_time=fit_time,
             infer_time_per_sample=infer_time / proba.shape[0], n_params=model.n_params(),
             best_epoch=best, hist=hist[-1] if hist else None)
    return model, proba, m


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def job_main(job):
    rate, seed, method = job["rate"], job["seed"], job["method"]
    d, (tr, va, te) = data()
    p = prepared(rate, job.get("protocol", "mcar"))
    if method == "mapan":
        _, proba, m = train_one(rate, job.get("protocol", "mcar"), seed)
    elif method == "ft_lite":
        _, proba, m = train_one(rate, job.get("protocol", "mcar"), seed, variant="ft_lite")
    else:
        proba, fit_time = run_baseline(method, d["X"][tr], p["otr"], d["y"][tr],
                                       d["X"][te], p["ote"], d["kind"], seed=seed)
        m = T.metrics(d["y"][te], proba)
        m.update(model=method, fit_time=fit_time)
    m.update(exp="main", rate=rate, seed=seed, protocol=job.get("protocol", "mcar"),
             obs_ratio=float(p["ote"].mean()))
    return m


def job_ablation(job):
    rate, seed, variant = job["rate"], job["seed"], job["variant"]
    _, _, m = train_one(rate, "mcar", seed, variant=variant)
    m.update(exp="ablation", rate=rate, seed=seed, variant=variant)
    m.pop("hist", None)
    return m


def job_ablation06(job):
    """Same comparison as ``job_ablation`` but at 60% missingness.

    A retrieval prior can only show up when the observed evidence is thin, and
    30% is not thin: the encoder still sees roughly twenty of thirty-six fields.
    This job repeats the full-versus-prototype-free comparison at the highest
    reported rate, so that the claim "the memory pays when evidence is thin" is
    tested at the point where it should be true, not only where it is
    convenient.
    """
    rate, seed, variant = job["rate"], job["seed"], job["variant"]
    _, _, m = train_one(rate, "mcar", seed, variant=variant)
    m.update(exp="ablation06", rate=rate, seed=seed, variant=variant)
    m.pop("hist", None)
    return m


def job_pattern(job):
    rate, seed, protocol = job["rate"], job["seed"], job["protocol"]
    _, _, m = train_one(rate, protocol, seed)
    m.update(exp="pattern", rate=rate, seed=seed, protocol=protocol)
    m.pop("hist", None)
    return m


def job_detail(job):
    """Per-class metrics, confusion matrix, calibration and embeddings."""
    rate, seed = job["rate"], job["seed"]
    d, (tr, va, te) = data()
    model, proba, m = train_one(rate, "mcar", seed)
    p = prepared(rate, "mcar")
    pred = proba.argmax(axis=1)
    cm = np.zeros((11, 11), dtype=int)
    for t, q in zip(d["y"][te], pred):
        cm[t, q] += 1
    prf = T.prf_per_class(d["y"][te], proba, [c for c, _ in
                                              __import__("benchmark").CLASSES])
    # embeddings of the fused representation
    z = []
    for i in range(0, p["bte"].shape[0], 512):
        z.append(model.forward(p["bte"][i:i + 512], p["ote"][i:i + 512], training=False)["z"].data)
    z = np.concatenate(z, axis=0)
    np.savez_compressed(os.path.join(RESULTS, f"detail_r{int(rate*100)}.npz"),
                        cm=cm, proba=proba, y=d["y"][te], z=z)
    with open(os.path.join(RESULTS, f"detail_r{int(rate*100)}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(prf=prf, metrics={k: float(v) for k, v in m.items()
                                         if isinstance(v, (int, float))}), fh, indent=2)
    return dict(exp="detail", rate=rate, seed=seed, f1_macro=float(m["f1_macro"]))


def job_oneshot(job):
    """One model trained with randomised missing rates, evaluated at every rate."""
    d, (tr, va, te) = data()
    seed = job["seed"]
    rng = np.random.default_rng(seed)
    p_ref = prepared(0.0, "mcar")
    mapper = p_ref["mapper"]
    # curriculum: each sample draws its own hiding rate in [0, rho_max]
    otr = np.zeros_like(p_ref["otr"])
    from benchmark import impose_missing
    groups = [[j for j, n in enumerate(d["names"]) if T._layer_of(n) == g]
              for g in ("L3", "L4", "L7", "DV")]
    meta = dict(difficulty_idx=d["difficulty_idx"], groups=groups)
    rmax = job.get("rate_max", 0.6)
    for i in range(tr.size):
        r = rng.uniform(0, rmax)
        otr[i] = impose_missing(rng, d["X"][tr][i:i + 1], d["mask"][tr][i:i + 1], r, "mcar", meta)[0]
    btr = mapper.transform(d["X"][tr], otr)
    t0 = time.time()
    model, hist, best = T.train_map(btr, otr, d["y"][tr], d["y_coarse"][tr],
                                    prepared(0.0, "mcar")["bva"], prepared(0.0, "mcar")["ova"],
                                    d["y"][va], d["y_coarse"][va], mapper, 11, 4,
                                    cfg=dict(epochs=24, patience=8), seed=seed, verbose=False,
                                    **VARIANTS["full"])
    out = []
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        p = prepared(r, "mcar")
        pr = model.predict_proba(p["bte"], p["ote"])
        mm = T.metrics(d["y"][te], pr)
        mm.update(exp="oneshot", rate=r, seed=seed, fit_time=time.time() - t0)
        out.append(mm)
    return dict(exp="oneshot", seed=seed, points=out, model_params=model.n_params())


def job_budget(job):
    """Budget-aware progressive probing: accuracy as evidence is acquired.

    The policy starts from a fully hidden entity and acquires field *groups*: at
    each of ``n_step`` rounds it reveals enough fields to reach full evidence
    exactly at the last round, so every curve ends at the accuracy attainable
    with every obtainable field. Fields are chosen one at a time inside a round;
    the acquisition score is recomputed only at round boundaries, which bounds
    the cost of the experiment without changing the comparison between policies.
    """
    rate, seed = job["rate"], job["seed"]
    d, (tr, va, te) = data()
    model, proba, _ = train_one(rate, "mcar", seed)
    p = prepared(rate, "mcar")
    names = d["names"]
    # probing cost per field (relative units): network/transport are cheapest
    cost = np.array([1.0 if T._layer_of(n) in ("L3", "L4")
                     else (2.0 if T._layer_of(n) == "L7" else 3.0) for n in names])
    cost_full = float(cost.sum())

    rng = np.random.default_rng(seed)
    n_use = min(250, p["bte"].shape[0])
    sel = rng.choice(p["bte"].shape[0], size=n_use, replace=False)
    bins = p["bte"][sel].copy()
    obs = p["ote"][sel].copy()
    y = d["y"][te][sel]

    # fully hidden start: every field is set to its absence bin
    bins0 = bins.copy()
    for j in range(len(names)):
        bins0[~obs[:, j], j] = model.mapper.n_bins(j) - 1

    n_step = 9
    curves = {}
    for strat in ["random", "layer_order", "entropy", "cost_gain"]:
        cur_bins = bins0.copy()
        cur_obs = np.zeros(obs.shape, dtype=bool)
        accs, costs, order = [], [0.0], []
        spent = 0.0
        for step in range(n_step + 1):
            pr = model.predict_proba(cur_bins, cur_obs)
            accs.append(float((pr.argmax(1) == y).mean()))
            if step == n_step:
                break
            left = [j for j in range(len(names)) if not cur_obs[:, j].any()]
            if not left:
                break
            k = int(np.ceil(len(left) / float(n_step - step)))
            for _ in range(k):
                left = [j for j in range(len(names)) if not cur_obs[:, j].any()]
                if not left:
                    break
                if strat == "random":
                    pick = int(rng.choice(left))
                elif strat == "layer_order":
                    pick = min(left, key=lambda j: (cost[j], j))
                else:
                    pick = _acquisition(model, cur_bins, cur_obs, left, bins, cost, strat,
                                        full_obs=obs)[0]
                order.append(int(pick))
                spent += float(cost[pick])
                cur_obs[:, pick] = obs[:, pick]
                cur_bins[:, pick] = bins[:, pick]
            costs.append(spent)
        curves[strat] = dict(acc=accs, cost=costs, order=order)
    return dict(exp="budget", rate=rate, seed=seed, curves=curves,
                n=int(len(y)), cost_per_field=cost.tolist(), cost_full=cost_full)


def _acquisition(model, bins, obs, remaining, full_bins, cost, strat, full_obs=None):
    """Greedy selection of the next field to probe.

    ``entropy`` maximises the expected reduction of predictive entropy;
    ``cost_gain`` maximises the same gain per unit of probing cost.
    """
    base = model.predict_proba(bins, obs)
    base_h = -(base * np.log(base + 1e-9)).sum(axis=1).mean()
    best, best_score = None, -1e9
    for j in remaining:
        cand_bins = bins.copy()
        cand_obs = obs.copy()
        cand_obs[:, j] = full_obs[:, j] if full_obs is not None else True
        cand_bins[:, j] = full_bins[:, j]
        pr = model.predict_proba(cand_bins, cand_obs)
        gain = base_h + (pr * np.log(pr + 1e-9)).sum(axis=1).mean()
        s = gain if strat == "entropy" else gain / cost[j]
        if s > best_score:
            best, best_score = j, s
    return [best]


def job_lowdata(job):
    """Label-efficiency probe: same task, a fraction of the training labels.

    The full-data ablation (Section 6.3) shows the prototype branch is close to
    redundant when every label is available; the hypothesis it tests is that the
    branch pays for itself exactly where a retrieval prior should, i.e. when the
    classifier has too few examples to estimate a class-conditional boundary.
    """
    rate, seed, variant = job["rate"], job["seed"], job["variant"]
    frac = job["frac"]
    _, _, m = train_one(rate, job.get("protocol", "mcar"), seed, variant=variant, frac=frac)
    m.update(exp="lowdata", rate=rate, seed=seed, frac=frac,
             protocol=job.get("protocol", "mcar"), variant=variant)
    return m


JOBS = dict(main=job_main, ablation=job_ablation, pattern=job_pattern, detail=job_detail,
            oneshot=job_oneshot, budget=job_budget, lowdata=job_lowdata,
            ablation06=job_ablation06)


def job_key(job):
    return job["exp"] + "|" + "|".join(f"{k}={job[k]}" for k in sorted(job) if k != "exp")


def run(jobs, workers=10, out="results.jsonl"):
    from multiprocessing import Pool
    path = os.path.join(RESULTS, out)
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    # a job that raised is *not* done: keeping it in the resume
                    # set would silently drop it from the suite
                    if "job_key" in r and "error" not in r:
                        done.add(r["job_key"])
                except Exception:
                    pass
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(todo)} of {len(jobs)} jobs to run with {workers} workers", flush=True)
    t0 = time.time()
    # one job per child: a worker that finishes is replaced, so a slow leak in
    # any single job can never accumulate across jobs
    with Pool(workers, maxtasksperchild=1) as pool:
        for i, (job, res) in enumerate(pool.imap_unordered(_dispatch, [(j,) for j in todo]), 1):
            res["job_key"] = job_key(job)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
            tag = res.get("model") or res.get("variant") or res.get("protocol") or res.get("exp")
            rate = res.get("rate")
            f1 = res.get("f1_macro")
            print(f"[{i}/{len(todo)}] {res.get('exp')} {tag} r={rate} "
                  f"F1={f1 if f1 is None else round(f1,4)} ({time.time()-t0:.0f}s)", flush=True)
    print("done in %.1f min" % ((time.time() - t0) / 60))


def _dispatch(args):
    job = args[0]
    try:
        return job, JOBS[job["exp"]](job)
    except Exception:
        return job, dict(exp=job["exp"], error=traceback.format_exc())


def build_jobs(which, seeds=(0, 1, 2)):
    jobs = []
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    if "main" in which:
        for r in rates:
            for s in seeds:
                for m in ["mapan", "ft_lite", "svm", "rf", "hgb", "mlp", "knn_imp"]:
                    jobs.append(dict(exp="main", rate=r, seed=s, method=m))
    if "ablation" in which:
        for s in seeds:
            for v in VARIANTS:
                jobs.append(dict(exp="ablation", rate=0.3, seed=s, variant=v))
    if "pattern" in which:
        for s in seeds:
            for pr in ["mcar", "mar", "block", "mnar"]:
                jobs.append(dict(exp="pattern", rate=0.3, seed=s, protocol=pr))
    if "detail" in which:
        jobs.append(dict(exp="detail", rate=0.3, seed=0))
    if "oneshot" in which:
        for s in seeds:
            jobs.append(dict(exp="oneshot", seed=s, rate_max=0.6))
    if "budget" in which:
        jobs.append(dict(exp="budget", rate=0.3, seed=0))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main,ablation,pattern,detail,oneshot,budget")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()
    which = args.exp.split(",")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    run(build_jobs(which, seeds), workers=args.workers, out=args.out)
