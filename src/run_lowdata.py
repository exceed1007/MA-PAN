"""Label-efficiency probe for the prototype branch.

The component ablation answers "what happens when a component is removed with
every label available".  It cannot answer "is the component worth its
complexity", because a retrieval prior is by construction most useful when the
classifier has few labelled examples to estimate a class-conditional boundary
from.  This driver therefore repeats the comparison between the full model and
its prototype-free variant on stratified subsets of the training labels (10% and
25%), at the same 30% missingness, three seeds each.

Results go to ``results/lowdata.jsonl`` so that they never mix with the main
grid.
"""
import argparse

import experiments as E

RATE = 0.3
FRACS = (0.10, 0.25)
SEEDS = (0, 1, 2)
VARIANTS = ("full", "no_proto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [dict(exp="lowdata", rate=RATE, seed=s, frac=f, variant=v)
            for f in FRACS for s in SEEDS for v in VARIANTS]
    print(f"{len(jobs)} low-data cells, {args.workers} workers", flush=True)
    E.run(jobs, workers=args.workers, out="lowdata.jsonl")


if __name__ == "__main__":
    main()
