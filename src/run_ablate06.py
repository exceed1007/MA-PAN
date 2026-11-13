"""Does the prototype branch pay off when the evidence is thin?

The component ablation is run at 30% additional missingness, where the encoder
still observes about two thirds of the fields and a linear boundary over the
encoder output is already close to the best achievable.  The prototype branch
is a prior over class-conditional evidence, so its natural operating point is
the other end of the curve.  This driver compares the full model with its
prototype-free variant at 60% missingness, three seeds each, and appends to
``results/ablate06.jsonl``.
"""
import argparse

import experiments as E

VARIANTS = ("full", "no_proto")
SEEDS = (0, 1, 2)
RATE = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [dict(exp="ablation06", rate=RATE, seed=s, variant=v)
            for s in SEEDS for v in VARIANTS]
    print(f"{len(jobs)} high-missingness cells, {args.workers} workers", flush=True)
    E.run(jobs, workers=args.workers, out="ablate06.jsonl")


if __name__ == "__main__":
    main()
