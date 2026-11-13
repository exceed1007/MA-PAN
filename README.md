# MA-PAN: Missingness-Aware Prototype-Augmented Attention Network

Reference implementation of MA-PAN and of the reproducible benchmark used in
the study of fine-grained cyberspace entity classification from **partially
observed device fingerprints**.

The setting: an IPv6 entity is described by 36 probe-derived fields, but
middleboxes filter probe classes, low-power terminals answer a subset of
requests and rate limiting suppresses responses, so a large, structured and
non-random subset of the fields is missing. MA-PAN treats absence as a
first-class token instead of imputing values:

1. **Absence-aware tokenisation** — every field becomes a token through a
   per-field embedding table with one dedicated *absent* row; nothing is imputed.
2. **Masked self-attention with a learned missing-key bias** — absent keys
   receive a learnable additive bias, so the encoder can reason about *which*
   evidence is missing, not only about the observed values.
3. **Prototype memory with mutual cross-attention** — a memory holds K
   prototypes per class; sample-to-prototype attention retrieves supporting
   knowledge, prototype-to-sample attention measures how strongly each
   prototype is supported by the observed evidence.
4. **Reliability-gated fusion** — a per-dimension gate balances local evidence
   against retrieved knowledge.
5. **Training objectives** — a coarse-to-fine loss, prototype alignment and
   missingness-consistent self-distillation; a cost-aware rule ranks fields for
   budgeted probing.

Everything runs on CPU only. The network is built on a small in-repository
reverse-mode autograd engine (`src/autograd.py`) verified by finite
differences; no deep-learning framework is required.

## Repository layout

```
data/
  benchmark.py        deterministic benchmark generator (36 fields, 11 classes,
                      4 missingness mechanisms, controlled rates 10-60%)
  parse_nmap_osdb.py  projects the public Nmap nmap-os-db corpus onto the
                      36-field schema (an alternative, real-data path)
  mini_osdb           small Nmap OS-fingerprint excerpt used with --source nmap
  benchmark.npz       the generated benchmark (regenerable bit for bit)
  feature_spec.json   field names, types, layers and probing costs
src/
  autograd.py         reverse-mode autograd engine on NumPy
  model.py            MA-PAN architecture
  train.py            data handling, training loop, metrics
  baselines.py        classical baselines (HGB, RF, SVM, MLP, KNN/iterative
                      imputation pipelines)
  experiments.py      experiment definitions and job runner (main grid,
                      ablation, missingness patterns, fine-grained detail,
                      universal mode, budgeted acquisition)
  run_lowdata.py      label-efficiency probe of the prototype branch
  run_ablate06.py     full vs. prototype-free at the 60% operating point
  calib.py            reliability curves of the reference methods
  infer_time.py       per-entity inference-time measurements
  tests_grad.py       finite-difference gradient checks
results/              archived outputs of a full run (JSONL/JSON/NPZ)
```

## Requirements

Python 3.10+ with

```
numpy>=1.26
scikit-learn>=1.4
```

## Quick start

```bash
# 1. verify the autograd engine (fast, pure NumPy)
cd src && python tests_grad.py

# 2. the benchmark ships committed; to regenerate it from scratch:
cd ../data && python benchmark.py --out benchmark.npz

# 3. main suite: MA-PAN at every rate, component ablation, missingness
#    patterns, fine-grained detail, universal mode, budgeted acquisition
cd ../src && python experiments.py                # appends to results/results.jsonl

# 4. auxiliary experiments
python run_lowdata.py        # label-efficiency probe      -> results/lowdata.jsonl
python run_ablate06.py       # prototype ablation at 60%   -> results/ablate06.jsonl
python calib.py              # reliability curves          -> results/calib_baselines.npz
python infer_time.py         # inference timings           -> results/infer_time.json
```

Every run appends JSONL records keyed by an experiment/seed/rate/variant tuple,
so interrupted suites resume automatically: jobs whose key is already present
in the output file are skipped. `results/` contains the archived outputs of a
complete run for comparison.

## The Nmap path

Besides the synthetic generator, `data/parse_nmap_osdb.py` projects the public
Nmap OS-fingerprint corpus onto the 36-field schema:

```bash
cd data
python parse_nmap_osdb.py --db nmap-os-db --out nmap_os.npz   # full corpus
python parse_nmap_osdb.py --db mini_osdb  --out nmap_os.npz   # small excerpt
python benchmark.py --source nmap --nmap-db nmap_os.npz --out benchmark.npz
```

## Notes

- Determinism: the same seed reproduces the benchmark matrix and every number
  bit for bit; timings are hardware-dependent, all other results are not.
- Default training is deliberately small (32-dim encoder, single CPU core);
  a run of the full main suite takes hours rather than days.
