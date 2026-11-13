"""Finite-difference gradient checks for the autograd engine (run: python tests_grad.py)."""
import numpy as np
import autograd as ag
from autograd import Tensor

rng = np.random.default_rng(0)
EPS = 1e-6


def numeric_grad(f, x: Tensor, idx=None):
    """Central-difference gradient of the scalar f() w.r.t. x.data."""
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        old = x.data[mi]
        x.data[mi] = old + EPS
        fp = float(f().data)
        x.data[mi] = old - EPS
        fm = float(f().data)
        x.data[mi] = old
        g[mi] = (fp - fm) / (2 * EPS)
        it.iternext()
    return g


MAXERR = [0.0]
NCHECK = [0]


def check(name, f, params, tol=1e-5):
    for p in params:
        p.zero_grad()
    out = f()
    out.backward()
    ok = True
    for k, p in enumerate(params):
        num = numeric_grad(f, p)
        ana = p.grad
        err = np.abs(num - ana).max() / (np.abs(num).max() + 1e-12)
        MAXERR[0] = max(MAXERR[0], float(err))
        if err > tol:
            ok = False
            print(f"  [FAIL] {name} param{k} rel.err={err:.2e}")
    NCHECK[0] += 1
    print(("[ OK ] " if ok else "[FAIL] ") + name)
    return ok


def T(*shape, scale=1.0):
    return Tensor(rng.normal(size=shape) * scale, requires_grad=True)


all_ok = True

# elementwise / broadcasting
a, b = T(4, 5), T(4, 5)
all_ok &= check("add", lambda: (a + b).sum(), [a, b])
c = T(4, 1)
all_ok &= check("add_broadcast", lambda: (a + c).sum(), [a, c])
all_ok &= check("mul", lambda: (a * b).sum(), [a, b])
all_ok &= check("div", lambda: (a / (b * b + 1.0)).sum(), [a, b])
s = T(1)
all_ok &= check("scalar_mul", lambda: (a * s).sum(), [a, s])

# matmul
m1, m2 = T(3, 4), T(4, 5)
all_ok &= check("matmul2d", lambda: (m1 @ m2).sum(), [m1, m2])
bm, b2 = T(2, 3, 4), T(2, 4, 5)
all_ok &= check("matmul_batched", lambda: (bm @ b2).sum(), [bm, b2])
bm2, m3 = T(2, 3, 4), T(4, 5)
all_ok &= check("matmul_broadcast_rhs", lambda: (bm2 @ m3).sum(), [bm2, m3])

# shape ops
x = T(2, 3, 4)
all_ok &= check("transpose", lambda: (x.transpose((0, 2, 1)) * 3.0).sum(), [x])
all_ok &= check("reshape", lambda: (x.reshape((6, 4)) @ m3).sum(), [x, m3])
all_ok &= check("sum_axis", lambda: (a.sum(axis=1, keepdims=True) * 2.0).sum(), [a])
all_ok &= check("mean_axis", lambda: (a.mean(axis=0) ** 2).sum(), [a])
all_ok &= check("getitem", lambda: (a[1:3, ::2] ** 2).sum(), [a])

# nonlinear
all_ok &= check("exp", lambda: ag.exp(x).sum(), [x])
all_ok &= check("log", lambda: ag.log(ag.exp(x) + 1.0).sum(), [x])
all_ok &= check("sqrt", lambda: ag.sqrt(x * x + 1.0).sum(), [x])
all_ok &= check("tanh", lambda: ag.tanh(x).sum(), [x])
all_ok &= check("sigmoid", lambda: ag.sigmoid(x).sum(), [x])
all_ok &= check("relu", lambda: ag.relu(x).sum(), [x])
all_ok &= check("gelu", lambda: ag.gelu(x).sum(), [x])

# softmax family
all_ok &= check("softmax", lambda: (ag.softmax(x) ** 2).sum(), [x])
w = Tensor(rng.normal(size=(2, 3, 4)))
all_ok &= check("log_softmax", lambda: (ag.log_softmax(x) * w).sum(), [x])
w2 = Tensor(rng.normal(size=(4, 5)))
all_ok &= check("softmax_axis1", lambda: (ag.softmax(a, axis=1) * w2).sum(), [a])

# layernorm
g_, b_ = Tensor(np.ones(4), requires_grad=True), Tensor(np.zeros(4), requires_grad=True)
all_ok &= check("layernorm", lambda: (ag.layer_norm(x, g_, b_) ** 2).sum(), [x, g_, b_])

# concat / cosine / losses
y1, y2 = T(4, 3), T(4, 3)
all_ok &= check("concat", lambda: (ag.concat([y1, y2], axis=-1) ** 2).sum(), [y1, y2])
all_ok &= check("cosine", lambda: ag.cosine_similarity(y1, y2).sum(), [y1, y2])

lab = rng.integers(0, 3, size=4)
all_ok &= check("cross_entropy", lambda: ag.cross_entropy(y1, lab), [y1])
wgt = rng.random(4)
all_ok &= check("cross_entropy_weighted", lambda: ag.cross_entropy(y1, lab, wgt), [y1])
all_ok &= check("kl", lambda: ag.kl_divergence(y2, y1), [y1])

# dropout backward sanity (only masked entries receive gradient)
y1.zero_grad()
d = ag.dropout(y1, 0.5, True, np.random.default_rng(1))
d.sum().backward()
assert np.all(y1.grad[d.data == 0] == 0) and np.all(y1.grad[d.data != 0] != 0)
print("[ OK ] dropout")

print("\n%d checks, largest relative error %.2e" % (NCHECK[0], MAXERR[0]))

# the numbers are quoted in the report, so they are persisted rather than typed
import json
import os
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gradcheck.json")
os.makedirs(os.path.dirname(_out), exist_ok=True)
with open(_out, "w", encoding="utf-8") as _fh:
    json.dump(dict(checks=NCHECK[0], max_rel_err=MAXERR[0], passed=all_ok), _fh, indent=1)
print("\nALL PASSED" if all_ok else "\nSOME CHECKS FAILED")
