"""
A compact reverse-mode automatic differentiation engine on top of NumPy.

The experiments were executed in an environment without a deep-learning
framework available, so the whole network is implemented on top of this engine.
The engine intentionally supports only the primitive operations that MA-PAN
needs; every primitive is verified by finite differences (see tests_grad.py).
"""

from __future__ import annotations

import os

import numpy as np

# float32 keeps the experiments affordable on CPU; the gradient checks run in
# float64 (the default) so that finite differences remain meaningful.
DTYPE = np.float32 if os.environ.get("MAPAN_DTYPE", "f64") == "f32" else np.float64


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce a broadcasted gradient back to the shape of the operand."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _noop():
    return None


# When this flag is set no graph is built at all.  Inference and the periodic
# validation pass need values only, and building a graph for them costs both
# time and memory: every intermediate keeps its buffer alive until the cycle
# collector happens to run, because a node and its backward closure reference
# each other.
NO_GRAD = False


class _NoGrad:
    def __enter__(self):
        global NO_GRAD
        self._prev = NO_GRAD
        NO_GRAD = True
        return self

    def __exit__(self, *exc):
        global NO_GRAD
        NO_GRAD = self._prev
        return False


def no_grad():
    """Context manager that disables graph construction."""
    return _NoGrad()


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        if NO_GRAD:
            requires_grad = False
        self.requires_grad = requires_grad
        self._backward = _noop
        self._prev = _children if requires_grad else ()
        self._op = _op

    # ---------------------------------------------------------------- helpers
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def __repr__(self):  # pragma: no cover
        return f"Tensor(shape={self.data.shape}, op={self._op})"

    def backward(self, grad=None):
        """Back-propagate from this (scalar or seeded) tensor."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() without grad requires a scalar output")
            grad = np.ones_like(self.data)
        topo, visited = [], set()

        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = np.asarray(grad, dtype=DTYPE)
        for node in reversed(topo):
            if node.grad is not None:
                node._backward()
        # the graph is cyclic (closures capture their inputs); break it so that
        # reference counting frees the intermediates immediately
        for node in topo:
            node._prev = ()
            node._backward = _noop

    # ------------------------------------------------------------------ ops
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "add")

        def _backward():
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "mul")

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad * other.data, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad * self.data, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad,
                     (self, other), "div")

        def _backward():
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad / other.data, self.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Batched / mixed matmul: (B,n,k)@(B,k,m) or (B,n,k)@(k,m) or (n,k)@(k,m)."""
        assert isinstance(other, Tensor)
        a, b = self.data, other.data
        out_data = a @ b
        out = Tensor(out_data, self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            g = out.grad
            if a.ndim == b.ndim and a.ndim >= 3:
                if self.requires_grad:
                    _acc(self, g @ np.swapaxes(b, -1, -2))
                if other.requires_grad:
                    _acc(other, np.swapaxes(a, -1, -2) @ g)
            elif a.ndim == 3 and b.ndim == 2:
                if self.requires_grad:
                    _acc(self, g @ b.T)
                if other.requires_grad:
                    _acc(other, a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1]))
            elif a.ndim == 2 and b.ndim == 2:
                if self.requires_grad:
                    _acc(self, g @ b.T)
                if other.requires_grad:
                    _acc(other, a.T @ g)
            else:
                raise NotImplementedError(f"matmul {a.shape} @ {b.shape}")

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,), "transpose")
        inv = tuple(np.argsort(axes))

        def _backward():
            _acc(self, out.grad.transpose(inv))

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose(tuple(range(self.data.ndim))[::-1])

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward():
            _acc(self, out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad,
                     (self,), "sum")
        in_shape = self.shape

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _acc(self, np.broadcast_to(g, in_shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            g[idx] += out.grad
            _acc(self, g)

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert np.isscalar(p)
        out = Tensor(self.data ** p, self.requires_grad, (self,), "pow")

        def _backward():
            _acc(self, out.grad * p * self.data ** (p - 1))

        out._backward = _backward
        return out


# --------------------------------------------------------------------- unary
def exp(t: Tensor) -> Tensor:
    out = Tensor(np.exp(t.data), t.requires_grad, (t,), "exp")
    out._backward = lambda: _acc(t, out.grad * out.data)
    return out


def log(t: Tensor) -> Tensor:
    out = Tensor(np.log(t.data), t.requires_grad, (t,), "log")
    out._backward = lambda: _acc(t, out.grad / t.data)
    return out


def sqrt(t: Tensor) -> Tensor:
    out = Tensor(np.sqrt(t.data), t.requires_grad, (t,), "sqrt")
    out._backward = lambda: _acc(t, out.grad * 0.5 / out.data)
    return out


def tanh(t: Tensor) -> Tensor:
    out = Tensor(np.tanh(t.data), t.requires_grad, (t,), "tanh")
    out._backward = lambda: _acc(t, out.grad * (1.0 - out.data ** 2))
    return out


def sigmoid(t: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-t.data))
    out = Tensor(s, t.requires_grad, (t,), "sigmoid")
    out._backward = lambda: _acc(t, out.grad * s * (1.0 - s))
    return out


def relu(t: Tensor) -> Tensor:
    out = Tensor(np.maximum(t.data, 0.0), t.requires_grad, (t,), "relu")
    out._backward = lambda: _acc(t, out.grad * (t.data > 0))
    return out


def gelu(t: Tensor) -> Tensor:
    c = np.sqrt(2.0 / np.pi)
    x = t.data
    x2 = x * x
    inner = c * (x + 0.044715 * x2 * x)
    th = np.tanh(inner)
    out = Tensor(0.5 * x * (1.0 + th), t.requires_grad, (t,), "gelu")
    d = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * c * (1.0 + 3 * 0.044715 * x2)
    out._backward = lambda: _acc(t, out.grad * d)
    return out


def _acc(t: Tensor, g: np.ndarray):
    if t.grad is None:
        t.grad = g
    else:
        np.add(t.grad, g, out=t.grad)


# ------------------------------------------------------------------ reductions
def softmax(t: Tensor, axis: int = -1) -> Tensor:
    z = t.data - t.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, t.requires_grad, (t,), "softmax")

    def _backward():
        g = out.grad
        # dL/dz = s * (g - sum(g*s))
        gs = (g * s).sum(axis=axis, keepdims=True)
        _acc(t, s * (g - gs))

    out._backward = _backward
    return out


def log_softmax(t: Tensor, axis: int = -1) -> Tensor:
    z = t.data - t.data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    out = Tensor(z - lse, t.requires_grad, (t,), "log_softmax")

    def _backward():
        g = out.grad
        s = np.exp(out.data)
        _acc(t, g - s * g.sum(axis=axis, keepdims=True))

    out._backward = _backward
    return out


def layer_norm(t: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    mu = t.data.mean(axis=-1, keepdims=True)
    var = t.data.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (t.data - mu) * inv
    out = Tensor(xhat * gamma.data + beta.data, True, (t, gamma, beta), "layernorm")

    def _backward():
        g = out.grad
        n = t.data.shape[-1]
        _acc(gamma, (g * xhat).reshape(-1, n).sum(axis=0))
        _acc(beta, g.reshape(-1, n).sum(axis=0))
        gx = g * gamma.data
        _acc(t, inv * (gx - gx.mean(axis=-1, keepdims=True)
                       - xhat * (gx * xhat).mean(axis=-1, keepdims=True)))

    out._backward = _backward
    return out


def dropout(t: Tensor, p: float, training: bool, rng: np.random.Generator) -> Tensor:
    if not training or p <= 0.0:
        return t
    mask = (rng.random(t.shape) >= p) / (1.0 - p)
    out = Tensor(t.data * mask, t.requires_grad, (t,), "dropout")
    out._backward = lambda: _acc(t, out.grad * mask)
    return out


def cross_entropy(logits: Tensor, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Tensor:
    """Mean (optionally weighted) cross entropy; y is an integer label vector."""
    logp = log_softmax(logits, axis=-1)
    n = logits.shape[0]
    picked = logp.data[np.arange(n), y]
    if sample_weight is None:
        loss_val = -picked.mean()
        weights = np.full(n, 1.0 / n)
    else:
        weights = sample_weight / sample_weight.sum()
        loss_val = -(picked * weights).sum()
    out = Tensor(loss_val, True, (logits,), "xent")

    def _backward():
        # dL/dz = w * (softmax(z) - onehot(y))
        g = np.exp(logp.data)
        g[np.arange(n), y] -= 1.0
        _acc(logits, out.grad * (g * weights[:, None]))

    out._backward = _backward
    return out


def kl_divergence(logits_p: Tensor, logits_q: Tensor) -> Tensor:
    """KL(softmax(p) || softmax(q)) averaged over the batch.

    The first argument acts as the (stop-gradient) teacher distribution, so the
    gradient is propagated only into ``logits_q``.
    """
    lp = log_softmax(logits_p, axis=-1)
    lq = log_softmax(logits_q, axis=-1)
    p = np.exp(lp.data)
    val = (p * (lp.data - lq.data)).sum(axis=-1).mean()
    out = Tensor(val, True, (logits_q,), "kl")
    n = p.shape[0]

    def _backward():
        _acc(logits_q, out.grad * (np.exp(lq.data) - p) / n)

    out._backward = _backward
    return out


def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Row-wise cosine similarity between two 2-D tensors of equal shape."""
    na = np.sqrt((a.data ** 2).sum(axis=-1, keepdims=True)) + eps
    nb = np.sqrt((b.data ** 2).sum(axis=-1, keepdims=True)) + eps
    an, bn = a.data / na, b.data / nb
    sim = (an * bn).sum(axis=-1, keepdims=True)
    out = Tensor(sim, a.requires_grad or b.requires_grad, (a, b), "cos")

    def _backward():
        g = out.grad
        # d(cos)/da = b/(|a||b|) - cos * a/|a|^2
        if a.requires_grad:
            _acc(a, g * (bn / na - sim * an / na))
        if b.requires_grad:
            _acc(b, g * (an / nb - sim * bn / nb))

    out._backward = _backward
    return out


def gather_rows(table: Tensor, idx: np.ndarray) -> Tensor:
    """Row gather: table (S,d), idx (B,m) -> (B,m,d). Backward is a scatter-add."""
    out = Tensor(table.data[idx], table.requires_grad, (table,), "gather")

    def _backward():
        if table.grad is None:
            table.grad = np.zeros_like(table.data)
        np.add.at(table.grad, idx.reshape(-1), out.grad.reshape(-1, table.shape[1]))

    out._backward = _backward
    return out


def repeat_batch(t: Tensor, batch: int) -> Tensor:
    """Broadcast a 2-D tensor to (batch, n, k); gradient is summed over the batch."""
    out = Tensor(np.repeat(t.data[None], batch, axis=0), t.requires_grad, (t,), "repeat_batch")

    def _backward():
        _acc(t, out.grad.sum(axis=0))

    out._backward = _backward
    return out


def concat(tensors, axis: int = -1) -> Tensor:
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, any(t.requires_grad for t in tensors), tuple(tensors), "concat")
    sizes = np.cumsum([t.shape[axis] for t in tensors])[:-1]

    def _backward():
        parts = np.split(out.grad, sizes, axis=axis)
        for t, p in zip(tensors, parts):
            if t.requires_grad:
                _acc(t, p)

    out._backward = _backward
    return out
