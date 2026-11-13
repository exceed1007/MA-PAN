"""
MA-PAN: Missingness-Aware Prototype-Augmented Attention Network.

Pipeline
--------
1. Feature tokenisation.  Every fingerprint field becomes a token through a
   per-field embedding table.  A dedicated *absent* row encodes "not observed",
   so no imputation is performed at any stage.
2. Masked self-attention encoder.  Absent keys receive a learnable additive
   bias instead of being dropped, so the encoder can reason about *which*
   evidence is absent rather than only about the observed values.
3. Mutual cross-attention with a prototype memory.  The memory holds K
   prototypes per entity class -- a learned, compact form of the entity
   knowledge graph.  Sample-to-prototype attention retrieves supporting
   knowledge; prototype-to-sample attention measures how strongly each
   prototype is supported by the observed evidence.
4. Reliability-gated fusion.  A per-dimension gate balances local evidence
   against retrieved knowledge according to the available evidence.
5. Dual heads.  A linear head produces fine-grained logits, while a cosine
   head over class prototypes provides an interpretable, prototype-aligned
   decision; an auxiliary head predicts the four super-classes.

Training combines the fine cross entropy, the auxiliary coarse cross entropy,
a prototype-alignment term and a missingness-consistency self-distillation term
between a lightly and a heavily masked view of the same fingerprint.
"""

from __future__ import annotations

import numpy as np

import autograd as ag
from autograd import Tensor

NUM_BINS = 16
MISS_BIAS_INIT = -2.0


# --------------------------------------------------------------------------- #
# Discretisation
# --------------------------------------------------------------------------- #
class BinMapper:
    """Quantile discretisation for continuous fields; categorical fields keep codes."""

    def __init__(self, kind, ncat, num_bins: int = NUM_BINS):
        self.kind = np.asarray(kind)
        self.ncat = np.asarray(ncat)
        self.num_bins = num_bins
        self.edges: dict[int, np.ndarray] = {}

    def fit(self, X, mask):
        for j, k in enumerate(self.kind):
            if k == "num":
                col = X[mask[:, j], j]
                col = col[np.isfinite(col)]
                qs = np.linspace(0, 100, self.num_bins + 1)[1:-1]
                self.edges[j] = np.unique(np.percentile(col, qs))
        return self

    def n_bins(self, j: int) -> int:
        if self.kind[j] == "num":
            return len(self.edges[j]) + 1
        return int(self.ncat[j]) + 1          # last row == absent

    def transform(self, X, mask):
        n, m = X.shape
        out = np.zeros((n, m), dtype=np.int64)
        for j in range(m):
            if self.kind[j] == "num":
                b = np.searchsorted(self.edges[j], X[:, j], side="right")
                out[:, j] = np.where(mask[:, j], np.clip(b, 0, len(self.edges[j])), len(self.edges[j]))
            else:
                code = np.nan_to_num(X[:, j], nan=0.0).astype(np.int64)
                out[:, j] = np.where(mask[:, j], np.clip(code, 0, int(self.ncat[j]) - 1),
                                     int(self.ncat[j]))
        return out


def one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    o = np.zeros((idx.shape[0], n))
    o[np.arange(idx.shape[0]), idx] = 1.0
    return o


# --------------------------------------------------------------------------- #
# Optimiser / parameters
# --------------------------------------------------------------------------- #
class Param(Tensor):
    """A leaf tensor carrying its own first/second moment estimates."""

    def __init__(self, name: str, shape, rng, init: str = "glorot"):
        if init == "zeros":
            data = np.zeros(shape)
        elif init == "ones":
            data = np.ones(shape)
        else:
            lim = np.sqrt(6.0 / (shape[0] + shape[1])) if len(shape) == 2 else 0.02
            data = rng.uniform(-lim, lim, size=shape)
        super().__init__(data, requires_grad=True)
        self.name = name
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    @property
    def t(self) -> "Param":     # keeps `p.t` notation valid for leaf parameters
        return self


class AdamW:
    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.999), eps: float = 1e-8, wd: float = 1e-2):
        self.params, self.lr, self.eps, self.wd = params, lr, eps, wd
        self.b1, self.b2 = betas
        self.t = 0

    def step(self, lr: float | None = None):
        self.t += 1
        lr = self.lr if lr is None else lr
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            p.m = self.b1 * p.m + (1 - self.b1) * g
            p.v = self.b2 * p.v + (1 - self.b2) * g * g
            mh = p.m / (1 - self.b1 ** self.t)
            vh = p.v / (1 - self.b2 ** self.t)
            step = mh / (np.sqrt(vh) + self.eps)
            if len(p.shape) > 1:          # no weight decay on gains / biases
                step = step + self.wd * p.t.data
            p.t.data -= lr * step


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MAPAN:
    def __init__(self, mapper: BinMapper, n_classes: int, n_super: int = 4, d: int = 64,
                 heads: int = 4, layers: int = 2, K: int = 3, p_drop: float = 0.1,
                 seed: int = 0, use_proto: bool = True, use_gate: bool = True,
                 use_miss_bias: bool = True, use_coarse: bool = True):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.mapper = mapper
        self.m = len(mapper.kind)
        self.n_bins = [mapper.n_bins(j) for j in range(self.m)]
        self.n_classes, self.n_super = n_classes, n_super
        self.d, self.h, self.L, self.K = d, heads, layers, K
        self.p_drop = p_drop
        self.use_proto, self.use_gate = use_proto, use_gate
        self.use_miss_bias, self.use_coarse = use_miss_bias, use_coarse
        self.params: list[Param] = []

        def P(name, shape, init="glorot"):
            p = Param(name, shape, rng, init=init)
            self.params.append(p)
            return p

        # tokeniser: one shared table, per-field offsets (single gather op)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_bins)[:-1]]).astype(np.int64)
        self.absent_idx = np.array([self.n_bins[j] - 1 for j in range(self.m)], dtype=np.int64)
        self.E = P("E", (int(np.sum(self.n_bins)), d))
        # encoder
        self.Wq = [P(f"Wq{l}", (d, d)) for l in range(layers)]
        self.Wk = [P(f"Wk{l}", (d, d)) for l in range(layers)]
        self.Wv = [P(f"Wv{l}", (d, d)) for l in range(layers)]
        self.Wo = [P(f"Wo{l}", (d, d)) for l in range(layers)]
        self.ln1g = [P(f"ln1g{l}", (d,), "ones") for l in range(layers)]
        self.ln1b = [P(f"ln1b{l}", (d,), "zeros") for l in range(layers)]
        self.F1 = [P(f"F1_{l}", (d, 2 * d)) for l in range(layers)]
        self.F1b = [P(f"F1b{l}", (2 * d,), "zeros") for l in range(layers)]
        self.F2 = [P(f"F2_{l}", (2 * d, d)) for l in range(layers)]
        self.F2b = [P(f"F2b{l}", (d,), "zeros") for l in range(layers)]
        self.ln2g = [P(f"ln2g{l}", (d,), "ones") for l in range(layers)]
        self.ln2b = [P(f"ln2b{l}", (d,), "zeros") for l in range(layers)]
        self.miss_bias = P("miss_bias", (1,), "zeros")
        self.miss_bias.t.data[:] = MISS_BIAS_INIT
        self.cls = P("cls", (1, 1, d))
        # prototype memory + cross attention
        self.Pmem = P("prototypes", (n_classes * K, d))
        self.Cq, self.Ck, self.Cv, self.Co = (P(f"c_{n}", (d, d)) for n in "qkvo")
        self.Qq, self.Qk, self.Qv, self.Qo = (P(f"p_{n}", (d, d)) for n in "qkvo")
        # fusion
        self.Wkf = P("Wkf", (2 * d, d))
        self.bkf = P("bkf", (d,), "zeros")
        self.Wg = P("Wg", (2 * d, d))
        self.bg = P("bg", (d,), "zeros")
        self.Wzp = P("Wzp", (d, d))
        # heads
        self.lng = P("lng", (d,), "ones")
        self.lnb = P("lnb", (d,), "zeros")
        self.Wh = P("Wh", (d, n_classes))
        self.bh = P("bh", (n_classes,), "zeros")
        self.Wc = P("Wc", (d, n_super))
        self.bc = P("bc", (n_super,), "zeros")
        self.tau = P("tau", (1,), "zeros")
        self.tau.t.data[:] = 4.0

    # ------------------------------------------------------------------ ops
    def _mha(self, Hq: Tensor, Hk: Tensor, Hv: Tensor, Wq, Wk, Wv, Wo, bias=None) -> Tensor:
        B, nq, d = Hq.shape
        h, dk = self.h, d // self.h
        nk = Hk.shape[1]

        def split(t):
            return t.reshape((B, -1, h, dk)).transpose((0, 2, 1, 3))

        Q = split(Hq @ Wq)
        K = split(Hk @ Wk)
        V = split(Hv @ Wv)
        scores = Q @ K.transpose((0, 1, 3, 2)) * (1.0 / np.sqrt(dk))
        if bias is not None:
            if self.use_miss_bias:
                scores = scores + self.miss_bias.reshape((1, 1, 1, 1)) * Tensor(
                    np.broadcast_to(bias[:, None, :, :], (B, h, nq, nk)))
            else:
                scores = scores + Tensor(np.broadcast_to(bias[:, None, :, :], (B, h, nq, nk)))
        A = ag.softmax(scores, axis=-1)
        ctx = (A @ V).transpose((0, 2, 1, 3)).reshape((B, nq, d))
        return ctx @ Wo

    def _tokenise(self, bins, obs_eff):
        gidx = bins + self.offsets[None, :]
        return ag.gather_rows(self.E, gidx)

    def _encode(self, bins, obs, training: bool, extra_hide: float = 0.0, rng=None):
        rng = rng if rng is not None else self.rng
        B, m = bins.shape
        obs_eff = obs
        if extra_hide > 0:
            hide = (rng.random((B, m)) < extra_hide) & obs_eff
            obs_eff = obs_eff & ~hide
            bins = bins.copy()
            for j in range(m):
                absent = self.n_bins[j] - 1
                bins[~obs_eff[:, j], j] = absent
        H = self._tokenise(bins, obs_eff)
        # prepend the [CLS] token, which aggregates the whole fingerprint
        cls = ag.repeat_batch(self.cls.reshape((1, self.d)), B)
        H = ag.concat([cls, H], axis=1)
        H = ag.dropout(H, self.p_drop, training, rng)
        obs_eff = np.concatenate([np.ones((B, 1), dtype=bool), obs_eff], axis=1)

        bias = None
        if self.use_miss_bias:
            bias = np.broadcast_to((~obs_eff).astype(float)[:, None, :], (B, m + 1, m + 1)).copy()
        for l in range(self.L):
            Hn = ag.layer_norm(H, self.ln1g[l], self.ln1b[l])
            H = H + self._mha(Hn, Hn, Hn, self.Wq[l], self.Wk[l], self.Wv[l], self.Wo[l], bias)
            Hn = ag.layer_norm(H, self.ln2g[l], self.ln2b[l])
            ff = ag.gelu(Hn @ self.F1[l] + self.F1b[l]) @ self.F2[l] + self.F2b[l]
            H = H + ag.dropout(ff, self.p_drop, training, rng)
        return H, obs_eff, bias

    @staticmethod
    def _masked_mean(H: Tensor, obs) -> Tensor:
        w = Tensor(obs.astype(float))
        return (H * w[:, :, None]).sum(axis=1) / (w.sum(axis=1) + 1e-6)[:, None]

    # ------------------------------------------------------------- forward
    def forward(self, bins, obs, training: bool = False, extra_hide: float = 0.0, rng=None):
        B = bins.shape[0]
        H, obs_eff, bias = self._encode(bins, obs, training, extra_hide, rng)
        obs_feat = obs_eff[:, 1:]
        h_cls = H[:, 0]
        h_pool = 0.5 * (h_cls + self._masked_mean(H[:, 1:], obs_feat))

        if self.use_proto:
            prot = self.Pmem
            Pb = ag.repeat_batch(prot, B)
            bias_p = None
            if self.use_miss_bias:
                bias_p = np.broadcast_to((~obs_eff).astype(float)[:, None, :],
                                         (B, prot.shape[0], self.m + 1)).copy()
            a1 = self._mha(H, Pb, Pb, self.Cq, self.Ck, self.Cv, self.Co)            # B,m+1,d
            u1 = self._masked_mean(a1[:, 1:], obs_feat)
            a2 = self._mha(Pb, H, H, self.Qq, self.Qk, self.Qv, self.Qo, bias_p)     # B,CK,d
            u2 = a2.mean(axis=1)
            know = ag.tanh(ag.concat([u1, u2], axis=-1) @ self.Wkf + self.bkf)
            if self.use_gate:
                g = ag.sigmoid(ag.concat([h_pool, know], axis=-1) @ self.Wg + self.bg)
                z = g * h_pool + (1.0 - g) * know
            else:
                z = 0.5 * (h_pool + know)
        else:
            u1 = u2 = None
            z = h_pool

        zn = ag.layer_norm(z, self.lng, self.lnb)
        logits = zn @ self.Wh + self.bh
        coarse = zn @ self.Wc + self.bc if self.use_coarse else None

        sim = prot_hat = None
        if self.use_proto:
            zp = ag.layer_norm(z @ self.Wzp, self.lng, self.lnb)
            zp_hat = zp / ag.sqrt((zp * zp).sum(axis=-1, keepdims=True) + 1e-8)
            prot = self.Pmem.reshape((self.n_classes, self.K, self.d)).mean(axis=1)
            prot_hat = prot / ag.sqrt((prot * prot).sum(axis=-1, keepdims=True) + 1e-8)
            sim = zp_hat @ prot_hat.T
            logits = logits + self.tau.reshape((1, 1)) * sim
        return dict(logits=logits, coarse=coarse, z=z, sim=sim, h_pool=h_pool,
                    u1=u1, u2=u2, prot=prot_hat)

    # ---------------------------------------------------------------- loss
    def compute_loss(self, out, y, yc, wts, cons_out=None, temp: float = 2.0):
        ce = ag.cross_entropy(out["logits"], y)
        total = ce
        comp = {"ce": float(ce.data)}
        if self.use_coarse and out["coarse"] is not None and wts.get("coarse", 0) > 0:
            cce = ag.cross_entropy(out["coarse"], yc)
            total = total + wts["coarse"] * cce
            comp["cce"] = float(cce.data)
        if self.use_proto and wts.get("proto", 0) > 0 and out["sim"] is not None:
            lp = ag.cross_entropy(out["sim"] * temp, y)
            total = total + wts["proto"] * lp
            comp["proto"] = float(lp.data)
        if cons_out is not None and wts.get("cons", 0) > 0:
            kl = ag.kl_divergence(out["logits"], cons_out["logits"])
            total = total + wts["cons"] * kl
            comp["cons"] = float(kl.data)
        if wts.get("l2", 0) > 0:
            reg = None
            for p in self.params:
                if len(p.shape) == 2 and not p.name.startswith("E"):
                    reg = (p * p).sum() if reg is None else reg + (p * p).sum()
            total = total + wts["l2"] * reg
        return total, comp

    # ------------------------------------------------------------- predict
    def predict_proba(self, bins, obs, batch: int = 512) -> np.ndarray:
        outs = []
        with ag.no_grad():
            for i in range(0, bins.shape[0], batch):
                logits = self.forward(bins[i:i + batch], obs[i:i + batch],
                                      training=False)["logits"].data
                logits = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(logits)
                outs.append(e / e.sum(axis=1, keepdims=True))
        return np.concatenate(outs, axis=0)

    def n_params(self) -> int:
        return int(sum(p.data.size for p in self.params))


def count_macs(model: MAPAN) -> int:
    """Approximate multiply-accumulate count of a single-sample forward pass."""
    d, L, m = model.d, model.L, model.m
    f = sum(model.n_bins) * d
    f += L * (4 * m * d * d + 2 * m * m * d + 4 * m * d * d)
    if model.use_proto:
        ck = model.n_classes * model.K
        f += 4 * m * d * d + 2 * m * d * ck
        f += 4 * ck * d * d + 2 * ck * d * m
        f += 4 * d * d
    return int(f)
