"""Shared pieces of the fault-diagnosis experiments: model-free quality checks, per-sensor model scores, feature rows for the fault head."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as Fn
from scipy.signal import medfilt, hilbert
from evalcommon import *


def qc(X):
    """(n, 4, 5) model-free quality scores; larger = more suspicious."""
    n = len(X); out = np.zeros((n, 4, 5), np.float32); ax = np.abs(X); mx = ax.max(-1)
    out[..., 0] = (mx == 0)
    out[..., 1] = (ax >= 0.995 * mx[..., None]).mean(-1) * (mx > 0)
    r = X - medfilt(X.reshape(-1, X.shape[-1]), [1, 5]).reshape(X.shape); p995 = np.quantile(ax, 0.995, axis=-1, keepdims=True) + 1e-12
    out[..., 2] = np.log1p(((np.abs(r) > 0.6 * ax) & (ax > 2.5 * p995)).sum(-1))
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2; f = np.fft.rfftfreq(X.shape[-1], 1e-4)
    out[..., 3] = np.log10(P[..., f < 3].sum(-1) / (P.sum(-1) + 1e-20) + 1e-12)
    env = np.log(np.abs(hilbert(X, axis=-1)).reshape(n, 4, 1000, 10).mean(-1) + 1e-12)
    env = (env - env.mean(-1, keepdims=True)) / (env.std(-1, keepdims=True) + 1e-9); E = np.fft.rfft(env, 2048, axis=-1); L = 300
    C = np.zeros((n, 4, 4), np.float32)
    for i in range(4):
        for j in range(i + 1, 4):
            cc = np.fft.irfft(E[:, i] * np.conj(E[:, j]), 2048, axis=-1) / 1000
            C[:, i, j] = C[:, j, i] = np.concatenate([cc[:, :L], cc[:, -L:]], 1).max(1)
    out[..., 4] = -np.stack([np.median(np.delete(C[:, s], s, 1), 1) for s in range(4)], 1)
    out[mx == 0, 1:] = 0
    return out


def js(p, q):
    m_ = 0.5 * (p + q); kl = lambda a, b: (a * (torch.log(a + 1e-9) - torch.log(b + 1e-9))).sum(-1)
    return 0.5 * kl(p, m_) + 0.5 * kl(q, m_)


@torch.no_grad()
def content(m, x, f, alive):
    B, S, _ = x.shape; emb = torch.zeros(B * S, 256, device=dev); fl = alive.reshape(-1)
    if fl.any(): emb[fl] = m.encode(x.reshape(B * S, -1)[fl]).float()
    return m.tok(torch.cat([emb.reshape(B, S, 256), m.desc(f.float())], -1)).float()


@torch.no_grad()
def fuse(m, c, alive, reset):
    """Class probabilities, attention-pooling weights and (ReSeT) CSP residuals for a given alive mask, from pre-rank tokens c."""
    a = alive.clone(); a[a.sum(1) == 0, 0] = True; B, S, _ = c.shape; t = c + m.rank.weight[:S][None]; e = None
    if reset:
        bias = c.new_zeros(B, S); na = a.sum(1, keepdim=True)
        if m.use_csp:
            e = 1 - Fn.cosine_similarity(m.predict_tokens(c, a).float(), c, dim=-1)
            if m.use_gate:
                ed = e * a; z = ed / (ed.sum(1, keepdim=True) / na + 1e-6) - 1; bias = -Fn.softplus(m.gate_a) * z * a * (na > 1)
        key = bias.masked_fill(~a, float("-inf")); am = key[:, None, None, :].expand(B, m.heads, S, S).reshape(B * m.heads, S, S)
        u = m.fuse(t, mask=am); w = (m.pool(u).squeeze(-1) + key).softmax(1)
    else:
        u = m.fuse(t, src_key_padding_mask=~a); w = m.pool(u).squeeze(-1).masked_fill(~a, float("-inf")).softmax(1)
    z_ = m.classifer(torch.cat([(w[..., None] * u).sum(1), (u * a[..., None]).sum(1) / a.sum(1, keepdim=True)], 1))
    return z_.softmax(1), w, e


@torch.no_grad()
def scores(ms, a, X, F, drop=None, bs=96):
    """Model scores (n, 4, k) averaged over seeds [disagree, outlier, lowatt, (csp)], ensemble probabilities, and ensemble
    probabilities with sensor drop[j] removed (drop = -1 keeps all)."""
    reset = a.get("arch") == "reset"; n = len(X); SC, P, PD = [], [], []
    for m in ms:
        sc, p_, pd_ = [], [], []
        for i in range(0, n, bs):
            x = torch.tensor(X[i:i + bs], device=dev); f = torch.tensor(F[i:i + bs], device=dev); alive = x.abs().amax(-1) > 0
            c = content(m, x, f, alive); p, w, e = fuse(m, c, alive, reset); cols = []
            dis = torch.zeros_like(w); out = torch.zeros_like(w)
            for s in range(4):
                only = torch.zeros_like(alive); only[:, s] = alive[:, s]; wo = alive.clone(); wo[:, s] = False
                dis[:, s] = js(fuse(m, c, only, reset)[0], fuse(m, c, wo, reset)[0])
                o = wo.float()[..., None]; out[:, s] = (c[:, s] - (c * o).sum(1) / o.sum(1).clamp(min=1)).norm(dim=-1)
            cols = [dis, out, -w * alive.sum(1, keepdim=True)] + ([e] if e is not None else [])
            sc.append(torch.stack(cols, -1).cpu().numpy()); p_.append(p.cpu().numpy())
            if drop is not None:
                d = torch.tensor(drop[i:i + bs], device=dev); al2 = alive.clone(); hit = d >= 0; al2[hit, d[hit]] = False
                pd_.append(fuse(m, c, al2, reset)[0].cpu().numpy())
        SC.append(np.concatenate(sc)); P.append(np.concatenate(p_))
        if drop is not None: PD.append(np.concatenate(pd_))
    return np.mean(SC, 0), np.mean(P, 0), (np.mean(PD, 0) if PD else None)


def feats(Q, S):
    """Per-sensor feature rows from qc (n,4,5) and model scores (n,4,k): raw values and contrast with the other sensors of the event."""
    A = np.concatenate([Q, S], -1); other = (A.sum(1, keepdims=True) - A) / 3.0
    return np.concatenate([A, A - other], -1)                      # (n, 4, 2*(5+k))


def cols(nq, k, which):
    base = dict(qc=list(range(5)), model=list(range(5, 5 + k)), all=list(range(5 + k)))[which] if which in ("qc", "model", "all") else [5 + which]
    return base + [b + 5 + k for b in base]


