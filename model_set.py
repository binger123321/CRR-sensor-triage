"""SensorSet-SincNet: the multi-stream network rebuilt as a model over a SET of sensors.

Each sensor is encoded on its own by shared weights (Sinc front end + MPA + BiLSTM, multi-resolution spectrogram CNN,
and an MLP on its 47 descriptors). The resulting sensor tokens (plus a learned embedding of the sensor's SNR rank)
are fused by a small transformer whose attention ignores dead sensors, and pooled by masked attention.

Why: in the four-channel network a dead sensor is an all-zero input that still flows into every fused feature,
so the output changes in ways never seen in training. Here a dead sensor is removed from the set: the prediction
is exactly what the network would give if that sensor had never been installed. Dead sensors are detected from the
data itself (a channel that is identically zero), so no external status flag is needed; `alive` can also be given.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.ippa import AAP, AMP
from frontends import FRONTENDS
from model_plus import SpecBranch


class SensorSetSincNet(nn.Module):
    def __init__(self, n_desc=47, n_cls=3, d=128, layers=2, heads=4, drop=0.3, max_sensors=4, frontend="sinc"):
        super().__init__()
        self.sincnet = nn.Sequential(FRONTENDS[frontend](), nn.MaxPool1d(3), nn.LayerNorm(3250), nn.LeakyReLU(0.2))
        self.feature_extraction1 = nn.Sequential(
            nn.Conv1d(40, 64, 5, stride=1, padding=0), nn.MaxPool1d(3), nn.LayerNorm(1082), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, stride=1, padding=1), nn.MaxPool1d(3), nn.LayerNorm(360), nn.LeakyReLU(0.2))
        self.aap = AAP(in_channel=64, channel=64, reduction=16)
        self.amp = AMP(in_channel=64, channel=64, reduction=16)
        self.proj = nn.Sequential(nn.Conv1d(64, 128, 1), nn.MaxPool1d(3), nn.LayerNorm(120), nn.LeakyReLU(0.2))
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.spec = SpecBranch(64)                                         # -> 128 per sensor
        self.n_desc = n_desc
        if n_desc:
            self.desc = nn.Sequential(nn.Linear(n_desc, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.GELU())
        self.tok = nn.Sequential(nn.Linear(128 + 128 + (64 if n_desc else 0), d), nn.LayerNorm(d))
        self.rank = nn.Embedding(max_sensors, d)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.pool = nn.Linear(d, 1)
        self.drop = nn.Dropout(drop)
        self.classifer = nn.Linear(2 * d, n_cls)

    def encode(self, xs):                                                  # xs: (N, 10000) live sensors only
        h = self.amp(self.aap(self.feature_extraction1(self.sincnet(xs))))   # (N, 64, 360)
        o, _ = self.lstm(self.proj(h).transpose(1, 2).float())            # (N, 120, 128)
        return torch.cat([0.5 * (o.mean(1) + o.amax(1)), self.spec(xs).float()], 1)   # (N, 256)

    def forward(self, x, f=None, alive=None):                              # x: (B, S, 10000); f: (B, S, 47)
        B, S, _ = x.shape
        if alive is None: alive = x.abs().amax(-1) > 0                    # a dead sensor is an identically zero channel
        alive = alive.clone(); alive[alive.sum(1) == 0, 0] = True          # guard: keep at least one token
        flat = alive.reshape(-1)
        emb = x.new_zeros(B * S, 256, dtype=torch.float32)
        emb[flat] = self.encode(x.reshape(B * S, -1)[flat])                # dead sensors never pass through the encoders
        parts = [emb.reshape(B, S, 256)]
        if self.n_desc: parts.append(self.desc(f.float()))
        t = self.tok(torch.cat(parts, -1)) + self.rank.weight[:S][None]
        pad = ~alive
        t = self.fuse(t, src_key_padding_mask=pad)
        w = self.pool(t).squeeze(-1).masked_fill(pad, float("-inf")).softmax(1)          # (B, S)
        att = (w[..., None] * t).sum(1)
        mean = (t * alive[..., None]).sum(1) / alive.sum(1, keepdim=True)
        return self.classifer(self.drop(torch.cat([att, mean], 1)))


_FASTPATH_OFF = hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled")
if _FASTPATH_OFF: torch.backends.mha.set_fastpath_enabled(False)   # the inference fast path mishandles additive (float) attention masks


class ReSeT(SensorSetSincNet):
    """Reliability-aware Sensor-set Transformer.

    Adds two things to SensorSetSincNet:
      CSP  cross-sensor prediction: a one-layer transformer predicts the token of each sensor from the tokens of the other
           live sensors (the sensor's own token is replaced by a learned mask token). The cosine distance e_i between the
           predicted and the observed token says how well sensor i agrees with the rest of the array. It is trained without
           labels of any kind (target = stop-gradient of the observed token).
      RGA  reliability-gated attention: the relative residual z_i = e_i / mean_j e_j - 1 is turned into an additive bias
           -softplus(a) * z_i on the attention logits of the fusion layers (as a key bias) and of the pooling, so a sensor
           that the others cannot explain is listened to less. a is learned from the classification loss alone.
    Optional latent imputation: the token of a dead sensor is replaced by its prediction instead of being removed.
    forward(x, f) returns logits; forward(x, f, aux=True) also returns the residuals, predictions and tokens.
    """
    def __init__(self, n_desc=47, csp=True, gate=True, impute=False, d=128, heads=4, **kw):
        super().__init__(n_desc=n_desc, d=d, heads=heads, **kw)
        self.use_csp, self.use_gate, self.use_impute, self.heads = csp, gate and csp, impute and csp, heads
        if csp:
            self.mask_tok = nn.Parameter(torch.zeros(d)); nn.init.normal_(self.mask_tok, std=0.02)
            enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.pred = nn.TransformerEncoder(enc, 1, enable_nested_tensor=False)
            self.pred_out = nn.Linear(d, d)
            self.gate_a = nn.Parameter(torch.tensor(0.5)); self.imp_b = nn.Parameter(torch.tensor(1.0))

    def content(self, x, f, alive):
        B, S, _ = x.shape; flat = alive.reshape(-1)
        emb = x.new_zeros(B * S, 256, dtype=torch.float32)
        emb[flat] = self.encode(x.reshape(B * S, -1)[flat])
        parts = [emb.reshape(B, S, 256)]
        if self.n_desc: parts.append(self.desc(f.float()))
        return self.tok(torch.cat(parts, -1))                                  # (B, S, d), no rank embedding yet

    def predict_tokens(self, c, alive):
        """p[:, i] = prediction of sensor i's token from the other live sensors."""
        B, S, d = c.shape; r = self.rank.weight[:S]
        eye = torch.eye(S, dtype=torch.bool, device=c.device)
        t = (c + r[None])[:, None].expand(B, S, S, d).clone()
        t[:, eye] = (self.mask_tok[None] + r).to(t.dtype)
        pad = (~alive)[:, None].expand(B, S, S).clone(); pad[:, eye] = False
        out = self.pred(t.reshape(B * S, S, d), src_key_padding_mask=pad.reshape(B * S, S)).reshape(B, S, S, d)
        return self.pred_out(out[:, eye])                                      # (B, S, d)

    def forward(self, x, f=None, alive=None, aux=False):
        B, S, _ = x.shape
        if alive is None: alive = x.abs().amax(-1) > 0
        alive = alive.clone(); alive[alive.sum(1) == 0, 0] = True
        c = self.content(x, f, alive).float()
        bias = c.new_zeros(B, S); e = p = None; used = alive
        if self.use_csp:
            p = self.predict_tokens(c, alive).float()
            e = 1 - F.cosine_similarity(p, c.detach(), dim=-1)                 # (B, S); meaningful for live sensors
            n_alive = alive.sum(1, keepdim=True)
            if self.use_gate:
                ed = e.detach() * alive
                z = ed / (ed.sum(1, keepdim=True) / n_alive + 1e-6) - 1
                bias = -F.softplus(self.gate_a) * z * alive * (n_alive > 1)
            if self.use_impute:
                dead = ~alive & (n_alive > 0)
                c = torch.where(dead[..., None], p, c); used = alive | dead
                bias = torch.where(dead, -F.softplus(self.imp_b).expand_as(bias), bias)
        t = c + self.rank.weight[:S][None]
        key = bias.masked_fill(~used, float("-inf"))                           # additive key bias; -inf removes a sensor
        am = key[:, None, None, :].expand(B, self.heads, S, S).reshape(B * self.heads, S, S)
        if _FASTPATH_OFF or torch.is_grad_enabled(): u = self.fuse(t, mask=am)
        else:                                                                  # older PyTorch: the fast path is skipped when gradients are enabled
            with torch.enable_grad(): u = self.fuse(t, mask=am).detach()
        w = (self.pool(u).squeeze(-1) + key).softmax(1)
        att = (w[..., None] * u).sum(1)
        mean = (u * used[..., None]).sum(1) / used.sum(1, keepdim=True)
        z_ = self.classifer(self.drop(torch.cat([att, mean], 1)))
        if aux: return z_, dict(e=e, p=p, c=c, alive=alive, w=w, bias=bias)
        return z_


class DeepSetSincNet(SensorSetSincNet):
    """Deep Sets baseline: the same per-sensor encoder, but the tokens are fused by a per-token MLP followed by masked mean and
    max pooling (no attention between sensors)."""
    def __init__(self, n_desc=47, d=128, **kw):
        super().__init__(n_desc=n_desc, d=d, **kw)
        self.fuse = None; self.pool = None
        self.phi = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU())

    def forward(self, x, f=None, alive=None):
        B, S, _ = x.shape
        if alive is None: alive = x.abs().amax(-1) > 0
        alive = alive.clone(); alive[alive.sum(1) == 0, 0] = True
        flat = alive.reshape(-1)
        emb = x.new_zeros(B * S, 256, dtype=torch.float32); emb[flat] = self.encode(x.reshape(B * S, -1)[flat])
        parts = [emb.reshape(B, S, 256)]
        if self.n_desc: parts.append(self.desc(f.float()))
        u = self.phi(self.tok(torch.cat(parts, -1)) + self.rank.weight[:S][None])
        mean = (u * alive[..., None]).sum(1) / alive.sum(1, keepdim=True)
        mx = u.masked_fill(~alive[..., None], float("-inf")).amax(1)
        return self.classifer(self.drop(torch.cat([mx, mean], 1)))
