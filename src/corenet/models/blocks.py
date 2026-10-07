"""Neural-network primitives of CoReNet.

    GhostBatchNorm1d   ghost-batch normalisation (https://arxiv.org/abs/1705.08741), optionally
                       followed by a convolution with kernel = stride
    conv1d             1d convolution with optional phase symmetry (https://arxiv.org/abs/1603.05201)
    linear             linear layer
    AttentionGate      small MLP producing per-pair offsets of attention scores / values
    MinimalAttention   masked attention with GBN-normalised scores and per-pair features

Tensors are laid out ``(batch, features, pixels)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourvectors import NonLU
from .variance import TensorVariance


class GhostBatchNorm1d(nn.Module):
    """Ghost batch normalisation.

    In training mode with ``nGhostBatches > 0`` the batch is split into ``nGhostBatches`` ghost
    batches that are normalised with their own (optionally masked) statistics while running
    statistics are accumulated; with ``nGhostBatches == 0`` and in eval mode the running (or
    frozen, see ``initMeanStd``) statistics are used. With ``conv=True`` the normalised tensor
    goes through a ``conv1d`` (kernel = stride = ``stride``), otherwise through a per-feature
    affine transform.
    """

    def __init__(self, features, ghost_batch_size=32, number_of_ghost_batches=64, nAveraging=1, stride=1, eta=0.9,
                 bias=True, device="cuda", name="", conv=False, features_out=None, phase_symmetric=False):
        super().__init__()
        self.name = name
        self.stride = stride
        self.device = device
        self.features = features
        self.features_out = features_out if features_out is not None else self.features
        self.register_buffer("ghost_batch_size", torch.tensor(ghost_batch_size, dtype=torch.long))
        self.register_buffer("nGhostBatches", torch.tensor(number_of_ghost_batches * nAveraging, dtype=torch.long))
        self.conv = False
        self.gamma = None
        self.bias = None
        if conv:
            self.conv = conv1d(self.features, self.features_out, self.stride, self.stride, name="%s conv" % name,
                               bias=bias, phase_symmetric=phase_symmetric)
        else:
            self.gamma = nn.Parameter(torch.ones(self.features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.features))
        self._mean_var = TensorVariance()
        self.runningStats = True
        self.initialized = False

        self.register_buffer("eps", torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float))
        self.register_buffer("m", torch.zeros((1, 1, self.stride, self.features), dtype=torch.float))
        self.register_buffer("s", torch.ones((1, 1, self.stride, self.features), dtype=torch.float))
        self.register_buffer("zero", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float))
        self.register_buffer("two", torch.tensor(2.0, dtype=torch.float))

    @torch.no_grad()
    def updateMeanStd(self, x, mask=None):
        """Accumulate input statistics (``mask`` True = ignore that pixel). Only for stride 1."""
        batch_size = x.shape[0]
        pixels = x.shape[2]
        x = x.detach().transpose(1, 2).contiguous().view(batch_size * pixels, 1, self.features)
        if mask is not None:
            mask = mask.detach().view(batch_size * pixels)
            x = x[mask == 0, :, :]
        x = x.view(-1, 1, self.stride, self.features)
        self._mean_var += TensorVariance(x)

    @torch.no_grad()
    def initMeanStd(self):
        """Freeze the accumulated statistics (non-finite means -> 0, std floored at 1e-4)."""
        m = self._mean_var.mean.to(self.m.device)
        s = self._mean_var.variance_unbiased.sqrt().to(self.s.device)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s))
        s = s.clamp(min=1e-4)
        self.m.copy_(m)
        self.s.copy_(s)
        self.initialized = True
        self.runningStats = False

    def setGhostBatches(self, nGhostBatches):
        self.nGhostBatches = torch.tensor(nGhostBatches, dtype=torch.long).to(self.nGhostBatches.device)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        pixel_groups = torch.div(pixels, self.stride, rounding_mode="trunc")

        if self.training and self.nGhostBatches != 0:
            self.ghost_batch_size = batch_size // self.nGhostBatches.abs()
            x = (x.transpose(1, 2).contiguous()
                 .view(self.nGhostBatches.abs(), self.ghost_batch_size * pixel_groups, self.stride, self.features))
            if mask is None:
                gbm = x.mean(dim=1, keepdim=True)
                gbs = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
            else:
                # masked mean and std for each ghost batch
                mask = mask.view(self.nGhostBatches.abs(), self.ghost_batch_size * pixel_groups, self.stride, 1)
                nUnmasked = (mask == 0).sum(dim=1, keepdim=True).float()
                unmasked0 = (nUnmasked == self.zero).float()
                unmasked1 = (nUnmasked == self.one).float()
                denomMean = nUnmasked + unmasked0                                   # prevent divide by zero
                denomVar = nUnmasked + unmasked0 * self.two + unmasked1 - self.one  # ... with Bessel correction
                gbm = x.masked_fill(mask, 0).sum(dim=1, keepdim=True) / denomMean
                gbs = (((x - gbm).masked_fill(mask, 0) ** 2).sum(dim=1, keepdim=True) / denomVar + self.eps).sqrt()

            if self.runningStats:  # running mean and standard deviation
                bm = gbm.detach().mean(dim=0, keepdim=True)
                bs = gbs.detach().mean(dim=0, keepdim=True)
                if self.initialized:
                    self.m.copy_(self.eta * self.m + (self.one - self.eta) * bm)
                    self.s.copy_(self.eta * self.s + (self.one - self.eta) * bs)
                else:
                    self.m.copy_(bm)
                    self.s.copy_(bs)
                    self.initialized = True

            if self.nGhostBatches > 0:
                x = x - gbm
                x = x / gbs
            else:
                x = x.view(batch_size, pixel_groups, self.stride, self.features)
                x = x - self.m
                x = x / self.s
        else:
            # running (or frozen) statistics
            x = x.transpose(1, 2).view(batch_size, pixel_groups, self.stride, self.features)
            x = x - self.m
            x = x / self.s

        if self.conv:
            x = x.view(batch_size, pixels, self.features).transpose(1, 2).contiguous()
            x = self.conv(x)
        else:
            x = x * self.gamma
            if self.bias is not None:
                x = x + self.bias
            x = x.view(batch_size, pixels, self.features).transpose(1, 2).contiguous()
        return x


class conv1d(nn.Module):
    """``nn.Conv1d`` wrapper; with ``phase_symmetric`` it computes half the output channels and
    returns ``cat(x, -x)`` plus a separate bias (https://arxiv.org/abs/1603.05201)."""

    def __init__(self, in_channels, out_channels=None, kernel_size=1, stride=1, bias=True, name="",
                 phase_symmetric=False, device=None):
        super().__init__()
        self.bias = bias and not phase_symmetric
        self.phase_symmetric = phase_symmetric
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        if phase_symmetric:
            self.out_channels = self.out_channels // 2
        # unused by the forward pass; kept so that checkpoints keep the same state-dict keys
        self.register_buffer("eps", torch.tensor(1e-5, dtype=torch.float))
        self.module = nn.Conv1d(self.in_channels, self.out_channels, kernel_size, stride=stride, bias=self.bias)
        if self.phase_symmetric and bias:
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels * 2, 1))
        self.name = name

    def forward(self, x):
        x = self.module(x)
        if self.phase_symmetric:
            x = torch.cat((x, -x), 1)
            if type(self.bias) is nn.Parameter:
                x = x + self.bias
        return x


class linear(nn.Module):
    def __init__(self, in_channels, out_channels, name=None, bias=True):
        super().__init__()
        self.module = nn.Linear(in_channels, out_channels, bias=bias)
        self.name = name

    def forward(self, x):
        return self.module(x)


class AttentionGate(nn.Module):
    """Two-layer MLP producing a per-(query, value) offset for attention scores or values."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class MinimalAttention(nn.Module):
    """Masked attention of query pixels over value pixels (https://arxiv.org/abs/1706.03762), reduced
    to what CoReNet uses.

    ``q`` (batch, d, qsl), ``v`` (batch, d, vsl), ``mask`` (batch, qsl, vsl), True where a (query,
    value) pair is ignored, ``q0`` the residual stream of the queries, ``qv`` optional per-pair
    features (batch, d, qsl, vsl) added to the scores and values. Scores are GBN-normalised,
    softmax-ed over the values and scaled by their sigmoid, so a query need not pick any value.

    ``use_attention_gate``: pair features enter scores and values through small MLP gates instead of
    a plain sum. ``use_kv_proj``: bottleneck query/key projections (d -> d/2) and a value projection.
    """

    def __init__(self, dim=8, heads=1, phase_symmetric=True, use_attention_gate=False, use_kv_proj=False, device="cuda"):
        super().__init__()
        self.device = device
        self.d = dim
        self.phase_symmetric = phase_symmetric
        self.h = heads
        self.dh = self.d // self.h
        self.use_attention_gate = use_attention_gate
        self.use_kv_proj = use_kv_proj

        if self.use_kv_proj:
            self.dh_k = self.d // (2 * self.h)
            self.dim_k = self.dh_k * self.h
            self.q_proj = conv1d(self.d, out_channels=self.dim_k, kernel_size=1, bias=True, device=device)
            self.k_proj = conv1d(self.d, out_channels=self.dim_k, kernel_size=1, bias=True, device=device)
            self.v_proj = conv1d(self.d, out_channels=self.d, kernel_size=1, bias=True, device=device)

        self.q_GBN = GhostBatchNorm1d(self.d, name="attention q GBN", device=device)
        self.v_GBN = GhostBatchNorm1d(self.d, name="attention v GBN", device=device)
        self.qv_GBN = GhostBatchNorm1d(self.d, name="attention qv GBN", device=device)
        if self.use_attention_gate:
            self.score_gate = AttentionGate(self.dh, 1)
            self.value_gate = AttentionGate(self.dh, self.dh)

        self.score_GBN = GhostBatchNorm1d(self.h, name="attention score GBN", device=device)
        self.conv = GhostBatchNorm1d(dim, phase_symmetric=phase_symmetric, conv=True, name="attention out convolution",
                                     device=device)
        self.negativeInfinity = torch.tensor(-1e9, dtype=torch.float).to(device)

    def attention(self, q, v, mask, qv=None, k=None):
        bs, qsl, vsl = q.shape[0], q.shape[3], v.shape[4]
        # with key projection: q (bs,h,dh_k,qsl,1), k (bs,h,dh_k,1,vsl), v (bs,h,dh,1,vsl)
        # otherwise:           q (bs,h,dh,qsl,1),                     v (bs,h,dh,1,vsl)
        if k is not None:
            score = (q * k).sum(dim=2)
        else:
            score = (q * v).sum(dim=2)

        if qv is not None:
            if self.use_attention_gate:
                qv_perm = qv.permute(0, 1, 3, 4, 2)
                score = score + self.score_gate(qv_perm).squeeze(-1)
            else:
                score = score + qv.sum(dim=2)

        # masked ghost batch normalisation of the (bs,h,qsl,vsl) scores
        score = score.view(bs, self.h, qsl * vsl)
        score = self.score_GBN(score, mask.view(bs, qsl * vsl) if mask is not None else None)
        score = score.view(bs, self.h, 1, qsl, vsl)  # extra dim for broadcasting over features
        if mask is not None:
            score = score.masked_fill(mask, self.negativeInfinity)

        v_weights = F.softmax(score, dim=4)          # which values best correspond to each query
        score = torch.sigmoid(score)                 # no value has to be picked if none match
        v_weights = v_weights * score

        if qv is not None:
            if self.use_attention_gate:
                qv_perm = qv.permute(0, 1, 3, 4, 2)
                v = v + self.value_gate(qv_perm).permute(0, 1, 4, 2, 3)
            else:
                v = v + qv
        if mask is not None:                         # mask is (bs, 1, 1, qsl, vsl)
            v_weights = v_weights.masked_fill(mask, 0)
            v = v.masked_fill(mask, 0)
        q_res = (v * v_weights).sum(dim=4)           # weighted sum of values, (bs, h, dh, qsl)
        return q_res, v_weights

    def setGhostBatches(self, nGhostBatches):
        self.score_GBN.setGhostBatches(nGhostBatches)
        self.q_GBN.setGhostBatches(nGhostBatches)
        self.v_GBN.setGhostBatches(nGhostBatches)
        self.qv_GBN.setGhostBatches(nGhostBatches)
        self.conv.setGhostBatches(nGhostBatches)

    def forward(self, q, v, mask=None, q0=None, qv=None):
        bs = q.shape[0]
        qsl = q.shape[2]
        vsl = v.shape[2]

        q_mask, v_mask = None, None
        if mask is not None:
            # flag queries / values whose pairs are all masked
            q_mask = mask.all(2).view(bs, 1, qsl)
            v_mask = mask.all(1).view(bs, 1, vsl)
            mask = mask.reshape(bs, qsl * vsl)

        if qv is not None:
            qv = qv.view(bs, self.d, qsl * vsl)
            qv = self.qv_GBN(qv, mask)
        q = self.q_GBN(q, q_mask)
        v = self.v_GBN(v, v_mask)

        if self.use_kv_proj:
            q_proj = self.q_proj(q)
            k_proj = self.k_proj(v)
            v_proj = self.v_proj(v)

        if mask is not None:  # broadcast over heads and features
            mask = mask.view(bs, 1, 1, qsl, vsl)

        if self.use_kv_proj:
            q = q_proj.view(bs, self.h, self.dh_k, qsl, 1)
            k = k_proj.view(bs, self.h, self.dh_k, 1, vsl)
            v = v_proj.view(bs, self.h, self.dh, 1, vsl)
        else:
            q = q.view(bs, self.h, self.dh, qsl, 1)
            k = None
            v = v.view(bs, self.h, self.dh, 1, vsl)

        if qv is not None:
            qv = qv.view(bs, self.h, self.dh, qsl, vsl)

        q, v_weights = self.attention(q, v, mask, qv, k=k)
        q, v_weights = q.view(bs, self.d, qsl), v_weights.view(bs, self.h, qsl, vsl)
        q = NonLU(q)
        q = self.conv(q, q_mask)
        if q_mask is not None:
            q = q.masked_fill(q_mask, 0)
        q = q0 + q  # residual
        q0 = q.clone()
        q = NonLU(q)
        return q, q0, v_weights


__all__ = ["GhostBatchNorm1d", "conv1d", "linear", "AttentionGate", "MinimalAttention"]
