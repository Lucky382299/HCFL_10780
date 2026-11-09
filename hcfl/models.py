"""Local classifier F_theta: MLP 64-32-16, ReLU, dropout 0.25, softmax output.

Two interchangeable backends expose the same flat-parameter interface used by
the federated code:
  * TorchMLP  - PyTorch implementation (as in the manuscript; used when torch is installed)
  * NumpyMLP  - dependency-free NumPy implementation (same architecture, Adam/SGD,
                cross-entropy); used automatically when torch is unavailable,
                e.g. on a minimal Raspberry Pi image.

grad_hook(flat_params) returns an extra flat gradient term. FedProx uses
mu*(w - w_global); SCAFFOLD uses (c - c_i).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpyMLP:
    def __init__(self, d_in, n_classes=2, hidden=(64, 32, 16), dropout=0.25, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [d_in, *hidden, n_classes]
        self.shapes = []
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))
            params += [W, np.zeros(b)]
            self.shapes += [(a, b), (b,)]
        self.params = params
        self.dropout = dropout
        self.rng = np.random.default_rng(seed + 1)
        self.num_params = int(sum(p.size for p in params))

    # ---- flat interface ----
    def get_flat(self):
        return np.concatenate([p.ravel() for p in self.params]).astype(np.float64)

    def set_flat(self, v):
        i = 0
        for k, s in enumerate(self.shapes):
            n = int(np.prod(s))
            self.params[k] = np.asarray(v[i:i + n], dtype=np.float64).reshape(s).copy()
            i += n

    def _flat_grads(self, grads):
        return np.concatenate([g.ravel() for g in grads])

    def _split(self, v):
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
        return out

    # ---- forward / backward ----
    def _forward(self, X, train):
        acts, masks = [X], []
        h = X
        L = len(self.params) // 2
        for l in range(L):
            W, b = self.params[2 * l], self.params[2 * l + 1]
            z = h @ W + b
            if l < L - 1:
                h = np.maximum(z, 0.0)
                if train and self.dropout > 0:
                    m = (self.rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
                    h = h * m
                else:
                    m = None
                masks.append(m)
                acts.append(h)
            else:
                h = z
        return h, acts, masks

    def _backward(self, logits, y, acts, masks):
        n = len(y)
        P = _softmax(logits)
        P[np.arange(n), y] -= 1.0
        g = P / n
        grads = [None] * len(self.params)
        L = len(self.params) // 2
        for l in reversed(range(L)):
            W = self.params[2 * l]
            grads[2 * l] = acts[l].T @ g
            grads[2 * l + 1] = g.sum(axis=0)
            if l > 0:
                g = g @ W.T
                if masks[l - 1] is not None:
                    g = g * masks[l - 1]
                g = g * (acts[l] > 0)
        return grads

    def _per_sample_flat_grads(self, X, y):
        """(b, P) matrix of per-example gradients of the cross-entropy loss."""
        logits, acts, masks = self._forward(X, train=True)
        b = len(y)
        g = _softmax(logits)
        g[np.arange(b), y] -= 1.0
        L = len(self.params) // 2
        per = [None] * len(self.params)
        for l in reversed(range(L)):
            W = self.params[2 * l]
            per[2 * l] = np.einsum("bi,bj->bij", acts[l], g).reshape(b, -1)
            per[2 * l + 1] = g
            if l > 0:
                g = g @ W.T
                if masks[l - 1] is not None:
                    g = g * masks[l - 1]
                g = g * (acts[l] > 0)
        return np.concatenate(per, axis=1)

    def _apply(self, grads, opt_state, lr, optimizer):
        m, v, t = opt_state
        b1, b2, eps = 0.9, 0.999, 1e-8
        if optimizer == "adam":
            t += 1
            for k, g in enumerate(grads):
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
                self.params[k] -= lr * mh / (np.sqrt(vh) + eps)
        else:
            for k, g in enumerate(grads):
                self.params[k] -= lr * g
        return (m, v, t)

    def train_dpsgd(self, X, y, epochs, batch_size, lr, clip, sigma, seed=0, noise_seed=0,
                    optimizer="adam", grad_hook=None):
        """DP-SGD: Poisson sampling (q = B/n), per-example L2 clipping to `clip`,
        Gaussian noise N(0, sigma^2 clip^2) on the summed gradient, divided by B.
        sigma = 0 gives the clipped non-private twin used for the PDI."""
        samp = np.random.default_rng(seed)
        noise = np.random.default_rng(noise_seed)
        self.rng = np.random.default_rng(seed + 7)      # same dropout masks for the twin run
        n = len(X); q = min(1.0, batch_size / n)
        steps_per_epoch = int(np.ceil(n / batch_size))
        state = ([np.zeros_like(p) for p in self.params], [np.zeros_like(p) for p in self.params], 0)
        steps = 0
        for _ in range(epochs * steps_per_epoch):
            idx = np.where(samp.random(n) < q)[0]
            if len(idx):
                G = self._per_sample_flat_grads(X[idx], y[idx])
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                gsum = (G / np.maximum(1.0, norms / clip)).sum(0)
            else:
                gsum = np.zeros(self.num_params)
            z = noise.normal(0.0, sigma * clip, self.num_params) if sigma > 0 else 0.0
            g = (gsum + z) / batch_size
            if grad_hook is not None:
                g = g + grad_hook(self.get_flat())
            state = self._apply(self._split(g), state, lr, optimizer)
            steps += 1
        return steps

    def train_local(self, X, y, epochs, batch_size, lr, optimizer="adam", grad_hook=None, seed=0):
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        steps = 0
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch_size):
                idx = order[s:s + batch_size]
                logits, acts, masks = self._forward(X[idx], train=True)
                grads = self._backward(logits, y[idx], acts, masks)
                if grad_hook is not None:
                    extra = self._split(grad_hook(self.get_flat()))
                    grads = [g + e for g, e in zip(grads, extra)]
                if optimizer == "adam":
                    t += 1
                    for k, g in enumerate(grads):
                        m[k] = b1 * m[k] + (1 - b1) * g
                        v[k] = b2 * v[k] + (1 - b2) * g * g
                        mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
                        self.params[k] -= lr * mh / (np.sqrt(vh) + eps)
                else:
                    for k, g in enumerate(grads):
                        self.params[k] -= lr * g
                steps += 1
        return steps

    def predict_proba(self, X, batch=8192):
        out = []
        for s in range(0, len(X), batch):
            logits, _, _ = self._forward(X[s:s + batch], train=False)
            out.append(_softmax(logits))
        return np.vstack(out) if out else np.zeros((0, 2))

    def loss(self, X, y):
        P = self.predict_proba(X)
        return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


if _HAS_TORCH:
    class _Net(nn.Module):
        def __init__(self, d_in, n_classes, hidden, dropout):
            super().__init__()
            layers, a = [], d_in
            for h in hidden:
                layers += [nn.Linear(a, h), nn.ReLU(), nn.Dropout(dropout)]
                a = h
            layers.append(nn.Linear(a, n_classes))  # softmax is inside the CE loss
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class TorchMLP:
        def __init__(self, d_in, n_classes=2, hidden=(64, 32, 16), dropout=0.25, seed=0, device="cpu"):
            torch.manual_seed(seed)
            self.device = torch.device(device)
            self.net = _Net(d_in, n_classes, list(hidden), dropout).to(self.device).double()
            self.num_params = int(sum(p.numel() for p in self.net.parameters()))

        def get_flat(self):
            return torch.nn.utils.parameters_to_vector(self.net.parameters()).detach().cpu().numpy().astype(np.float64)

        def set_flat(self, v):
            t = torch.as_tensor(np.asarray(v), dtype=torch.float64, device=self.device)
            torch.nn.utils.vector_to_parameters(t, self.net.parameters())

        def train_local(self, X, y, epochs, batch_size, lr, optimizer="adam", grad_hook=None, seed=0):
            g = torch.Generator().manual_seed(int(seed))
            Xt = torch.as_tensor(X, dtype=torch.float64, device=self.device)
            yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
            opt = (torch.optim.Adam(self.net.parameters(), lr=lr) if optimizer == "adam"
                   else torch.optim.SGD(self.net.parameters(), lr=lr))
            lossf = nn.CrossEntropyLoss()
            self.net.train()
            steps = 0
            for _ in range(epochs):
                order = torch.randperm(len(Xt), generator=g)
                for s in range(0, len(Xt), batch_size):
                    idx = order[s:s + batch_size].to(self.device)
                    opt.zero_grad()
                    loss = lossf(self.net(Xt[idx]), yt[idx])
                    loss.backward()
                    if grad_hook is not None:
                        extra = torch.as_tensor(grad_hook(self.get_flat()), dtype=torch.float64, device=self.device)
                        i = 0
                        for p in self.net.parameters():
                            n = p.numel()
                            p.grad.add_(extra[i:i + n].view_as(p)); i += n
                    opt.step()
                    steps += 1
            return steps

        def train_dpsgd(self, X, y, epochs, batch_size, lr, clip, sigma, seed=0, noise_seed=0,
                        optimizer="adam", grad_hook=None):
            """DP-SGD (same algorithm as NumpyMLP.train_dpsgd)."""
            samp = np.random.default_rng(seed)
            gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
            torch.manual_seed(int(seed) + 7)               # same dropout masks for the twin run
            Xt = torch.as_tensor(X, dtype=torch.float64, device=self.device)
            yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
            params = [p for p in self.net.parameters()]
            opt = (torch.optim.Adam(params, lr=lr) if optimizer == "adam" else torch.optim.SGD(params, lr=lr))
            lossf = nn.CrossEntropyLoss(reduction="sum")
            self.net.train()
            n = len(X); q = min(1.0, batch_size / n)
            steps_per_epoch = int(np.ceil(n / batch_size))
            steps = 0
            for _ in range(epochs * steps_per_epoch):
                idx = np.where(samp.random(n) < q)[0]
                gsum = [torch.zeros_like(p) for p in params]
                for j in idx:                                  # per-example gradients
                    self.net.zero_grad()
                    lossf(self.net(Xt[j:j + 1]), yt[j:j + 1]).backward()
                    gs = [p.grad.detach().clone() for p in params]
                    norm = torch.sqrt(sum((g ** 2).sum() for g in gs))
                    scale = 1.0 / max(1.0, float(norm) / clip)
                    for a, g in zip(gsum, gs):
                        a.add_(g * scale)
                opt.zero_grad()
                for p, a in zip(params, gsum):
                    z = (torch.normal(0.0, sigma * clip, size=p.shape, generator=gen, dtype=torch.float64)
                         .to(self.device) if sigma > 0 else 0.0)
                    p.grad = (a + z) / batch_size
                if grad_hook is not None:
                    extra = torch.as_tensor(grad_hook(self.get_flat()), dtype=torch.float64, device=self.device)
                    i = 0
                    for p in params:
                        k = p.numel(); p.grad.add_(extra[i:i + k].view_as(p)); i += k
                opt.step()
                steps += 1
            return steps

        @torch.no_grad()
        def predict_proba(self, X, batch=8192):
            self.net.eval()
            out = []
            for s in range(0, len(X), batch):
                xb = torch.as_tensor(X[s:s + batch], dtype=torch.float64, device=self.device)
                out.append(torch.softmax(self.net(xb), dim=1).cpu().numpy())
            return np.vstack(out) if out else np.zeros((0, 2))

        def loss(self, X, y):
            P = self.predict_proba(X)
            return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def make_model(d_in, n_classes, tc, seed=0):
    backend = tc.backend
    if backend == "auto":
        backend = "torch" if _HAS_TORCH else "numpy"
    if backend == "torch":
        if not _HAS_TORCH:
            raise ImportError("PyTorch is not installed; set train.backend: numpy")
        return TorchMLP(d_in, n_classes, tc.hidden, tc.dropout, seed, tc.device)
    return NumpyMLP(d_in, n_classes, tc.hidden, tc.dropout, seed)
