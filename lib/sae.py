"""TopK sparse dictionaries (SAEs): the model, training, encoding in chunks, ablation, and mean activations per mask.

The model computes f = TopK(W_e (x - b_dec) + b_e) and x_hat = W_d f + b_dec. TopK keeps the k largest
pre-activations and applies ReLU, so activations are never negative and each row has at most k nonzero. Training
minimizes mean squared error with Adam and rescales decoder columns to unit norm after each step. Every 10 epochs
(not in the last 5), latents that did not fire in those epochs are reset to point at random training pixels.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from lib.config import SAE_DIM, SAE_LATENTS, SAE_K, DIXIE_CKPT

class TopKSAE(nn.Module):
    def __init__(self, d=SAE_DIM, m=SAE_LATENTS, k=SAE_K):
        super().__init__()
        self.k = k
        self.W_e = nn.Parameter(torch.randn(m, d) * 0.05)
        self.b_e = nn.Parameter(torch.zeros(m))
        self.W_d = nn.Parameter(self.W_e.data.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_d.data /= self.W_d.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_e.T + self.b_e
        topv, topi = torch.topk(pre, self.k, dim=-1)
        f = torch.zeros_like(pre)
        f.scatter_(-1, topi, torch.relu(topv))
        return f

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_d.T + self.b_dec, f

def load_sae(path=None, **dims):
    """Load a trained dictionary in eval mode (by default the Dixie dictionary). Pass d, m, k for other sizes, such as
    AlphaEarth's d=64, m=512, k=8."""
    sae = TopKSAE(**dims)
    sae.load_state_dict(torch.load(path or DIXIE_CKPT))
    sae.eval()
    return sae

def train(X, epochs=40, bs=4096, lr=1e-3, seed=0, n_val=50_000, log=True,
          d=None, m=SAE_LATENTS, k=SAE_K):
    """Train on (N, D) float32. The last n_val rows are held out to compute the logged R2."""
    torch.manual_seed(seed)
    Xt, Xv = torch.from_numpy(X[:-n_val]), torch.from_numpy(X[-n_val:])
    sae = TopKSAE(d=d or X.shape[1], m=m, k=k)
    sae.b_dec.data = Xt.mean(0)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    act_count = torch.zeros(m)
    var = (Xv - Xv.mean(0)).pow(2).sum()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0] - bs + 1, bs):
            xb = Xt[perm[i:i + bs]]
            xh, f = sae(xb)
            loss = (xh - xb).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.normalize_decoder()
            act_count += (f > 0).float().sum(0)
        if (ep + 1) % 10 == 0 and ep < epochs - 5:      # reinitialize dead latents
            dead = act_count == 0
            if int(dead.sum()):
                with torch.no_grad():
                    xb = Xt[torch.randperm(Xt.shape[0])[:int(dead.sum())]]
                    dirs = xb - sae.b_dec
                    dirs /= dirs.norm(dim=1, keepdim=True)
                    sae.W_d.data[:, dead] = dirs.T
                    sae.W_e.data[dead] = dirs * 0.5
                    sae.b_e.data[dead] = 0.0
            act_count.zero_()
        if log and (ep + 1) % 5 == 0:
            with torch.no_grad():
                xh, f = sae(Xv)
                fvu = (xh - Xv).pow(2).sum() / var
                print(f"ep{ep+1:3d}  R2={1-fvu:.4f}  L0={(f>0).float().sum(1).mean():.1f}"
                      f"  dead(val)={((f>0).sum(0)==0).float().mean():.2f}"
                      f"  {time.time()-t0:.0f}s", flush=True)
    return sae

def train_and_save(X, path, threads=32, **kw):
    """Train with the paper's defaults (40 epochs, seed 0) and save the state dict. threads=None keeps torch's own."""
    if threads is not None:
        torch.set_num_threads(threads)
    sae = train(X, **{"epochs": 40, "seed": 0, "log": True, **kw})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), path)
    return sae

@torch.no_grad()
def encode_array(sae, x, chunk=65536):
    """Encode a (..., D) array into (..., M) activations, in chunks on the CPU."""
    flat = torch.from_numpy(np.ascontiguousarray(x).reshape(-1, x.shape[-1]))
    out = [sae.encode(flat[i:i + chunk]).numpy() for i in range(0, flat.shape[0], chunk)]
    return np.concatenate(out).reshape(x.shape[:-1] + (-1,))

@torch.no_grad()
def val_r2(sae, x):
    xt = torch.from_numpy(x.reshape(-1, x.shape[-1]))
    xh, _ = sae(xt)
    return float(1 - (xh - xt).pow(2).sum() / (xt - xt.mean(0)).pow(2).sum())

@torch.no_grad()
def ablate(sae, x, feats):
    """Remove the chosen features from x: x' = x - sum_i f_i(x) d_i. The shape stays the same."""
    f = encode_array(sae, x)
    d = sae.W_d.data.numpy()[:, feats]
    return x - f[..., feats] @ d.T

def group_sums(sae, x, masks, chunk=192):
    """Sum the activations under each mask over an (H, W, D) array, encoding `chunk` rows at a time.

    Returns ({mask: (M,) activation sums}, {mask: pixel count}). Masks are cropped to the array.
    """
    H, W = x.shape[:2]
    ms = {k: m[:H, :W] for k, m in masks.items()}
    sums = {k: np.zeros(sae.W_e.shape[0]) for k in ms}
    counts = {k: 0 for k in ms}
    for r0 in range(0, H, chunk):
        f = encode_array(sae, x[r0:r0 + chunk])
        for k, m in ms.items():
            mm = m[r0:r0 + chunk]
            sums[k] += f[mm].sum(0)
            counts[k] += int(mm.sum())
    return sums, counts

def group_means(sae, x, masks, chunk=192):
    """{mask: mean activation of every feature over the mask's pixels}. An empty mask gives zeros."""
    sums, counts = group_sums(sae, x, masks, chunk)
    return {k: sums[k] / max(counts[k], 1) for k in sums}
