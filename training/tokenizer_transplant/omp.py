"""Core OMP routines for tokenizer-embedding transplantation.

Model- and path-agnostic. The numerics here are the audited, verified version
from the original ``omp_transplant.py`` (centered-OMP / cosine selection /
normal-equation least squares with adaptive ridge) — kept byte-for-byte so the
held-out reconstruction fidelity is unchanged. See ``../docs/tokenizer/transplant.md``.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> str:
    """Pick a device: explicit override, else cuda if available, else cpu.

    The original script hardcoded ``cuda`` and silently failed on CPU-only
    login nodes; auto-detection avoids that.
    """
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def batch_omp(
    targets: torch.Tensor,
    atoms: torch.Tensor,
    k: int,
    device: str = "cuda",
    chunk: int = 2048,
    ridge: float = 1e-3,
    raw_select: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Orthogonal Matching Pursuit in the donor embedding space.

    targets [N,d] (donor space)  atoms [A,d] (donor space, shared anchors)
    returns idx [N,k] (into atoms), coef [N,k]

    Selection uses unit-normalized atoms (cosine) by default; ``raw_select=True``
    switches to canonical raw inner-product selection. Coefficients are solved
    by normal equations with an adaptive ridge for collinear atoms.
    """
    N = targets.shape[0]
    atoms = atoms.to(device, torch.float32)
    # canonical OMP (mergekit/paper) selects on RAW inner product; my variant used unit-norm atoms
    sel_atoms = atoms if raw_select else atoms / atoms.norm(dim=1, keepdim=True).clamp_min(1e-12)
    anorm = sel_atoms
    out_idx = torch.zeros(N, k, dtype=torch.long)
    out_coef = torch.zeros(N, k, dtype=torch.float32)
    for s in range(0, N, chunk):
        t = targets[s:s + min(chunk, N - s)].to(device, torch.float32)  # [B,d]
        B = t.shape[0]
        resid = t.clone()
        sel = torch.zeros(B, k, dtype=torch.long, device=device)
        coef = torch.zeros(B, k, device=device)
        for i in range(k):
            corr = resid @ anorm.T                       # [B,A]
            score = corr.abs()                           # OMP selects max |correlation|
            if i > 0:
                score.scatter_(1, sel[:, :i], -1.0)      # mask already-picked (scores are >=0)
            best = score.argmax(dim=1)                   # [B]
            sel[:, i] = best
            G = atoms[sel[:, :i + 1]]                     # [B,m,d]
            # solve min_x || x @ G - t ||  -> A=G^T [B,d,m], b=t [B,d,1]
            At = G.transpose(1, 2)                        # [B,d,m]
            # normal equations with ridge for stability
            GtG = torch.bmm(G, At)                        # [B,m,m]
            # adaptive ridge scaled to per-item diagonal magnitude (handles collinear atoms)
            diag = GtG.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-6)  # [B]
            GtG = GtG + (ridge * diag).view(-1, 1, 1) * torch.eye(i + 1, device=device)
            rhs = torch.bmm(G, t.unsqueeze(2))            # [B,m,1]
            x = torch.linalg.solve(GtG, rhs)              # [B,m,1]
            coef[:, :i + 1] = x.squeeze(2)
            resid = t - torch.bmm(coef[:, :i + 1].unsqueeze(1), G).squeeze(1)
        out_idx[s:s + B] = sel.cpu()
        out_coef[s:s + B] = coef.cpu()
        if (s // chunk) % 4 == 0:
            rms = resid.pow(2).mean().sqrt()
            print(f"    omp {s + B}/{N}  (last resid rms {rms:.4f})", flush=True)
    return out_idx, out_coef


@torch.no_grad()
def reconstruct(
    out_idx: torch.Tensor,
    out_coef: torch.Tensor,
    base_anchor_emb: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """out[t] = sum coef * base_anchor[idx].  base_anchor_emb [A,d]"""
    base_anchor_emb = base_anchor_emb.to(device, torch.float32)
    N, k = out_idx.shape
    res = torch.zeros(N, base_anchor_emb.shape[1], device=device)
    for s in range(0, N, 4096):
        e = min(s + 4096, N)
        idx = out_idx[s:e].to(device)
        cf = out_coef[s:e].to(device)
        picked = base_anchor_emb[idx]                    # [b,k,d]
        res[s:e] = torch.bmm(cf.unsqueeze(1), picked).squeeze(1)
    return res.cpu()
