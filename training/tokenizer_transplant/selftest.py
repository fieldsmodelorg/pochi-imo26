"""Held-out reconstruction sanity check for a transplant config.

Holds out a sample of shared anchors, reconstructs them from the rest via the
same OMP path used in production, and reports cosine to the true base rows.
Unlike the original (which only tested the uncentered path), this honours
``cfg.centered`` so it validates exactly what ``run`` ships. Needs the real
base + donor weights, so it is an opt-in integration check, not a unit test.
"""

from __future__ import annotations

import torch

from .omp import batch_omp, reconstruct, resolve_device
from .transplant import TransplantConfig, build_anchor_map, load_tensor


def selftest(cfg: TransplantConfig, device: str | None = None, hold: int = 500,
             which: str = "embed") -> dict:
    device = resolve_device(device)
    if which == "embed":
        donor_name, base_name = cfg.tensors.donor_embed, cfg.tensors.base_embed
    elif which == "lm_head":
        donor_name, base_name = cfg.tensors.donor_head, cfg.tensors.base_head
    else:
        raise ValueError(f"which must be 'embed' or 'lm_head', got {which!r}")

    print(f"=== SELF TEST: hold out {hold} shared {which} tokens, reconstruct, measure cosine "
          f"(centered={cfg.centered}, device={device}) ===")
    a_d, a_b, _new, _nd, _nb = build_anchor_map(cfg.base, cfg.donor_tokenizer)
    d_emb = load_tensor(cfg.donor_weights, donor_name).float()
    b_emb = load_tensor(cfg.base, base_name).float()

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(a_d), generator=g)
    held, keep = perm[:hold], perm[hold:]
    held_d, held_b = a_d[held], a_b[held]
    dict_d, dict_b = a_d[keep], a_b[keep]

    targets = d_emb[held_d]
    atoms = d_emb[dict_d]
    raw_select = not cfg.cosine_select
    if cfg.centered:
        md, mb = atoms.mean(0, keepdim=True), b_emb[dict_b].mean(0, keepdim=True)
        idx, coef = batch_omp(targets - md, atoms - md, cfg.k, device=device,
                              ridge=cfg.ridge, raw_select=raw_select)
        recon = reconstruct(idx, coef, b_emb[dict_b] - mb, device=device) + mb
    else:
        idx, coef = batch_omp(targets, atoms, cfg.k, device=device,
                              ridge=cfg.ridge, raw_select=raw_select)
        recon = reconstruct(idx, coef, b_emb[dict_b], device=device)

    truth = b_emb[held_b]
    cos = torch.nn.functional.cosine_similarity(recon, truth, dim=1)
    rel = (recon - truth).norm(dim=1) / truth.norm(dim=1).clamp_min(1e-9)
    out = {
        "cos_mean": cos.mean().item(), "cos_median": cos.median().item(),
        "cos_p10": cos.kthvalue(max(1, hold // 10)).values.item(),
        "rel_l2_mean": rel.mean().item(), "rel_l2_median": rel.median().item(),
    }
    print(f"  cosine to true base {which}: mean {out['cos_mean']:.4f} "
          f"median {out['cos_median']:.4f} p10 {out['cos_p10']:.4f}")
    print(f"  relative L2 error       : mean {out['rel_l2_mean']:.4f} "
          f"median {out['rel_l2_median']:.4f}")
    return out
