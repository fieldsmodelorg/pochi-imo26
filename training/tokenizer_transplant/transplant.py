"""Tokenizer transplant pipeline: rebuild embed_tokens / lm_head for a new vocab.

Keeps the base model's transformer body untouched; only the two vocab-sized
matrices are rebuilt. Shared tokens (string match) are copied; new tokens are
filled by (centered) OMP over the shared anchors. Configuration is external
(YAML) so the same code works for any base/donor pair.
"""

from __future__ import annotations

import json
import os
import time

import torch
import yaml
from pydantic import BaseModel
from safetensors import safe_open

from .omp import batch_omp, reconstruct, resolve_device


class TensorNames(BaseModel):
    base_embed: str = "model.embed_tokens.weight"
    base_head: str = "lm_head.weight"
    donor_embed: str = "embed.weight"
    donor_head: str = "head.weight"


class TransplantConfig(BaseModel):
    """Everything that was previously hardcoded in omp_transplant.py."""

    base: str                 # local path to base model (weights kept)
    donor_weights: str        # local path to donor weights (only embed/head shards needed)
    donor_tokenizer: str      # HF id or local path for the donor tokenizer
    out: str                  # output dir for the transplanted model
    k: int = 64               # OMP sparsity
    centered: bool = True     # mean-centered OMP (essential for lm_head logit calibration)
    cosine_select: bool = True  # cosine atom selection (False -> raw canonical OMP)
    ridge: float = 1e-3
    tensors: TensorNames = TensorNames()
    # Chat / special-token handling. ``special_map`` maps a donor special-token
    # *string* to a base-tokenizer *text*; the donor row is overwritten with the
    # (norm-rescaled) mean of the base rows that text tokenizes to. This replaces
    # the meaningless OMP fill for control tokens with the base model's trained
    # vectors, mapped by role (e.g. donor "<｜User｜>" <- base "<|im_start|>user\n").
    # ``chat_template`` (jinja) is written onto the output tokenizer.
    special_map: dict[str, str] = {}
    chat_template: str | None = None
    pad_token: str | None = None  # donor string to use as pad (else donor default / eos)

    @classmethod
    def from_yaml(cls, path: str) -> TransplantConfig:
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def load_tensor(model_dir: str, name: str) -> torch.Tensor:
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model_dir, idx[name]), framework="pt") as f:
        return f.get_tensor(name)


def build_anchor_map(base: str, donor_tokenizer: str):
    """Return (anchor_donor_ids, anchor_base_ids, new_ids, donor_vocab, base_vocab).

    Anchors are tokens whose *string* surface form exists in both vocabularies.
    """
    from transformers import AutoTokenizer

    tb = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    td = AutoTokenizer.from_pretrained(donor_tokenizer, trust_remote_code=True)
    vb, vd = tb.get_vocab(), td.get_vocab()          # str -> id
    shared = set(vb) & set(vd)
    a_d, a_b = [], []
    for s in shared:
        a_d.append(vd[s])
        a_b.append(vb[s])
    new_ids = sorted(set(vd.values()) - {vd[s] for s in shared})
    return (torch.tensor(a_d), torch.tensor(a_b), torch.tensor(new_ids), len(vd), len(vb))


def apply_special_map(out: torch.Tensor, b_mat: torch.Tensor, cfg: TransplantConfig,
                      base_tok, donor_tok, tag: str) -> None:
    """Overwrite donor special-token rows with role-mapped base rows (in place).

    For each ``donor_str -> base_text`` rule: tokenize ``base_text`` with the base
    tokenizer, take the mean of those base rows, and rescale to their mean norm
    (a no-op for a single token, i.e. an exact copy). Done in fp32 before cast.
    """
    if not cfg.special_map or base_tok is None or donor_tok is None:
        return
    dv = donor_tok.get_vocab()
    n = 0
    for donor_str, base_text in cfg.special_map.items():
        if donor_str not in dv:
            print(f"  [special] WARN donor token {donor_str!r} not in vocab; skipped")
            continue
        bids = base_tok(base_text, add_special_tokens=False)["input_ids"]
        if not bids:
            print(f"  [special] WARN {base_text!r} tokenized to nothing; skipped")
            continue
        rows = b_mat[bids]                                       # [m,d] fp32
        v = rows.mean(0)
        v = v / v.norm().clamp_min(1e-8) * rows.norm(dim=1).mean()
        out[dv[donor_str]] = v
        n += 1
    print(f"  {tag}: applied {n} special-token overrides")


def build_matrix(cfg: TransplantConfig, donor_name: str, base_name: str,
                 a_d, a_b, new_ids, nd, device: str,
                 base_tok=None, donor_tok=None) -> torch.Tensor:
    """Rebuild one vocab matrix (embed_tokens or lm_head) for the new vocab."""
    print(f"\n## transplant {base_name}  (donor {donor_name})")
    d_mat = load_tensor(cfg.donor_weights, donor_name).float()   # [nd_donor,d]
    b_mat = load_tensor(cfg.base, base_name).float()             # [nb,d]
    d = b_mat.shape[1]
    out = torch.zeros(nd, d, dtype=torch.float32)
    out[a_d] = b_mat[a_b]                                        # shared: copy base row exactly
    raw_select = not cfg.cosine_select
    if cfg.centered:
        # mean-center both spaces: pins the implicit mean-row coefficient to 1, so OMP's
        # unconstrained coeff-sum can't scale the dominant (rogue-dim) direction and blow up
        # output logits for new tokens. Fidelity-neutral; critical for the lm_head. See docs.
        md = d_mat[a_d].mean(0, keepdim=True)
        mb = b_mat[a_b].mean(0, keepdim=True)
        idx, coef = batch_omp(d_mat[new_ids] - md, d_mat[a_d] - md, cfg.k,
                              device=device, ridge=cfg.ridge, raw_select=raw_select)
        out[new_ids] = reconstruct(idx, coef, b_mat[a_b] - mb, device=device) + mb
        mode = "mean-centered"
    else:
        idx, coef = batch_omp(d_mat[new_ids], d_mat[a_d], cfg.k,
                              device=device, ridge=cfg.ridge, raw_select=raw_select)
        out[new_ids] = reconstruct(idx, coef, b_mat[a_b], device=device)
        mode = "uncentered"
    print(f"  {base_name}: copied {len(a_d)} shared rows, OMP {len(new_ids)} new rows ({mode})")
    apply_special_map(out, b_mat, cfg, base_tok, donor_tok, base_name)
    return out.to(torch.bfloat16)


def run(cfg: TransplantConfig, device: str | None = None) -> str:
    """Build the transplanted model and save it to ``cfg.out``. Returns the path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    t0 = time.time()
    a_d, a_b, new_ids, nd, nb = build_anchor_map(cfg.base, cfg.donor_tokenizer)
    base_tok = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
    donor_tok = AutoTokenizer.from_pretrained(cfg.donor_tokenizer, trust_remote_code=True)
    print(f"donor vocab {nd}, base vocab {nb}, anchors {len(a_d)}, new tokens {len(new_ids)} "
          f"| device={device} k={cfg.k} centered={cfg.centered} cosine={cfg.cosine_select} "
          f"special_map={len(cfg.special_map)}")
    emb_new = build_matrix(cfg, cfg.tensors.donor_embed, cfg.tensors.base_embed,
                           a_d, a_b, new_ids, nd, device, base_tok, donor_tok)
    head_new = build_matrix(cfg, cfg.tensors.donor_head, cfg.tensors.base_head,
                            a_d, a_b, new_ids, nd, device, base_tok, donor_tok)

    print("\n## loading base model & assembling")
    model = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, trust_remote_code=True)
    model.resize_token_embeddings(nd)
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(emb_new)
        model.get_output_embeddings().weight.copy_(head_new)

    # update token ids to match the donor tokenizer
    td = donor_tok
    if cfg.chat_template:
        td.chat_template = cfg.chat_template
    if cfg.pad_token is not None:
        td.pad_token = cfg.pad_token
    model.config.vocab_size = nd
    model.config.bos_token_id = td.bos_token_id
    model.config.eos_token_id = td.eos_token_id
    model.config.pad_token_id = td.pad_token_id if td.pad_token_id is not None else td.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = td.bos_token_id
        model.generation_config.eos_token_id = td.eos_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id

    os.makedirs(cfg.out, exist_ok=True)
    model.save_pretrained(cfg.out, safe_serialization=True)
    td.save_pretrained(cfg.out)
    print(f"\nDONE in {time.time() - t0:.0f}s -> {cfg.out}")
    return cfg.out
