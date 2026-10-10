"""Load only the embedding table and lm_head from the target checkpoint."""

import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig


class TargetEmbeddingsAndHead(nn.Module):
    """Frozen target embed_tokens + lm_head, loaded straight from safetensors."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: str = "model.embed_tokens.weight",
        lm_head_key: str = "lm_head.weight",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TargetEmbeddingsAndHead":
        from olmo3_sink import register_olmo3_sink

        register_olmo3_sink()
        config = AutoConfig.from_pretrained(model_path)
        instance = cls(config)

        tie_weights = getattr(config, "tie_word_embeddings", False)
        instance._load_weights(model_path, embed_key, lm_head_key, tie_weights)
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)
        return instance

    def _load_weights(self, model_path, embed_key, lm_head_key, tie_weights):
        wanted = {embed_key} | (set() if tie_weights else {lm_head_key})
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        if index_files:
            with open(index_files[0]) as f:
                weight_map = json.load(f)["weight_map"]
            files = {os.path.join(model_path, weight_map[k]) for k in wanted}
        else:
            sts = glob.glob(os.path.join(model_path, "*.safetensors"))
            if not sts:
                raise FileNotFoundError(f"No safetensors under {model_path}")
            files = set(sts)

        found = set()
        for fp in files:
            with safe_open(fp, framework="pt") as f:
                for k in wanted & set(f.keys()):
                    t = f.get_tensor(k)
                    if k == embed_key:
                        self.embed_tokens.weight.data.copy_(t)
                    else:
                        self.lm_head.weight.data.copy_(t)
                    found.add(k)
        missing = wanted - found
        if missing:
            raise KeyError(f"Keys {missing} not found in {model_path}")
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
