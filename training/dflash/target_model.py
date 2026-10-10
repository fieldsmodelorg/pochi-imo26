"""FSDP2 frozen target model (olmo3_sink) for DFlash training.

Loads the stage-1 olmo3_sink checkpoint via the canonical `olmo3_sink` package
(per-layer RoPE fix + guarded sink init — NEVER load these checkpoints through
stock transformers, tf>=5 silently zeroes trained sinks) with the in-kernel
sink FA3 backend. Hidden states are captured with forward hooks; packing is
expressed through per-document position_ids (olmo3_sink derives the varlen
cu_seqlens metadata from position resets internally).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import print_on_rank0


@dataclass
class TargetOutput:
    hidden_states: torch.Tensor  # (B, S, n_capture * H), GPU
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    last_hidden: Optional[torch.Tensor] = None  # (B, S, H) post final-norm, GPU


class FSDP2TargetModel:
    """FSDP2-sharded frozen olmo3_sink target with hook-based hidden capture."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.capture_layer_ids: Optional[List[int]] = None
        self._captured_hidden: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._last_hidden: Optional[torch.Tensor] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "olmo3_sink_fa3",
        fsdp: bool = True,
    ) -> "FSDP2TargetModel":
        from olmo3_sink import register_olmo3_sink
        from transformers import AutoModelForCausalLM

        register_olmo3_sink()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()
        model.config.use_cache = False

        # The lm_head is never used here (greedy tokens come from a separately
        # loaded head, see target_utils) — replace to avoid (B, S, V) logits.
        model.lm_head = nn.Identity()
        model.requires_grad_(False)

        if fsdp:
            from torch.distributed._composable.fsdp import fully_shard
            from torch.distributed.device_mesh import init_device_mesh

            world = dist.get_world_size()
            local = int(os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))
            if world > local and world % local == 0:
                # Multi-node: HSDP — shard within the node, replicate across
                # nodes. A flat world-wide shard would all-gather every layer
                # over IB on every step; intra-node stays on NVLink.
                mesh = init_device_mesh(
                    "cuda", (world // local, local), mesh_dim_names=("replicate", "shard")
                )
            else:
                mesh = init_device_mesh("cuda", (world,))
            for layer in model.model.layers:
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
        else:
            model.cuda()

        print_on_rank0(
            f"Target model loaded ({attn_implementation}, fsdp={fsdp}): "
            f"{sum(p.numel() for p in model.parameters()):,} params"
        )
        return cls(model)

    def set_capture_layers(self, layer_ids: List[int], capture_final_norm: bool = True) -> None:
        """Register forward hooks on decoder layers (+ final norm for greedy)."""
        self.capture_layer_ids = layer_ids
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for layer_id in layer_ids:
            layer = self.model.model.layers[layer_id]
            self._hooks.append(
                layer.register_forward_hook(self._make_capture_hook(layer_id))
            )
        if capture_final_norm:
            self._hooks.append(
                self.model.model.norm.register_forward_hook(self._norm_capture_hook)
            )

    def _make_capture_hook(self, layer_id: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured_hidden[layer_id] = hidden.detach()
        return hook

    def _norm_capture_hook(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._last_hidden = hidden.detach()

    @torch.no_grad()
    def generate_hidden_states(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> TargetOutput:
        """Forward the frozen target and capture context hidden states.

        position_ids: per-document-reset positions for packed bins; olmo3_sink
        derives varlen cu_seqlens from the resets (packing-metadata reuse), so
        no attention_mask is needed on the FA3 path.
        """
        assert self.capture_layer_ids is not None, "set_capture_layers() first"

        self._captured_hidden.clear()
        self._last_hidden = None
        # Must call through the ROOT module: the FSDP2 root group (embed_tokens,
        # final norm) unshards in the root pre-forward hook — calling the inner
        # Olmo3SinkModel directly leaves those params as bare DTensors
        # ("mixed torch.Tensor and DTensor" error). lm_head is Identity, so the
        # CausalLM wrapper adds no logits cost.
        self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )
        selected = [self._captured_hidden[i] for i in self.capture_layer_ids]
        hidden = torch.cat(selected, dim=-1)
        last_hidden = self._last_hidden
        self._captured_hidden.clear()
        self._last_hidden = None

        return TargetOutput(
            hidden_states=hidden,
            input_ids=input_ids,
            loss_mask=loss_mask,
            last_hidden=last_hidden,
        )
