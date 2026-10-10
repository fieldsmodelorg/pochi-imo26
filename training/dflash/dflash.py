"""DFlash Online Training Wrapper — block-wise CE loss with flex attention.

Ported from dflash-train (tf 4.57/torch 2.9) to tf 5.9/torch 2.12. Mechanism
unchanged; one addition over the original: cross-document label masking
(`document_ids` gating of label positions) — in the original, a block whose
anchor sits within `block_size` of a packed-document boundary silently takes
its tail labels from the NEXT document. Tiny fraction of anchors, but exact to
fix, so we fix it.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    _compiled_create_block_mask = torch.compile(create_block_mask)
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None
    _compiled_create_block_mask = None


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    context_doc_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    causal: bool = False,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional (or causal if causal=True).
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
      5. (Packing) Context attention is restricted to the same document.
      6. (SWA) If sliding_window is set, each Q at position anchor_pos + k
         sees context in [anchor_pos + k - sliding_window, anchor_pos).
    """
    if sliding_window is not None and sliding_window <= 0:
        sliding_window = None

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        # All block positions share the same context view: [0, anchor).
        # NOT [0, anchor+k) — that would leak target's gold hidden states
        # at positions anchor..anchor+k-1, which are unavailable at inference.
        is_context = kv_idx < S
        mask_context = is_context & (kv_idx < anchor_pos)

        if sliding_window is not None:
            q_actual_pos = anchor_pos + (q_idx % block_size)
            mask_context = mask_context & (kv_idx >= q_actual_pos - sliding_window)

        if context_doc_ids is not None:
            anchor_doc = context_doc_ids[b, anchor_pos.clamp(max=S - 1)]
            kv_doc = context_doc_ids[b, kv_idx.clamp(max=S - 1)]
            mask_context = mask_context & (anchor_doc == kv_doc)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        if causal:
            q_pos_in_block = q_idx % block_size
            kv_pos_in_block = (kv_idx - S) % block_size
            mask_draft = mask_draft & (kv_pos_in_block <= q_pos_in_block)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return _compiled_create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        greedy_match_threshold: Optional[float] = None,
        use_cce: bool = False,
        sliding_window: Optional[int] = None,
        causal: bool = False,
        focal_gamma: Optional[float] = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.greedy_match_threshold = greedy_match_threshold
        self.use_cce = use_cce
        self.sliding_window = sliding_window
        self.causal = causal
        self.focal_gamma = focal_gamma
        # flex attention kernel requires Q_LEN >= one Q block (128); production
        # shapes (anchors*block_size in the tens of thousands) are always fine,
        # fail loudly on degenerate configs.
        assert num_anchors * block_size >= 128, (
            f"num_anchors*block_size = {num_anchors * block_size} < 128 "
            f"(flex attention minimum Q length)"
        )

        if use_cce:
            from cut_cross_entropy import linear_cross_entropy
            self._linear_cross_entropy = linear_cross_entropy

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions; returns (anchors, keep_mask).

        Always returns exactly self.num_anchors blocks so Q_LEN/KV_LEN
        stay constant across batches (avoids torch.compile recompilations).
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        N = self.num_anchors
        # Causal: labels go up to anchor+bs, need anchor+bs < seq_len
        max_anchor = max(seq_len - bs - (1 if self.causal else 0), 0)

        if self.causal:
            pool_input = loss_mask[:, 1 : max_anchor + bs + 1].float().unsqueeze(1)
        else:
            pool_input = loss_mask[:, : max_anchor + bs].float().unsqueeze(1)
        block_valid = -F.max_pool1d(-pool_input, kernel_size=bs, stride=1).squeeze(1)
        valid = block_valid > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(N, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("No valid anchor positions found — check data preprocessing.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors_sampled = gathered[:, :max_n].sort(dim=1).values

        anchors = torch.zeros(bsz, N, dtype=torch.long, device=device)
        anchors[:, :max_n] = anchors_sampled

        keep_mask = torch.arange(N, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )
        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        noise_mask = noise_ids == self.mask_token_id
        return self.embed_tokens(noise_ids), noise_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        compute_accuracy: bool = True,
        document_ids: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.Tensor] = None,
        greedy_tokens: Optional[torch.Tensor] = None,
        sample_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parallel block-wise training forward pass.

        context_position_ids: optional (B, S) positions of the context tokens
        (per-document reset for packed data). Defaults to arange(S). RoPE
        scores are relative so a per-document offset is equivalent; passing the
        per-document positions keeps the draft numerically aligned with the
        positions it will see at inference.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding, noise_mask = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        if context_position_ids is None:
            context_position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            )
        # Block position k = context position of the anchor + k (same document
        # frame as the context positions).
        anchor_ctx_pos = torch.gather(context_position_ids, 1, anchor_positions)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = (anchor_ctx_pos.unsqueeze(-1) + offsets).view(bsz, -1)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        full_attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=seq_len,
            block_size=self.block_size,
            device=device,
            context_doc_ids=document_ids,
            causal=self.causal,
        )

        if self.sliding_window is not None:
            swa_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
                context_doc_ids=document_ids,
                sliding_window=self.sliding_window,
                causal=self.causal,
            )
        else:
            swa_attn_mask = None

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            noise_mask=noise_mask,
            target_hidden=hidden_states,
            attention_mask=full_attn_mask,
            swa_attention_mask=swa_attn_mask,
        )

        # --- Labels ---
        if self.causal:
            # Causal: position k predicts NEXT token at anchor+k+1
            label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        else:
            # Original: position k predicts token at anchor+k
            label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if not self.causal:
            # Original: skip position 0 (anchor predicts itself)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # --- Cross-document label masking (new vs upstream port) ---
        # A block whose anchor is within block_size of a document boundary
        # would otherwise take its tail labels from the next document.
        if document_ids is not None:
            anchor_docs = torch.gather(
                document_ids, 1, anchor_positions.clamp(max=seq_len - 1)
            )
            label_docs = torch.gather(
                document_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                safe_label_indices,
            )
            weight_mask = weight_mask * (label_docs == anchor_docs.unsqueeze(-1)).float()

        # --- Greedy prefix filtering ---
        # Only train on positions where the sample matches target model's greedy output.
        # At the first mismatch: correct label to greedy, compute loss, stop for rest of block.
        if greedy_tokens is not None:
            if self.causal:
                # Causal: position k predicts anchor+k+1, greedy check at anchor+k
                greedy_offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
            else:
                # Original: position k predicts anchor+k, greedy check at anchor+k-1
                greedy_offsets = torch.arange(self.block_size, device=device).view(1, 1, -1) - 1
            greedy_positions = (anchor_positions.unsqueeze(-1) + greedy_offsets).clamp(0, seq_len - 1)

            greedy_gathered = torch.gather(
                greedy_tokens.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                greedy_positions,
            )  # (B, N, block_size)

            match = (target_ids == greedy_gathered)

            # Relaxed matching: also accept high-probability sample tokens
            if self.greedy_match_threshold is not None and sample_probs is not None:
                probs_gathered = torch.gather(
                    sample_probs.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                    2,
                    greedy_positions,
                )
                match = match | (probs_gathered > self.greedy_match_threshold)

            if not self.causal:
                match[:, :, 0] = True  # k=0 is the anchor itself, force True for cumprod

            # prefix_match[k] = match[0] & ... & match[k]
            prefix_match = match.long().cumprod(dim=-1)

            # include_in_loss[k] = prefix_match[k-1]
            # (include if all PREVIOUS positions matched — the first mismatch
            # itself is included, with its label corrected to greedy)
            greedy_mask = torch.ones_like(prefix_match, dtype=torch.float)
            greedy_mask[:, :, 1:] = prefix_match[:, :, :-1].float()

            weight_mask = weight_mask * greedy_mask

            # Replace labels with greedy tokens everywhere
            target_ids = greedy_gathered

        binary_mask = weight_mask > 0.5

        # --- Loss decay ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.causal else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        # --- Cross entropy ---
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)
        flat_binary_mask = binary_mask.view(-1)

        if self.use_cce:
            cce_targets = target_ids.clone()
            cce_targets[~binary_mask] = -100
            flat_hidden = output_hidden.view(-1, output_hidden.size(-1))
            flat_cce_targets = cce_targets.view(-1)
            assert getattr(self.lm_head, "bias", None) is None
            loss_per_token = self._linear_cross_entropy(
                flat_hidden,
                self.lm_head.weight,
                flat_cce_targets,
                reduction="none",
            )
            if self.focal_gamma is not None and self.focal_gamma > 0:
                p_t = torch.exp(-loss_per_token.detach())
                loss_per_token = (1 - p_t) ** self.focal_gamma * loss_per_token
            valid_token_count = flat_weights.sum() + 1e-6
            loss = (loss_per_token * flat_weights).sum() / valid_token_count
        else:
            logits = self.lm_head(output_hidden)
            flat_logits = logits.view(-1, logits.size(-1))
            loss_per_token = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )
            if self.focal_gamma is not None and self.focal_gamma > 0:
                p_t = torch.exp(-loss_per_token.detach())
                loss_per_token = (1 - p_t) ** self.focal_gamma * loss_per_token
            valid_token_count = flat_weights.sum() + 1e-6
            loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # --- Metrics ---
        if compute_accuracy:
            with torch.no_grad():
                if self.use_cce:
                    flat_hidden_d = output_hidden.view(-1, output_hidden.size(-1))
                    pred_ids = torch.empty(
                        flat_hidden_d.size(0), dtype=torch.long, device=device
                    )
                    acc_chunk_size = 1024
                    for i in range(0, flat_hidden_d.size(0), acc_chunk_size):
                        chunk_logits = self.lm_head(flat_hidden_d[i : i + acc_chunk_size])
                        pred_ids[i : i + acc_chunk_size] = chunk_logits.argmax(dim=-1)
                else:
                    pred_ids = torch.argmax(flat_logits, dim=-1)
                correct = (pred_ids == flat_targets) & flat_binary_mask
                actual_token_count = flat_binary_mask.sum() + 1e-6
                accuracy = correct.sum().float() / actual_token_count

                pred_block = pred_ids.view(bsz, -1, self.block_size)
                target_block = flat_targets.view(bsz, -1, self.block_size)
                mask_block = binary_mask  # (B, N, block_size)

                valid_block_count = block_keep_mask.sum().float() + 1e-6

                metrics = {
                    "accuracy": accuracy,
                    "effective_tokens": flat_binary_mask.sum().float(),
                }

                start_k = 0 if self.causal else 1
                for k in range(start_k, self.block_size):
                    pos_mask = mask_block[:, :, k]
                    pos_count = pos_mask.sum().float() + 1e-6
                    pos_correct = (
                        (pred_block[:, :, k] == target_block[:, :, k]) & pos_mask
                    ).sum().float()
                    metrics[f"acc/pos_{k}"] = pos_correct / pos_count
                    metrics[f"train_ratio/pos_{k}"] = pos_mask.sum().float() / valid_block_count

                if greedy_tokens is not None:
                    prefix_lens = mask_block.sum(dim=-1).float()
                    valid_prefix_lens = prefix_lens[block_keep_mask]
                    metrics["greedy/mean_prefix_len"] = (
                        valid_prefix_lens.sum() / (valid_prefix_lens.numel() + 1e-6)
                    )
                    pre_greedy_mask = (
                        original_loss_mask_gathered
                        * valid_label_mask.float()
                        * block_keep_mask.unsqueeze(-1).float()
                    )
                    if not self.causal:
                        pre_greedy_mask = pre_greedy_mask * (pos_in_block > 0).float()
                    pre_greedy_count = pre_greedy_mask.sum()
                    post_greedy_count = flat_binary_mask.sum().float()
                    metrics["greedy/match_rate"] = post_greedy_count / (pre_greedy_count + 1e-6)
        else:
            metrics = None

        return loss, metrics
