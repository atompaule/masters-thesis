#!/usr/bin/env python3
"""
Train a latent head on top of a frozen language model.

Pipeline per token position:
  1. Frozen LM produces hidden states + logits
  2. Latent head projects the final hidden state → latent embedding [d]
  3. Cosine similarities against the full vocabulary → [V]
  4. Softmax over cosine sims → probability distribution
  5. MSE loss ‖p − q‖² between LM softmax p and latent-head softmax q over the vocabulary

Only the latent head's parameters are updated; the LLM is fully frozen.

Usage:
    python -m src.latent_embedding_experiments.training.train_latent_head
    python -m src.latent_embedding_experiments.training.train_latent_head \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --n_epochs 3 --lr 1e-4 --output_dir checkpoints/latent_head
"""

import argparse
import math
import os
from io import TextIOWrapper
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LOG_FILE = "latent_embedding_experiments/logs/llama_8b_latent_head_training.txt"


# ── Logging ────────────────────────────────────────────────────────────────────


def emit(text: str, fh: TextIOWrapper | None = None) -> None:
    print(text)
    if fh is not None:
        fh.write(text + "\n")
        fh.flush()


# ── Latent head ────────────────────────────────────────────────────────────────


class LatentHead(nn.Module):
    """Single linear projection from hidden-state space into the embedding space.

    Kept intentionally small so gradient flow is clean. The output lives in the
    same dimensionality as the input token embeddings, enabling direct cosine
    similarity comparisons with the vocabulary matrix.

    Fixed temperature τ = exp(log_temp) (stored as a buffer, not trained) scales cosine
    similarities before softmax. Set via ``init_temp`` at construction.
    """

    def __init__(self, hidden_dim: int, init_temp: float = 0.6):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Buffer (not Parameter): follows .to(device), no optimizer step, saved in state_dict.
        self.register_buffer(
            "log_temp",
            torch.tensor(math.log(init_temp), dtype=torch.float32),
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., hidden_dim] → [..., hidden_dim]"""
        return self.proj(x)


# ── Data helpers ───────────────────────────────────────────────────────────────


def epoch_batches(
    ds,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    shuffle_seed: int,
    max_samples: int,
    shuffle_buffer: int = 10_000,
    min_tokens: int = 8,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input_ids [B, L], attention_mask [B, L]) for one pass over ds.

    Uses streaming shuffle + take so only max_samples examples are downloaded
    per epoch. Shuffle is approximate (buffer-local) which is fine for training.
    """
    ds_epoch = ds.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed).take(
        max_samples
    )

    def _encode(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor] | None:
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        if enc["input_ids"].size(1) < min_tokens:
            return None
        return enc["input_ids"], enc["attention_mask"]

    buffer: list[str] = []
    for example in ds_epoch:
        text = (example.get("text") or "").strip()
        if len(text) < 64:
            continue
        buffer.append(text)
        if len(buffer) == batch_size:
            batch = _encode(buffer)
            if batch is not None:
                yield batch
            buffer = []

    if buffer:
        batch = _encode(buffer)
        if batch is not None:
            yield batch


# ── Training ───────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_fh = open(args.log_file, "w")

    # ── Frozen language model ──────────────────────────────────────────────────
    emit(f"Loading model: {args.model_id}", log_fh)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Vocabulary embeddings (normalised, static) ─────────────────────────────
    # [V, d] — kept in fp32 for stable cosine similarity computation
    vocab_embs_raw = model.get_input_embeddings().weight.detach().float()  # [V, d]
    vocab_embs_norm = F.normalize(vocab_embs_raw, dim=1)  # [V, d]

    # ── Dataset ────────────────────────────────────────────────────────────────
    emit(f"Loading dataset: {args.dataset} (streaming)", log_fh)
    load_kwargs = {"split": args.split, "streaming": True}
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, **load_kwargs)
    else:
        ds = load_dataset(args.dataset, **load_kwargs)

    total_examples = args.max_samples
    total_batches = math.ceil(total_examples / args.batch_size)
    emit(
        f"Dataset: {total_examples:,} examples  "
        f"→ ~{total_batches:,} batches/epoch  "
        f"× {args.n_epochs} epochs",
        log_fh,
    )
    # ── Latent head ────────────────────────────────────────────────────────────
    hidden_dim = model.config.hidden_size

    # Detect device lazily on first forward pass
    latent_head: LatentHead | None = None
    optimizer: torch.optim.Optimizer | None = None
    model_device = next(model.parameters()).device

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    global_step = 0
    running_loss = 0.0

    for epoch in range(1, args.n_epochs + 1):
        emit(f"\n{'='*60}\nEpoch {epoch}/{args.n_epochs}\n{'='*60}", log_fh)
        batches_this_epoch = 0

        for input_ids, attention_mask in epoch_batches(
            ds,
            tokenizer,
            args.max_length,
            args.batch_size,
            shuffle_seed=epoch,
            max_samples=args.max_samples,
        ):
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)

            # Forward pass through frozen model: hidden [B, L, d], logits [B, L, V]
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            hidden = outputs.hidden_states[-1]  # [B, L, d], bfloat16
            hidden_device = hidden.device
            B, L, _ = hidden.shape

            # Lazy initialisation of latent head on first observed device
            if latent_head is None:
                latent_head = (
                    LatentHead(hidden_dim, init_temp=args.init_temp)
                    .to(hidden_device)
                    .to(torch.float32)
                )
                optimizer = torch.optim.AdamW(latent_head.parameters(), lr=args.lr)
                vocab_embs_norm = vocab_embs_norm.to(hidden_device)
                emit(
                    f"Latent head initialised on {hidden_device} | hidden_dim={hidden_dim} "
                    f"| init τ={latent_head.temperature.item():.4f}",
                    log_fh,
                )

            # Causal shift: position i predicts token i+1 → one row per (b, t)
            flat_len = B * (L - 1)
            lm_logits_flat = outputs.logits[:, :-1, :].reshape(flat_len, -1)
            hidden_flat = hidden[:, :-1, :].reshape(flat_len, -1)
            pred_mask_flat = attention_mask[:, 1:].bool().reshape(-1)

            n_valid = int(pred_mask_flat.sum().item())
            if n_valid == 0:
                continue

            row_chunk = (
                flat_len if args.micro_batch_rows <= 0 else args.micro_batch_rows
            )

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for row_start in range(0, flat_len, row_chunk):
                row_end = min(row_start + row_chunk, flat_len)
                chunk_mask = pred_mask_flat[row_start:row_end]
                if not chunk_mask.any():
                    continue

                lm_logits_valid = lm_logits_flat[row_start:row_end][chunk_mask].float()
                hidden_valid = hidden_flat[row_start:row_end][chunk_mask].float()

                # ── Target p: full LM softmax ─────────────────────────────
                with torch.no_grad():
                    p = F.softmax(lm_logits_valid, dim=-1)  # [n, V]

                # ── Student q: temperature-scaled cosine sims ─────────────
                tau = latent_head.temperature
                latent = F.normalize(latent_head(hidden_valid), dim=1)
                cos_sims = latent @ vocab_embs_norm.T
                q = F.softmax(cos_sims / tau, dim=-1)

                # ── MSE distillation: sum_i (p_i − q_i)², mean over token rows ──
                error = (p - q).abs()
                error_loss = error.sum() / n_valid

                p_min = p.min(dim=-1, keepdim=True).values
                p_max = p.max(dim=-1, keepdim=True).values
                p_norm = (p - p_min) / (p_max - p_min).clamp(min=1e-12)

                cos_sims_magnitude_error = F.relu(cos_sims - 0.2) * (1 - p_norm)
                cos_sims_loss = cos_sims_magnitude_error.sum() / n_valid

                total_loss = error_loss + 0.001 * cos_sims_loss
                total_loss.backward()
                step_loss += float(total_loss.detach().item())

            torch.nn.utils.clip_grad_norm_(latent_head.parameters(), 1.0)
            optimizer.step()

            running_loss += step_loss
            global_step += 1
            batches_this_epoch += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                pct = 100.0 * batches_this_epoch / total_batches
                emit(
                    f"\nepoch {epoch}/{args.n_epochs}  "
                    f"step {global_step:6d}  "
                    f"{pct:5.1f}% of epoch  "
                    f"loss {avg_loss:.4f}  "
                    f"τ {latent_head.temperature.item():.4f}",
                    log_fh,
                )
                # Diagnostic: last position of the first sequence in the batch
                with torch.no_grad():
                    seq_len = (
                        attention_mask[0].sum().item() - 1
                    )  # last valid predict pos
                    diag_h = hidden[0, seq_len - 1].float().unsqueeze(0)
                    diag_l = F.normalize(latent_head(diag_h), dim=1)
                    # Raw dot product of unit vectors ∈ [-1, 1]; cos/τ is used only inside softmax (training)
                    diag_cos = (diag_l @ vocab_embs_norm.T).squeeze(0)
                    diag_lm = outputs.logits[0, seq_len - 1]
                print_top_k_table(
                    tokenizer,
                    diag_cos,
                    diag_lm,
                    k=args.table_k,
                    fh=log_fh,
                    latent_temperature=latent_head.temperature.detach(),
                )
                running_loss = 0.0

            if args.save_every > 0 and global_step % args.save_every == 0:
                _save(
                    latent_head,
                    hidden_dim,
                    args.output_dir,
                    suffix=f"_step{global_step}",
                    fh=log_fh,
                )

        emit(f"Epoch {epoch} complete  ({batches_this_epoch} batches)", log_fh)

    # ── Final checkpoint ───────────────────────────────────────────────────────
    if latent_head is not None:
        _save(latent_head, hidden_dim, args.output_dir, fh=log_fh)
    else:
        emit("No training steps completed — check dataset / filters.", log_fh)

    log_fh.close()


# ── Diagnostics ────────────────────────────────────────────────────────────────


def print_top_k_table(
    tokenizer: AutoTokenizer,
    cos_sims: torch.Tensor,  # [V] raw cosine sims in [-1, 1] (unit latent × unit vocab)
    lm_logits: torch.Tensor,  # [V] raw LM logits for one position
    k: int = 10,
    fh: TextIOWrapper | None = None,
    latent_temperature: torch.Tensor | float | None = None,
) -> None:
    """Print a side-by-side top-k table: latent cosine sims vs. LM logit distribution.

    If ``latent_temperature`` is set, latent probabilities match training:
    ``softmax(cos_sims / τ)``. The printed ``cos`` column stays in [-1, 1].
    """
    cos_sims = cos_sims.float()
    lm_logits = lm_logits.float()

    lm_probs = torch.softmax(lm_logits, dim=-1)
    if latent_temperature is None:
        lat_logits = cos_sims
    else:
        tau = (
            latent_temperature.float()
            if isinstance(latent_temperature, torch.Tensor)
            else latent_temperature
        )
        lat_logits = cos_sims / tau
    lat_probs = torch.softmax(lat_logits, dim=-1)

    top_cos_vals, top_cos_ids = torch.topk(cos_sims, k)
    top_lm_vals, top_lm_ids = torch.topk(lm_logits, k)

    def tok(tid: int) -> str:
        return repr(tokenizer.decode([tid]))

    header = (
        f"{'rank':>4}  {'── latent head (cos sim) ──':30s}  {'cos':>6}  {'p':>7}"
        f"    {'── language head (logits) ──':30s}  {'logit':>7}  {'p':>7}"
    )
    emit(header, fh)
    emit("─" * len(header), fh)
    for i in range(k):
        lat_tok = tok(top_cos_ids[i].item())
        lat_sim = top_cos_vals[i].item()
        lat_p = lat_probs[top_cos_ids[i]].item()
        lm_tok = tok(top_lm_ids[i].item())
        lm_logit = top_lm_vals[i].item()
        lm_p = lm_probs[top_lm_ids[i]].item()
        emit(
            f"{i+1:>4}  {lat_tok:30s}  {lat_sim:>6.4f}  {lat_p:>7.4f}"
            f"    {lm_tok:30s}  {lm_logit:>7.3f}  {lm_p:>7.4f}",
            fh,
        )
    emit("", fh)


def _save(
    head: LatentHead,
    hidden_dim: int,
    output_dir: str,
    suffix: str = "",
    fh: TextIOWrapper | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"latent_head{suffix}.pt")
    torch.save({"state_dict": head.state_dict(), "hidden_dim": hidden_dim}, path)
    emit(f"Saved → {path}", fh)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a latent head on a frozen LLM (MSE between LM softmax and latent-head softmax)"
    )
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    p.add_argument(
        "--dataset", default="DKYoon/SlimPajama-6B", help="HuggingFace dataset name"
    )
    p.add_argument(
        "--dataset_config",
        default=None,
        help="Dataset config/subset (pass empty string to omit)",
    )
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token sequence length per example",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    p.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument(
        "--max_samples",
        type=int,
        default=1_000_000,
        help="Cap dataset to this many examples (None = use all)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of sequences per training step",
    )
    p.add_argument(
        "--micro_batch_rows",
        type=int,
        default=512,
        help=(
            "Max token rows per backward pass after flattening (B×(L−1)). "
            "Splits the [N,V] cosine/MSE graph to reduce VRAM; use 0 for one chunk (full N)."
        ),
    )
    p.add_argument(
        "--init_temp",
        type=float,
        default=0.07,
        help="Fixed τ; cosine sims are divided by τ before vocab softmax (stored as non-trainable buffer)",
    )
    p.add_argument(
        "--log_every", type=int, default=20, help="Loss logging interval (steps)"
    )
    p.add_argument(
        "--table_k",
        type=int,
        default=10,
        help="Rows in the diagnostic similarity table",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save intermediate checkpoint every N steps (0 = disabled)",
    )
    p.add_argument(
        "--output_dir",
        default="/work/utsch/masters-thesis/latent_embedding_experiments/latent_head",
        help="Directory for saved checkpoints",
    )
    p.add_argument("--log_file", default=LOG_FILE, help="Path to the training log file")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
