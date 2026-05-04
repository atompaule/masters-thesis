import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_embedding_experiments.algorithms.config import CFG
from src.latent_embedding_experiments.algorithms.soft_thinking import soft_thinking
from src.latent_embedding_experiments.algorithms.soft_thinking_sharpened import (
    soft_thinking_sharpened_aggregate,
    soft_thinking_sharpened_per_token,
)
from src.latent_embedding_experiments.algorithms.solver import geometric_solver
from src.latent_embedding_experiments.algorithms.utils import emit

# =========================================================
# CONFIG
# =========================================================


@dataclass
class ExperimentConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    display_k: int = 20
    log_file: str = (
        "src/latent_embedding_experiments/logs/llama_8b_latent_comparison_concept_descriptions.txt"
    )
    n_interlopers: int = 10
    target_sim: float = 0.90


EXP_CFG = ExperimentConfig()


# =========================================================
# UTIL
# =========================================================


def clean_token(token: str, max_len=12):
    token = token.replace("\n", "\\n").replace("\r", "\\r")
    return token[:max_len] + ".." if len(token) > max_len else token


def format_cell(token, value, width=28, marker=False):
    prefix = "*" if marker else " "
    content = f"{prefix}'{token}' ({value:.3f})"
    return f"{content:<{width}}"


def emit_section(title: str, f, width=120):
    """Bold section divider."""
    emit(f"\n  ┌─ {title} {'─' * max(0, width - len(title) - 5)}", f)


def emit_generation(text: str, f, indent=4, wrap=110):
    """Print generation text wrapped and indented, truncated at 600 chars."""
    pad = " " * indent
    display = text[:600] + "  [...]" if len(text) > 600 else text
    # replace real newlines with a visible marker then re-wrap
    display = display.replace("\n", " ↵ ")
    words = display.split(" ")
    line, lines = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > wrap:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for i, l in enumerate(lines):
        prefix = pad + "│ " if i > 0 else pad
        emit(prefix + l, f)


def emit_jaccard_table(per_word_j: dict, union_j: float, f, indent=4):
    """Print Jaccard scores as a compact aligned table."""
    pad = " " * indent
    col = max(len(w) for w in per_word_j) + 1
    bar_max = 20
    emit(f"{pad}{'Word':<{col}}  {'Jaccard':>7}  Bar", f)
    emit(f"{pad}{'-'*col}  {'-'*7}  {'-'*bar_max}", f)
    for w, v in per_word_j.items():
        bar = "█" * int(v * bar_max) + "░" * (bar_max - int(v * bar_max))
        emit(f"{pad}{w:<{col}}  {v:>7.3f}  {bar}", f)
    emit(f"{pad}{'[union]':<{col}}  {union_j:>7.3f}", f)


def emit_proximity_table(prox: dict, f, indent=4):
    """Print embedding proximity scores as a compact aligned table."""
    pad = " " * indent
    col = max(len(w) for w in prox) + 1
    bar_max = 20
    # scale bar to range within this result set
    vals = [v for k, v in prox.items() if k != "[mean]"]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1e-9
    emit(f"{pad}{'Token':<{col}}  {'CosSim':>7}  Bar (relative)", f)
    emit(f"{pad}{'-'*col}  {'-'*7}  {'-'*bar_max}", f)
    for w, v in prox.items():
        if w == "[mean]":
            continue
        filled = int(((v - lo) / span) * bar_max)
        bar = "█" * filled + "░" * (bar_max - filled)
        emit(f"{pad}{w:<{col}}  {v:>7.4f}  {bar}", f)
    emit(f"{pad}{'[mean]':<{col}}  {prox['[mean]']:>7.4f}", f)


def processability_metrics(
    prefix: str,
    suffix: str,
    vec: torch.Tensor,
    baseline_hidden: list[torch.Tensor] | None = None,
) -> dict:
    """
    Measures how well the model can 'deal with' a concept embedding,
    independent of whether it resembles any discrete token.

    Returns:
        layer_entropy       : entropy of vocab distribution at concept pos, per layer
        layer_norm          : L2 norm of hidden state at concept pos, per layer
        norm_dev_from_base  : |norm - baseline_norm| per layer (None if no baseline)
        resolution_layer    : first layer where entropy < 1.0 nat (nan if never)
        entropy_auc         : area under entropy curve (lower = more processable)
        final_entropy       : entropy at last layer
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    prefix_emb = model.get_input_embeddings()(prefix_ids)
    suffix_emb = model.get_input_embeddings()(suffix_ids)
    vec_in = vec.unsqueeze(0).to(model.dtype) if vec.dim() == 2 else vec.to(model.dtype)

    concept_pos = prefix_ids.shape[1]
    inputs_emb = torch.cat([prefix_emb, vec_in, suffix_emb], dim=1)

    layer_hidden = []

    def make_hook(pos):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            layer_hidden.append(hs[0, pos, :].detach().float())

        return hook

    hooks = [
        layer.register_forward_hook(make_hook(concept_pos))
        for layer in model.model.layers
    ]
    with torch.no_grad():
        model(inputs_embeds=inputs_emb)
    for h in hooks:
        h.remove()

    final_norm = model.model.norm
    lm_head = model.lm_head

    layer_entropy = []
    layer_norm = []

    RESOLUTION_THRESHOLD = 1.0  # nats — tune as needed

    for hs in layer_hidden:
        # Norm of hidden state
        layer_norm.append(hs.norm(p=2).item())

        # Entropy of vocab distribution via logit lens
        with torch.no_grad():
            normed = final_norm(hs.unsqueeze(0).to(model.dtype))
            logits = lm_head(normed)[0].float()
        probs = F.softmax(logits, dim=-1)
        # clamp to avoid log(0)
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        layer_entropy.append(entropy)

    # Summary stats
    resolution_layer = next(
        (i for i, e in enumerate(layer_entropy) if e < RESOLUTION_THRESHOLD),
        float("nan"),
    )
    entropy_auc = sum(layer_entropy) / len(layer_entropy)  # mean entropy across layers

    norm_dev = None
    if baseline_hidden is not None:
        norm_dev = [
            abs(layer_norm[i] - baseline_hidden[i].norm(p=2).item())
            for i in range(len(layer_norm))
        ]

    return {
        "layer_entropy": layer_entropy,
        "layer_norm": layer_norm,
        "norm_dev_from_base": norm_dev,
        "resolution_layer": resolution_layer,
        "entropy_auc": entropy_auc,
        "final_entropy": layer_entropy[-1],
    }


def emit_processability(
    metrics_per_method: dict[str, dict],
    baseline_name: str,
    report_layers: set,
    f,
    indent: int = 4,
):
    pad = " " * indent
    names = list(metrics_per_method.keys())

    # Summary line
    emit(f"\n{pad}Processability summary:", f)
    col = max(len(n) for n in names) + 1
    emit(
        f"{pad}{'Method':<{col}}  {'MeanEntropy':>12}  {'FinalEntropy':>12}  {'ResolutionL':>12}  {'MeanNorm':>9}",
        f,
    )
    emit(f"{pad}{'-'*col}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*9}", f)
    for name, m in metrics_per_method.items():
        res = (
            f"{m['resolution_layer']:.0f}"
            if not isinstance(m["resolution_layer"], float)
            or not __import__("math").isnan(m["resolution_layer"])
            else "never"
        )
        mean_norm = sum(m["layer_norm"]) / len(m["layer_norm"])
        emit(
            f"{pad}{name:<{col}}  {m['entropy_auc']:>12.4f}  {m['final_entropy']:>12.4f}  {res:>12}  {mean_norm:>9.3f}",
            f,
        )

    # Layer-wise entropy table
    emit(f"\n{pad}Layer entropy (lower = model has resolved the concept):", f)
    col_w = 9
    header = f"{pad}{'L':>3}  " + "  ".join(f"{n:>{col_w}}" for n in names)
    emit(header, f)
    emit(f"{pad}{'-'*3}  {'-'*(len(names)*(col_w+2))}", f)
    n_layers = len(next(iter(metrics_per_method.values()))["layer_entropy"])
    report = set(range(0, n_layers, 4))
    report.add(n_layers - 1)
    for i in range(n_layers):
        if i not in report:
            continue
        cells = "  ".join(
            f"{metrics_per_method[n]['layer_entropy'][i]:>{col_w}.3f}" for n in names
        )
        emit(f"{pad}{i:>3}  {cells}", f)

    # Layer-wise norm deviation from baseline
    if metrics_per_method[baseline_name]["norm_dev_from_base"] is None:
        return
    emit(f"\n{pad}Hidden state norm deviation from {baseline_name}:", f)
    emit(header, f)
    emit(f"{pad}{'-'*3}  {'-'*(len(names)*(col_w+2))}", f)
    for i in range(n_layers):
        if i not in report:
            continue
        cells = []
        for n in names:
            dev = metrics_per_method[n]["norm_dev_from_base"]
            cells.append(
                f"{dev[i]:>{col_w}.3f}" if dev is not None else f"{'—':>{col_w}}"
            )
        emit(f"{pad}{i:>3}  {'  '.join(cells)}", f)


def emit_lens_table(lens: list, report_layers: set, f, indent=4):
    """Print logit lens rows with fixed-width token columns."""
    pad = " " * indent
    top_k = len(lens[0][1]) if lens else 5
    col_w = 18
    header = f"{pad}{'L':>3}  " + "  ".join(
        f"{'#'+str(i+1):<{col_w}}" for i in range(top_k)
    )
    emit(header, f)
    emit(f"{pad}{'-'*3}  {'-'*(top_k*(col_w+2))}", f)
    for layer_idx, tokens in lens:
        if layer_idx not in report_layers:
            continue
        cells = "  ".join(
            f"'{t}' {p*100:4.1f}%"[:col_w].ljust(col_w) for t, p in tokens
        )
        emit(f"{pad}{layer_idx:>3}  {cells}", f)


def early_resolution_comparison(
    hidden_states: list[torch.Tensor],
    resolution_layer: float,
    top_k: int = 10,
) -> dict | None:
    """
    At the resolution layer and the final layer, apply final_norm + lm_head
    and compare the resulting distributions.

    Returns None if the entropy threshold was never reached.
    """
    if math.isnan(resolution_layer):
        return None

    res_idx = int(resolution_layer)

    final_norm = model.model.norm
    lm_head = model.lm_head

    def project(hs: torch.Tensor):
        with torch.no_grad():
            normed = final_norm(hs.unsqueeze(0).to(model.dtype))
            logits = lm_head(normed)[0].float()
        probs = F.softmax(logits, dim=-1)
        return probs, logits

    early_probs, early_logits = project(hidden_states[res_idx])
    final_probs, final_logits = project(hidden_states[-1])

    def top_tokens(probs, k):
        top_p, top_i = torch.topk(probs, k)
        return [
            (clean_token(tokenizer.decode([tid.item()])), p.item())
            for tid, p in zip(top_i, top_p)
        ]

    early_top = top_tokens(early_probs, top_k)
    final_top = top_tokens(final_probs, top_k)

    # KL(early || final) — how surprising is the early prediction relative to final?
    kl = (
        (early_probs * ((early_probs + 1e-12) / (final_probs + 1e-12)).log())
        .sum()
        .item()
    )

    # Does the model already commit to the same top-1 token at resolution layer?
    top1_agree = early_probs.argmax().item() == final_probs.argmax().item()

    # Cosine similarity of raw logit vectors (geometry of the prediction space)
    logit_cosine = F.cosine_similarity(
        early_logits.unsqueeze(0), final_logits.unsqueeze(0)
    ).item()

    # Jaccard of top-k token sets (vocabulary overlap)
    early_set = {i.item() for i in torch.topk(early_probs, top_k).indices}
    final_set = {i.item() for i in torch.topk(final_probs, top_k).indices}
    top_k_jaccard = len(early_set & final_set) / len(early_set | final_set)

    return {
        "resolution_layer": res_idx,
        "early_top_k": early_top,
        "final_top_k": final_top,
        "kl_divergence": kl,
        "top1_agree": top1_agree,
        "logit_cosine": logit_cosine,
        "top_k_jaccard": top_k_jaccard,
    }


def emit_early_resolution(
    early_res_per_method: dict[str, dict | None],
    f,
    indent: int = 4,
):
    pad = " " * indent
    col_w = 20

    # Summary table
    names = list(early_res_per_method.keys())
    emit(f"\n{pad}Early-resolution vs final logit distribution:", f)
    col = max(len(n) for n in names) + 1
    emit(
        f"{pad}{'Method':<{col}}  {'ResL':>4}  {'KL(e||f)':>9}  {'LogitCos':>9}"
        f"  {'Top-k Jac':>9}  {'Top1 Agree':>10}",
        f,
    )
    emit(f"{pad}{'-'*col}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}", f)
    for name, res in early_res_per_method.items():
        if res is None:
            emit(f"{pad}{name:<{col}}  {'—':>4}  {'never resolved':>31}", f)
            continue
        emit(
            f"{pad}{name:<{col}}  {res['resolution_layer']:>4}  "
            f"{res['kl_divergence']:>9.4f}  {res['logit_cosine']:>9.4f}  "
            f"{res['top_k_jaccard']:>9.3f}  {'yes' if res['top1_agree'] else 'NO':>10}",
            f,
        )

    # Per-method side-by-side top-k comparison
    for name, res in early_res_per_method.items():
        if res is None:
            continue
        emit(
            f"\n{pad}[{name}] — early (L{res['resolution_layer']}) vs final (L{model.config.num_hidden_layers - 1}):",
            f,
        )
        emit(f"{pad}  {'#':<3}  {'Early token':<{col_w}}  {'Final token':<{col_w}}", f)
        emit(f"{pad}  {'-'*3}  {'-'*col_w}  {'-'*col_w}", f)
        for i, (et, ft) in enumerate(zip(res["early_top_k"], res["final_top_k"])):
            match = "✓" if et[0] == ft[0] else " "
            early_cell = f"'{et[0]}' {et[1]*100:5.2f}%"
            final_cell = f"'{ft[0]}' {ft[1]*100:5.2f}%"
            emit(
                f"{pad}  {i+1:<3}  {match}{early_cell:<{col_w-1}}  {final_cell:<{col_w}}",
                f,
            )


# =========================================================
# MODEL
# =========================================================

print(f"Loading model: {EXP_CFG.model_id}")
tokenizer = AutoTokenizer.from_pretrained(EXP_CFG.model_id)
model = AutoModelForCausalLM.from_pretrained(
    EXP_CFG.model_id, device_map="auto", torch_dtype=torch.bfloat16
)

device = model.device

vocab_embs = model.get_input_embeddings().weight.detach().to(torch.float32)
vocab_embs_norm = F.normalize(vocab_embs, dim=1)


# =========================================================
# VECTOR FORGING
# =========================================================


def build_fake_logits(target_ids, logit_values, vocab_size):
    """
    Build a full-vocabulary logit vector from handcrafted target logits.
    All non-target tokens get a very low logit so top-p will select only targets.
    """
    full_logits = torch.full((vocab_size,), -100.0, device=device)
    for tid, lv in zip(target_ids, logit_values):
        full_logits[tid] = lv
    return full_logits


def build_vectors(target_ids, logit_values, target_probs_scaled):
    vocab_size = vocab_embs.size(0)
    full_logits = build_fake_logits(target_ids, logit_values, vocab_size)

    raw_embs = vocab_embs[target_ids]

    v_soft = soft_thinking(full_logits, vocab_embs)

    with torch.enable_grad():
        v_solver = geometric_solver(full_logits, vocab_embs)

    v_colar = raw_embs.sum(dim=0, keepdim=True) / math.sqrt(len(target_ids))
    v_centroid = raw_embs.mean(dim=0, keepdim=True)

    target_ids_tensor = torch.tensor(target_ids, device=device)
    target_magnitude = v_soft.norm(p=2)

    shared_kwargs = dict(
        vocab_embs=vocab_embs,
        vocab_embs_norm=vocab_embs_norm,
        target_ids=target_ids_tensor,
        target_probs_scaled=target_probs_scaled.float(),
        target_magnitude=target_magnitude,
        n_interlopers=EXP_CFG.n_interlopers,
        target_sim=EXP_CFG.target_sim,
    )

    v_sharpened_agg = soft_thinking_sharpened_aggregate(
        v_soft=v_soft.squeeze(0).float(),
        **shared_kwargs,
    )

    v_sharpened_per = soft_thinking_sharpened_per_token(**shared_kwargs)

    return v_soft, v_solver, v_colar, v_centroid, v_sharpened_agg, v_sharpened_per


def build_baseline_vector(word: str) -> torch.Tensor:
    """Discrete embedding for a single token — the 'what if the model just saw this word' baseline."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    ids2 = tokenizer.encode(word.strip(), add_special_tokens=False)
    ids = ids if len(ids) == 1 else ids2
    if len(ids) != 1:
        raise ValueError(
            f"Baseline word '{word}' fragments into multiple tokens: {ids}"
        )
    return vocab_embs[ids[0]].unsqueeze(0).to(torch.float32)


# =========================================================
# TABLE RENDERING
# =========================================================


def render_similarity_table(vectors, labels, target_ids_set, title, f):
    emit(f"\n--- {title} — * = target token ---", f)

    vectors_unit = F.normalize(vectors, dim=1)
    sims = vectors_unit @ vocab_embs_norm.T

    col_width = 30

    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"
    emit(header[:-2], f)
    emit("-" * len(header), f)

    vals, idxs = torch.topk(sims, EXP_CFG.display_k, dim=1)

    for r in range(EXP_CFG.display_k):
        row = f"{r+1:<4} |"
        for i in range(len(labels)):
            tid = idxs[i, r].item()
            token = clean_token(tokenizer.decode([tid]))
            marker = tid in target_ids_set
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


def render_dot_table(vectors, labels, target_ids_set, title, f):
    emit(f"\n--- {title} — * = target token ---", f)

    vecs = vectors.to(torch.float32)
    emb = vocab_embs.to(torch.float32)
    dots = vecs @ emb.T

    col_width = 30

    header = f"{'Rank':<4} |"
    for label in labels:
        header += f" {label:<{col_width}} |"
    emit(header[:-2], f)
    emit("-" * len(header), f)

    vals, idxs = torch.topk(dots, EXP_CFG.display_k, dim=1)

    for r in range(EXP_CFG.display_k):
        row = f"{r+1:<4} |"
        for i in range(len(labels)):
            tid = idxs[i, r].item()
            token = clean_token(tokenizer.decode([tid]))
            marker = tid in target_ids_set
            row += " " + format_cell(token, vals[i, r].item(), col_width, marker) + " |"
        emit(row[:-2], f)


# =========================================================
# GENERATION
# =========================================================


def splice_and_evaluate(prefix, suffix, vec):
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    prefix_emb = model.get_input_embeddings()(prefix_ids)
    suffix_emb = model.get_input_embeddings()(suffix_ids)

    vec = vec.unsqueeze(0).to(model.dtype)
    inputs = torch.cat([prefix_emb, vec, suffix_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs,
            max_new_tokens=200,
            temperature=0.01,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)

    logits = out.scores[0][0].float()
    probs = F.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, 5)

    lines = []
    for i, (tid, p) in enumerate(zip(top_i, top_p)):
        token = clean_token(tokenizer.decode([tid]))
        lines.append(f"      {i+1}. {token:<12} | {p.item()*100:>5.2f}%")

    return text.strip(), lines


# =========================================================
# ANALYSIS HELPERS
# =========================================================


def generated_token_ids(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_discrete_generations(
    words: list[str], prefix: str, suffix: str
) -> dict[str, tuple[str, set[int]]]:
    """
    Run generation for each concept word individually using its discrete embedding.
    Returns {word: (generated_text, token_id_set)}.
    """
    results = {}
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(w.strip(), add_special_tokens=False)
        vec = vocab_embs[ids[0]].unsqueeze(0).to(torch.float32)
        text, _ = splice_and_evaluate(prefix, suffix, vec)
        results[w.strip()] = (text, set(generated_token_ids(text)))
    return results


def embedding_proximity(text: str, concept_ids: list[int]) -> dict[str, float]:
    """
    For each generated token, compute cosine similarity to each concept token embedding.
    Returns mean-over-generated-tokens for each concept token, plus an overall mean.
    """
    gen_ids = generated_token_ids(text)
    if not gen_ids:
        return {}

    gen_unit = F.normalize(vocab_embs[gen_ids].float(), dim=1)  # (n_gen, d)
    conc_unit = F.normalize(vocab_embs[concept_ids].float(), dim=1)  # (k, d)

    sim = gen_unit @ conc_unit.T  # (n_gen, k)

    out = {}
    for i, tid in enumerate(concept_ids):
        word = tokenizer.decode([tid]).strip()
        out[word] = sim[:, i].mean().item()
    out["[mean]"] = sim.mean().item()
    return out


def emit_hidden_cosine_table(
    all_hidden: dict[str, list[torch.Tensor]],
    baseline_name: str,
    report_layers: set,
    f,
    indent: int = 4,
):
    """
    For each layer, show cosine similarity of each method's hidden state
    to the baseline method's hidden state at that position.
    """
    pad = " " * indent
    names = [n for n in all_hidden if n != baseline_name]
    col_w = 9

    header = f"{pad}{'L':>3}  " + "  ".join(f"{n:>{col_w}}" for n in names)
    emit(header, f)
    emit(f"{pad}{'-'*3}  {'-'*(len(names)*(col_w+2))}", f)

    baseline_vecs = all_hidden[baseline_name]
    n_layers = len(baseline_vecs)

    for i in range(n_layers):
        if i not in report_layers:
            continue
        b = F.normalize(baseline_vecs[i].unsqueeze(0), dim=1)
        cells = []
        for n in names:
            m = F.normalize(all_hidden[n][i].unsqueeze(0), dim=1)
            sim = (b @ m.T).item()
            cells.append(f"{sim:>{col_w}.4f}")
        emit(f"{pad}{i:>3}  {'  '.join(cells)}", f)


def logit_lens(
    prefix: str, suffix: str, vec: torch.Tensor, top_k: int = 5
) -> tuple[list[tuple[int, list[tuple[str, float]]]], list[torch.Tensor]]:
    """
    Returns:
        results     : [(layer_idx, [(token, prob), ...]), ...]
        hidden_states: [tensor(d_model,), ...] one per layer, at concept position
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    prefix_emb = model.get_input_embeddings()(prefix_ids)
    suffix_emb = model.get_input_embeddings()(suffix_ids)
    vec_in = vec.unsqueeze(0).to(model.dtype) if vec.dim() == 2 else vec.to(model.dtype)

    concept_pos = prefix_ids.shape[1]
    inputs_emb = torch.cat([prefix_emb, vec_in, suffix_emb], dim=1)

    layer_hidden = []

    def make_hook(pos):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            layer_hidden.append(hs[0, pos, :].detach().float())

        return hook

    hooks = [
        layer.register_forward_hook(make_hook(concept_pos))
        for layer in model.model.layers
    ]

    with torch.no_grad():
        model(inputs_embeds=inputs_emb)

    for h in hooks:
        h.remove()

    final_norm = model.model.norm
    lm_head = model.lm_head

    results = []
    for i, hs in enumerate(layer_hidden):
        with torch.no_grad():
            normed = final_norm(hs.unsqueeze(0).to(model.dtype))
            logits = lm_head(normed)[0].float()

        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, top_k)
        tokens = [
            (clean_token(tokenizer.decode([tid.item()])), p.item())
            for tid, p in zip(top_i, top_p)
        ]
        results.append((i, tokens))

    return results, layer_hidden


# =========================================================
# EXPERIMENTS
# =========================================================

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 1000)


def validate_experiment_tokens(experiments: list) -> None:
    """
    Check that every word in every experiment tokenizes to a single token.
    Also checks baseline_word. Raises ValueError listing all failures before
    any experiment runs.
    """
    failures = []

    for exp in experiments:
        name = exp["name"]

        # Check baseline
        bw = exp["baseline_word"]
        for candidate in [bw, bw.strip()]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                break
        else:
            failures.append(f"  [{name}] baseline_word '{bw}' → {ids}")

        # Check concept words
        for w in exp["words"]:
            for candidate in [w, w.strip()]:
                ids = tokenizer.encode(candidate, add_special_tokens=False)
                if len(ids) == 1:
                    break
            else:
                failures.append(f"  [{name}] word '{w}' → {ids}")

    if failures:
        raise ValueError(
            f"Token fragmentation in {len(failures)} word(s) — fix before running:\n"
            + "\n".join(failures)
        )

    print(
        f"Token validation passed — {sum(len(e['words']) + 1 for e in experiments)} tokens OK."
    )


experiments = [
    {
        "name": "Mountain — Mordor",
        "baseline_word": " mountain",
        "words": [" mountain", " power", " death", " evil"],
        "logits": [16.0, 15.5, 15.0, 14.5],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "City — Paris",
        "baseline_word": " city",
        "words": [" city", " Paris", " cheese", " wine"],
        "logits": [16.0, 15.5, 15.0, 14.5],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "City — London",
        "baseline_word": " city",
        "words": [" city", " London", " Buckingham", " Palace", " Ben"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "Forest — Middle Earth",
        "baseline_word": " forest",
        "words": [" forest", " wizard", " ring", " elves", " orc"],
        "logits": [16.0, 15.5, 15.0, 14.5, 14.0],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
    {
        "name": "Coffee — Disgusting",
        "baseline_word": " coffee",
        "words": [" coffee", " metallic", " bitter", " unpleasant", " strong"],
        "logits": [15.5, 15.5, 15.0, 14.5],
        "prefix": "Instruction: Vividly describe the concept represented here.\n\nConcept: ",
        "suffix": "\n\nDescription: ",
    },
]

validate_experiment_tokens(experiments)


with open(EXP_CFG.log_file, "w", encoding="utf-8") as f:

    emit("LATENT CONCEPT DESCRIPTION EXPERIMENT", f)
    emit(
        f"Temp: {CFG.temperature} | top_p: {CFG.top_p} | ",
        f,
    )
    emit("", f)

    for exp in experiments:

        emit("\n" + "=" * 120, f)
        emit(f"EXPERIMENT: {exp['name']}", f)
        emit("=" * 120, f)

        # --- Tokenize words ---
        target_ids = []
        clean_words = []

        for w in exp["words"]:
            t = tokenizer.encode(w, add_special_tokens=False)
            if len(t) == 1:
                target_ids.append(t[0])
                clean_words.append(w.strip())
            else:
                t2 = tokenizer.encode(w.strip(), add_special_tokens=False)
                if len(t2) == 1:
                    target_ids.append(t2[0])
                    clean_words.append(w.strip())
                else:
                    raise ValueError(f"Token fragmentation: {w}")

        logit_values = torch.tensor(exp["logits"], device=device)
        target_ids_set = set(target_ids)

        # --- Compute probabilities for display ---
        full_logits = build_fake_logits(target_ids, logit_values, vocab_embs.size(0))
        full_probs = F.softmax(full_logits, dim=-1)
        full_probs_scaled = F.softmax(full_logits / CFG.temperature, dim=-1)
        target_probs_raw = full_probs[target_ids]
        target_probs_scaled = full_probs_scaled[target_ids]

        # --- Input anatomy ---
        input_raw_embs = vocab_embs[target_ids]
        input_mags = torch.norm(input_raw_embs, p=2, dim=1)

        emit(f"\n--- INPUT TOKENS (k={len(target_ids)}) ---", f)
        header_in = f"  {'ID':<6} | {'Token':<12} | {'Logit':<6} | {'Raw P':<8} | {'Scaled P (T={CFG.temperature})':<20} | {'Magnitude':<9}"
        emit(header_in, f)
        emit("  " + "-" * (len(header_in) - 2), f)

        for tid, w, logit, p_raw, p_scaled, mag in zip(
            target_ids,
            clean_words,
            logit_values,
            target_probs_raw,
            target_probs_scaled,
            input_mags,
        ):
            emit(
                f"  {tid:<6} | {w:<12} | {logit.item():>5.1f} | {p_raw.item()*100:>7.2f}% | {p_scaled.item()*100:>19.2f}% | {mag.item():>7.3f}",
                f,
            )

        # --- Pre-forge cross-similarity ---
        input_unit = F.normalize(input_raw_embs, p=2, dim=1)
        cross_sim = input_unit @ input_unit.T
        df_inputs = pd.DataFrame(
            cross_sim.cpu().numpy(), index=clean_words, columns=clean_words
        )

        emit(f"\n--- INPUT CROSS-SIMILARITY ---", f)
        emit(df_inputs.to_string(), f)

        # --- Build vectors ---
        v_soft, v_solver, v_colar, v_centroid, v_sharpened_agg, v_sharpened_per = (
            build_vectors(target_ids, logit_values, target_probs_scaled)
        )
        v_baseline = build_baseline_vector(exp["baseline_word"])

        vectors = torch.cat(
            [
                v_baseline,
                v_soft,
                v_solver,
                v_colar,
                v_centroid,
                v_sharpened_agg,
                v_sharpened_per,
            ],
            dim=0,
        )
        labels = [
            "Baseline",
            "Soft",
            "Solver",
            "CoLaR",
            "Centroid",
            "SharpenedAgg",
            "SharpenedPer",
        ]

        # --- Cross-similarity of output vectors ---
        vectors_unit = F.normalize(vectors, dim=1)
        mags = vectors.norm(p=2, dim=1).cpu().numpy()

        sim_matrix = (vectors_unit @ vectors_unit.T).cpu().numpy()
        df_vecs = pd.DataFrame(sim_matrix, index=labels, columns=labels)
        emit(f"\n--- OUTPUT CROSS-SIMILARITY ---", f)
        for label, mag in zip(labels, mags):
            emit(f"  {label}: |v| = {mag:.3f}", f)
        emit(df_vecs.to_string(), f)

        # --- Tables ---
        render_similarity_table(
            vectors, labels, target_ids_set, "NEAREST NEIGHBORS (COSINE)", f
        )
        render_dot_table(
            vectors, labels, target_ids_set, "NEAREST NEIGHBORS (DOT PRODUCT)", f
        )

        # --- Run discrete baseline generations (one per concept word) ---
        discrete_gens = run_discrete_generations(
            exp["words"], exp["prefix"], exp["suffix"]
        )
        discrete_union = set().union(*[s for _, s in discrete_gens.values()])

        emit_section("DISCRETE BASELINE GENERATIONS", f)
        col = max(len(w) for w in discrete_gens) + 1
        for word, (text, _) in discrete_gens.items():
            snippet = (
                text.replace("\n", " ↵ ")[:280] + "  [...]"
                if len(text) > 280
                else text.replace("\n", " ↵ ")
            )
            emit(f"    {word:<{col}} → {snippet}", f)

        # --- Concept vector generations + analyses ---
        emit_section("CONCEPT DESCRIPTIONS + ANALYSIS", f)

        method_vecs = [
            v_baseline,
            v_soft,
            v_solver,
            v_colar,
            v_centroid,
            v_sharpened_agg,
            v_sharpened_per,
        ]
        method_names = [
            "Baseline",
            "Soft",
            "Solver",
            "CoLaR",
            "Centroid",
            "SharpenedAgg",
            "SharpenedPer",
        ]

        # Run logit lens for all methods upfront so we can cross-compare hidden states
        all_lens = {}
        all_hidden = {}
        for name, vec in zip(method_names, method_vecs):
            lens, hidden = logit_lens(exp["prefix"], exp["suffix"], vec)
            all_lens[name] = lens
            all_hidden[name] = hidden

        n_layers = len(next(iter(all_hidden.values())))
        report_layers = set(range(0, n_layers, 4))
        if (n_layers - 1) % 4 != 0:
            report_layers.add(n_layers - 1)
        # Always include last layer
        report_layers.add(n_layers - 1)

        # Hidden-layer cosine similarity vs baseline
        emit(
            f"\n  ── Hidden-state cosine similarity vs Baseline ──────────────────────",
            f,
        )
        emit_hidden_cosine_table(all_hidden, "Baseline", report_layers, f)

        # --- Processability metrics ---
        baseline_hidden = all_hidden["Baseline"]

        processability_per_method = {}
        for name, vec in zip(method_names, method_vecs):
            processability_per_method[name] = processability_metrics(
                exp["prefix"],
                exp["suffix"],
                vec,
                baseline_hidden=baseline_hidden,
            )

        emit_section("PROCESSABILITY METRICS", f)
        emit_processability(processability_per_method, "Baseline", report_layers, f)

        # --- Early resolution comparison ---
        early_res_per_method = {}
        for name in method_names:
            early_res_per_method[name] = early_resolution_comparison(
                hidden_states=all_hidden[name],
                resolution_layer=processability_per_method[name]["resolution_layer"],
                top_k=10,
            )

        emit_section("EARLY RESOLUTION vs FINAL LOGIT DISTRIBUTION", f)
        emit_early_resolution(early_res_per_method, f)

        for name, vec in zip(method_names, method_vecs):
            text, top5_lines = splice_and_evaluate(exp["prefix"], exp["suffix"], vec)
            method_ids = set(generated_token_ids(text))

            emit(f"\n  ── {name} {'─'*60}", f)

            emit(f"    Generation:", f)
            emit_generation(text, f)

            emit(f"    First-token distribution:", f)
            for line in top5_lines:
                emit(line, f)

            emit(f"    Jaccard overlap:", f)
            per_word_j = {
                w: jaccard(method_ids, s) for w, (_, s) in discrete_gens.items()
            }
            emit_jaccard_table(per_word_j, jaccard(method_ids, discrete_union), f)

            emit(f"    Embedding proximity:", f)
            prox = embedding_proximity(text, target_ids)
            emit_proximity_table(prox, f)

            emit(f"    Logit lens (concept position):", f)
            lens = all_lens[name]
            emit_lens_table(lens, report_layers, f)

            # Always show last layer prominently
            last_layer_tokens = lens[-1][1]
            tok_str = "  ".join(f"'{t}' {p*100:.1f}%" for t, p in last_layer_tokens)
            emit(f"    Last layer (L{n_layers-1}): {tok_str}", f)

    emit("\nDone.", f)

print(f"\nSaved → {EXP_CFG.log_file}")
