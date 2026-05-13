import re

import psutil
import torch

_proc = psutil.Process()


def log(seq_log, s: str = ""):
    print(s)
    seq_log.write(s + "\n")
    seq_log.flush()


def log_rollouts(seq_log, tokenizer, rollouts, rewards, question, answer, step):
    SEP = "─" * 80

    gt_match = re.search(r"####\s*([\d,]+)", answer)
    gt_str = gt_match.group(1) if gt_match else "?"
    emoji_row = "  ".join("✅" if r == 1.0 else "❌" for r in rewards)

    log(seq_log, f"\n{SEP}")
    log(seq_log, f"step {step:5d}  |  GT: {gt_str}  |  {emoji_row}")
    log(seq_log, f"Q: {question[:200].replace(chr(10), ' ')}")

    for i, r in enumerate(rollouts):
        tag = "✅" if rewards[i] == 1.0 else "❌"
        n = len(r["concept_vecs"])
        log(
            seq_log,
            f"\n  [{i}] {tag}  Sequence Length: {n}  (Halted by: {r['stop_reason']})",
        )

        # Decode the sheer, highest-probability instinct
        _GRAY = "\033[90m"
        _RESET = "\033[0m"

        if n > 0:
            log(seq_log, f"    🧠 Concept path ({n} steps):")
            line = "       "
            for ids, probs in zip(r["sample_ids_list"], r["sample_probs_list"]):
                top_tok = (
                    tokenizer.decode([ids[0].item()])
                    .replace("\n", "↵")
                    .replace("\t", "⇥")
                )
                alts = [
                    tokenizer.decode([tid]).replace("\n", "↵").replace("\t", "⇥")
                    for tid in ids[1:].tolist()
                ]
                alt_str = _GRAY + "/".join(alts) + _RESET if alts else ""
                line += top_tok + ("/" + alt_str if alt_str else "") + ""
            log(seq_log, line)
        else:
            log(seq_log, f"    🧠 Concept path: (no steps taken)")

        full = r["generated_text"].strip()
        indented = "\n           ".join(full.splitlines()) if full else "(empty)"
        log(seq_log, f"    🗣️  Answer:\n           {indented}")

    log(seq_log, SEP)


def get_memory_gb(device):
    rss_gb = _proc.memory_info().rss / 1e9
    dev = str(device)
    if "cuda" in dev:
        return (
            rss_gb,
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )
    if "mps" in dev:
        try:
            live = torch.mps.current_allocated_memory() / 1e9
            driver = torch.mps.driver_allocated_memory() / 1e9
        except AttributeError:
            live = driver = 0.0
        return rss_gb, live, driver
    return (rss_gb, 0.0, 0.0)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"
