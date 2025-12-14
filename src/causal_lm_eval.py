#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from accelerate import Accelerator  # type: ignore

    _HAS_ACCELERATE = True
except Exception:
    Accelerator = None  # type: ignore
    _HAS_ACCELERATE = False

# ----------------------------
# Prompt rows (repo CSVs)
# ----------------------------


@dataclass(frozen=True)
class PromptRow:
    template_id: str
    category: str
    template: str
    raw: Dict[str, str]


def _normalize_template_category(raw: str) -> str:
    raw = (raw or "").strip()
    if raw == "misleading":
        return "misleading-moderate"
    if raw == "misleading_extreme":
        return "misleading-extreme"
    return raw


def _split_tags(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.split(";") if t.strip()]


def load_prompt_rows_from_csv(
    csv_path: str,
    categories: Sequence[str],
    templates_per_category: int,
    experiment_name: Optional[str],
) -> List[PromptRow]:
    wanted = set(categories)
    by_cat: Dict[str, List[PromptRow]] = {c: [] for c in categories}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        header_map = {
            h.strip().lower(): h.strip() for h in header if h and h.strip()
        }

        exp_col = None
        for cand in ["experiment_name", "experiment", "experiments"]:
            if cand in header_map:
                exp_col = header_map[cand]
                break

        for row in reader:
            template_name = (
                row.get("template_name") or row.get("name") or ""
            ).strip()
            template_text = (
                row.get("template") or row.get("prompt") or ""
            ).strip()
            raw_cat = (
                row.get("template_category") or row.get("category") or ""
            ).strip()

            if not template_name or not template_text or not raw_cat:
                continue

            cat = _normalize_template_category(raw_cat)
            if cat not in wanted:
                continue

            # Skip encoder-only templates (contain {mask})
            if "{mask}" in template_text:
                continue

            if experiment_name is not None:
                if exp_col is None:
                    raise ValueError(
                        f"--experiment_name was provided ({experiment_name}), "
                        f"but CSV has no experiment_name column. Header: {header}"
                    )
                tags = set(_split_tags(row.get(exp_col, "") or ""))
                if experiment_name not in tags:
                    continue

            by_cat[cat].append(
                PromptRow(
                    template_id=template_name,
                    category=cat,
                    template=template_text,
                    raw=row,
                )
            )

    out: List[PromptRow] = []
    for cat in categories:
        items = sorted(by_cat.get(cat, []), key=lambda r: r.template_id)
        if templates_per_category > 0:
            items = items[:templates_per_category]
        out.extend(items)

    return out


# ----------------------------
# Uniform Answer anchor
# ----------------------------

_ANSWER_ANCHOR_RE = re.compile(r"\banswer\s*:\s*$", re.IGNORECASE)


def ensure_answer_anchor(prompt_text: str) -> str:
    s = prompt_text.rstrip()
    if _ANSWER_ANCHOR_RE.search(s):
        return _ANSWER_ANCHOR_RE.sub("Answer:", s)
    return s + "\nAnswer:"


# ----------------------------
# Chat formatting
# ----------------------------


def format_prompt(
    tokenizer, prompt_text: str, as_chat: bool, system_prompt: Optional[str]
) -> str:
    if not as_chat:
        return prompt_text
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ----------------------------
# Data loading
# ----------------------------


def load_nli_splits(dataset_name: str):
    dataset_name = dataset_name.strip().lower()

    if dataset_name == "rte":
        ds = load_dataset("super_glue", "rte")
        return (
            ds["train"],
            ds["validation"],
            ds["train"].features["label"].names,
            "premise",
            "hypothesis",
        )

    if dataset_name == "cb":
        ds = load_dataset("super_glue", "cb")
        return (
            ds["train"],
            ds["validation"],
            ds["train"].features["label"].names,
            "premise",
            "hypothesis",
        )

    if dataset_name == "mnli":
        ds = load_dataset("glue", "mnli")
        return (
            ds["train"],
            ds["validation_matched"],
            ds["train"].features["label"].names,
            "premise",
            "hypothesis",
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _label_name_norm(s: str) -> str:
    return re.sub(r"[^a-z]+", "_", (s or "").strip().lower()).strip("_")


def find_entailment_label_index(label_names: Sequence[str]) -> Optional[int]:
    """
    Robustly identify entailment index from label names.
    Fixes the common pitfall where 'not_entailment' contains the substring 'entail'.
    """
    if not label_names:
        return None

    norm = [_label_name_norm(x) for x in label_names]

    for i, n in enumerate(norm):
        if n in {"entailment", "entailed"}:
            return i

    for i, n in enumerate(norm):
        if (
            "entail" in n
            and not n.startswith("not_")
            and "not_entail" not in n
            and "non_entail" not in n
        ):
            return i

    return None


def is_entailment_label(
    label: int, label_names: Sequence[str]
) -> Optional[bool]:
    if label is None:
        return None
    if not label_names:
        return None

    idx_ent = find_entailment_label_index(label_names)
    if idx_ent is None:
        return None
    return int(label) == int(idx_ent)


# ----------------------------
# Label word extraction (sec4 vs sec5 CSVs)
# ----------------------------


def _get_row_value_case_insensitive(
    row: Dict[str, str], key: str
) -> Optional[str]:
    if key in row:
        return row.get(key)
    lk = key.lower()
    for k, v in row.items():
        if (k or "").lower() == lk:
            return v
    return None


def extract_label_words_from_row(
    row: Dict[str, str],
    label_names: Sequence[str],
) -> Optional[Dict[str, str]]:
    """
    Try multiple common encodings:
      1) label_words as JSON
      2) per-label columns like label_entailment, entailment, label_0, label_1, ...
    Returns mapping from label_name (original) -> label word string.
    """
    lw = _get_row_value_case_insensitive(row, "label_words")
    if lw:
        lw = lw.strip()
        if lw:
            try:
                obj = json.loads(lw)
                if isinstance(obj, dict):
                    out: Dict[str, str] = {}
                    for name in label_names:
                        nkey = _label_name_norm(name)
                        if name in obj and isinstance(obj[name], str):
                            out[name] = obj[name]
                        elif nkey in obj and isinstance(obj[nkey], str):
                            out[name] = obj[nkey]
                    return out if out else None
            except Exception:
                pass

    out2: Dict[str, str] = {}
    for i, name in enumerate(label_names):
        cands = [
            name,
            name.lower(),
            _label_name_norm(name),
            f"label_{name}",
            f"label_{name.lower()}",
            f"label_{_label_name_norm(name)}",
            f"verbalizer_{name}",
            f"verbalizer_{_label_name_norm(name)}",
            f"target_word_{_label_name_norm(name)}",
            f"label_{i}",
            f"verbalizer_{i}",
            f"target_word_{i}",
        ]
        val = None
        for c in cands:
            val = _get_row_value_case_insensitive(row, c)
            if val is not None and str(val).strip():
                break
        if val is not None and str(val).strip():
            out2[name] = str(val).strip()

    return out2 if out2 else None


def canonicalize_label_word(s: str) -> str:
    """
    Make label words safe as completions by ensuring they contribute at least one token.
    Leading space is usually correct for BPE tokenizers.
    """
    s = (s or "").strip()
    if not s:
        return s
    if not s.startswith(" "):
        s = " " + s
    return s


# ----------------------------
# Few-shot demonstrations
# ----------------------------


def sample_indices_nested(n: int, seed: int, max_k: int) -> List[int]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs[: min(max_k, n)]


def build_fewshot_prefix(
    examples: List[Dict],
    template_text: str,
    label_words: Dict[str, str],
    label_names: Sequence[str],
    premise_key: str,
    hypothesis_key: str,
    yes_means_entailment: bool,
) -> str:
    """
    Few-shot examples are always generated in the "baseline" task (NLI),
    optionally with flipped mapping for the flip condition.
    """
    idx_ent = find_entailment_label_index(label_names)
    if idx_ent is None:
        return ""

    name_ent = label_names[idx_ent]
    other_names = [n for n in label_names if n != name_ent]
    if len(label_names) != 2:
        raise ValueError(
            "This evaluator currently supports few-shot demonstrations only for binary label sets."
        )
    name_not = other_names[0]

    w_ent = canonicalize_label_word(label_words[name_ent])
    w_not = canonicalize_label_word(label_words[name_not])

    parts = []
    for ex in examples:
        gold_ent = is_entailment_label(ex.get("label"), label_names)
        if gold_ent is None:
            continue

        prompt = template_text.format(
            premise=ex[premise_key], hypothesis=ex[hypothesis_key]
        )
        prompt = ensure_answer_anchor(prompt)

        if yes_means_entailment:
            completion = w_ent if gold_ent else w_not
        else:
            completion = w_not if gold_ent else w_ent

        parts.append(prompt + completion)

    return "\n\n".join(parts).strip() + ("\n\n" if parts else "")


# ----------------------------
# Instruction overrides
# ----------------------------


def add_flip_instruction(raw_prompt: str, w_ent: str, w_not: str) -> str:
    """
    Flip instruction is described in terms of the actual scored label words.
    We assume a binary entailment setup where one word corresponds to entailment and the other to not-entailment.
    """
    a = w_ent.strip()
    b = w_not.strip()
    return (
        f"Instruction: For this task, answer '{a}' if the hypothesis is NOT entailed by the premise, "
        f"and answer '{b}' if it IS entailed.\n\n" + raw_prompt
    )


def add_task_switch_instruction(raw_prompt: str) -> str:
    return (
        "Instruction: Ignore any entailment framing. Instead, answer based only on whether the HYPOTHESIS "
        "sentence is grammatical English.\n\n" + raw_prompt
    )


# ----------------------------
# Fast logprob scoring with prompt cache
# ----------------------------


@torch.inference_mode()
def score_candidates_cached(
    model,
    tokenizer,
    prompt_text: str,
    candidates: Dict[str, str],
    device: torch.device,
) -> Dict[str, float]:
    """
    Returns log p(candidate | prompt) for each candidate label word.
    Uses a single forward pass for the prompt, then steps token-by-token for each candidate.
    """
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    out0 = model(prompt_ids, use_cache=True)
    past0 = out0.past_key_values
    last_logits = out0.logits[:, -1, :]
    last_logprobs = torch.log_softmax(last_logits, dim=-1)

    scores: Dict[str, float] = {}

    for label, completion in candidates.items():
        comp = canonicalize_label_word(completion)
        comp_ids = tokenizer(
            comp, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if comp_ids.shape[1] == 0:
            scores[label] = float("-inf")
            continue

        total = 0.0
        past = past0
        cur_logprobs = last_logprobs

        for j in range(comp_ids.shape[1]):
            tok_id = int(comp_ids[0, j].item())
            total += float(cur_logprobs[0, tok_id].item())

            if j < comp_ids.shape[1] - 1:
                step_inp = comp_ids[:, j : j + 1]
                outj = model(step_inp, past_key_values=past, use_cache=True)
                past = outj.past_key_values
                cur_logits = outj.logits[:, -1, :]
                cur_logprobs = torch.log_softmax(cur_logits, dim=-1)

        scores[label] = float(total)

    return scores


def argmax_label(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _get_distributed_env() -> Tuple[int, int]:
    """Return (rank, world_size) inferred from common launcher env vars."""
    world_size = int(
        os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or "1"
    )
    rank = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0")
    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0
    return rank, world_size


def _iter_shard(
    indices: Sequence[int], rank: int, world_size: int
) -> List[int]:
    if world_size <= 1:
        return list(indices)
    return [idx for j, idx in enumerate(indices) if (j % world_size) == rank]


def _trim_to_length(
    input_ids_1d: torch.Tensor, length: int, padding_side: str
) -> torch.Tensor:
    if length <= 0:
        return input_ids_1d[:0]
    if padding_side == "left":
        return input_ids_1d[-length:]
    return input_ids_1d[:length]


@torch.inference_mode()
def score_candidates_lastlogits_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_token_ids: Dict[str, int],
    padding_side: str,
) -> Dict[str, torch.Tensor]:
    """Fast path for single-token candidates: one forward for the whole batch."""
    forward_kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )

    # Some models (including Qwen2.*) support computing only the final logits to
    # avoid materializing [B, L, V] for long sequences.
    # We only ever request 1 position here; if a model returned a larger suffix
    # (L' > 1), the padding-aware indexing below would not apply.
    can_use_logits_keep = padding_side == "left" or bool(
        torch.all(attention_mask.to(torch.bool)).item()
    )
    out = None
    if can_use_logits_keep:
        for arg_name in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(**forward_kwargs, **{arg_name: 1})
                break
            except TypeError:
                out = None

    if out is None:
        out = model(**forward_kwargs)

    logits = out.logits  # [B, L', V]

    # If the model already returned only a suffix of logits, the last position is
    # simply the final timestep.
    if int(logits.shape[1]) == 1:
        last_logits = logits[:, -1, :]  # [B, V]
        last_logprobs = torch.log_softmax(last_logits, dim=-1)

        scores: Dict[str, torch.Tensor] = {}
        for label, tok_id in candidate_token_ids.items():
            scores[label] = last_logprobs[:, int(tok_id)]
        return scores

    lengths = attention_mask.sum(dim=1).to(torch.long)  # [B]
    seq_len = int(attention_mask.shape[1])
    if padding_side == "left":
        last_pos = torch.where(
            lengths > 0,
            torch.full_like(lengths, fill_value=seq_len - 1),
            torch.zeros_like(lengths),
        )
    else:
        last_pos = torch.where(
            lengths > 0,
            torch.clamp(lengths - 1, min=0),
            torch.zeros_like(lengths),
        )
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    last_logits = logits[batch_idx, last_pos, :]  # [B, V]
    last_logprobs = torch.log_softmax(last_logits, dim=-1)

    scores: Dict[str, torch.Tensor] = {}
    for label, tok_id in candidate_token_ids.items():
        scores[label] = last_logprobs[:, int(tok_id)]
    return scores


@torch.inference_mode()
def score_candidates_fullseq_batch(
    model,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    candidate_token_seqs: Dict[str, List[int]],
    padding_side: str,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """General path for multi-token candidates; batches (prompt, candidate) pairs."""
    device = prompt_input_ids.device
    labels = list(candidate_token_seqs.keys())
    n_prompts = int(prompt_input_ids.shape[0])
    prompt_lens = prompt_attention_mask.sum(dim=1).to(torch.long).tolist()
    seqs: List[torch.Tensor] = []
    meta: List[Tuple[int, str, int, List[int]]] = []

    for i in range(n_prompts):
        base = _trim_to_length(
            prompt_input_ids[i], int(prompt_lens[i]), padding_side
        )
        for lab in labels:
            comp_ids = candidate_token_seqs[lab]
            comp = torch.tensor(comp_ids, dtype=torch.long, device=device)
            seq = torch.cat([base, comp], dim=0)
            seqs.append(seq)
            meta.append((i, lab, int(base.shape[0]), comp_ids))

    max_len = max(int(s.shape[0]) for s in seqs) if seqs else 0
    batch = len(seqs)
    if batch == 0:
        return {lab: torch.empty((0,), device=device) for lab in labels}

    pad_id = int(pad_token_id)
    if pad_id < 0:
        pad_id = 0

    input_ids = torch.full(
        (batch, max_len),
        fill_value=int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch, max_len), dtype=torch.long, device=device
    )

    for b, seq in enumerate(seqs):
        L = int(seq.shape[0])
        if padding_side == "left":
            input_ids[b, -L:] = seq
            attention_mask[b, -L:] = 1
        else:
            input_ids[b, :L] = seq
            attention_mask[b, :L] = 1

    # Optional: keep only the last (max_completion_len + 1) logits.
    # This prevents enormous [B, L, V] allocations for long prompts on large-vocab models.
    max_comp_len = max(
        (len(v) for v in candidate_token_seqs.values()), default=0
    )
    logits_keep = int(max_comp_len + 1) if max_comp_len > 0 else 0
    forward_kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    out = None
    can_use_logits_keep = padding_side == "left" or bool(
        torch.all(attention_mask.to(torch.bool)).item()
    )
    if logits_keep > 0 and can_use_logits_keep:
        for arg_name in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(**forward_kwargs, **{arg_name: logits_keep})
                break
            except TypeError:
                out = None

    if out is None:
        out = model(**forward_kwargs)
    logits = out.logits
    logprobs = torch.log_softmax(logits, dim=-1)

    keep_base = 0
    if logits_keep > 0 and int(logits.shape[1]) == logits_keep:
        # Logits correspond to the last `logits_keep` positions of the padded sequence.
        keep_base = max_len - logits_keep

    scores_per_pair = torch.empty((batch,), dtype=torch.float32, device=device)
    for b, (_i, _lab, prompt_len, comp_ids) in enumerate(meta):
        total = 0.0
        # When left padded, positions are shifted by pad amount.
        if padding_side == "left":
            seq_len = int(attention_mask[b].sum().item())
            pad = max_len - seq_len
        else:
            pad = 0

        for j, tok_id in enumerate(comp_ids):
            pos = pad + prompt_len + j - 1
            pos_keep = pos - keep_base
            total += float(logprobs[b, pos_keep, int(tok_id)].item())
        scores_per_pair[b] = total

    # Reshape back to per-prompt scores
    out_scores: Dict[str, torch.Tensor] = {
        lab: torch.full((n_prompts,), float("-inf"), device=device)
        for lab in labels
    }
    for b, (i, lab, _prompt_len, _comp_ids) in enumerate(meta):
        out_scores[lab][i] = scores_per_pair[b]
    return out_scores


def _all_candidates_single_token(
    candidate_token_seqs: Dict[str, List[int]],
) -> bool:
    return all(len(v) == 1 for v in candidate_token_seqs.values())


def _pick_preds_from_scores(
    scores: Dict[str, torch.Tensor], labels_order: Sequence[str]
) -> List[str]:
    # scores[label] -> [B]
    stacked = torch.stack(
        [scores[label] for label in labels_order], dim=1
    )  # [B, K]
    best = torch.argmax(stacked, dim=1).tolist()
    return [labels_order[int(i)] for i in best]


@torch.inference_mode()
def compute_labelword_prior_logprobs(
    model,
    tokenizer,
    label_names: Sequence[str],
    row_lw: Dict[str, str],
    as_chat: bool,
    system_prompt: Optional[str],
    device: torch.device,
    amp_dtype,
) -> Dict[str, float]:
    """Compute log p(label_word | prior_prompt) for each label word.

    The prior prompt is a minimal neutral prompt ("Answer:") run through the same
    chat formatting as the main prompts.
    """
    prior_raw = "Answer:"
    prior_text = format_prompt(tokenizer, prior_raw, as_chat, system_prompt)
    enc = tokenizer(
        [prior_text],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        pad_to_multiple_of=8,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Pre-tokenize candidates.
    candidate_token_seqs: Dict[str, List[int]] = {}
    for ln in label_names:
        v = row_lw[ln]
        candidate_token_seqs[ln] = list(
            tokenizer(v, add_special_tokens=False).input_ids
        )
    single_tok = _all_candidates_single_token(candidate_token_seqs)

    autocast_enabled = device.type == "cuda"
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=autocast_enabled,
    ):
        if single_tok:
            candidate_token_ids = {
                k: int(v[0]) for k, v in candidate_token_seqs.items()
            }
            scores_t = score_candidates_lastlogits_batch(
                model,
                enc["input_ids"],
                enc["attention_mask"],
                candidate_token_ids,
                tokenizer.padding_side,
            )
        else:
            scores_t = score_candidates_fullseq_batch(
                model,
                enc["input_ids"],
                enc["attention_mask"],
                candidate_token_seqs,
                tokenizer.padding_side,
                int(tokenizer.pad_token_id),
            )

    out: Dict[str, float] = {}
    for ln in label_names:
        out[ln] = float(scores_t[ln][0].item())
    return out


def _ece_bins_init(n_bins: int = 10) -> Dict[str, List[float]]:
    return {
        "n": [0.0 for _ in range(n_bins)],
        "conf_sum": [0.0 for _ in range(n_bins)],
        "acc_sum": [0.0 for _ in range(n_bins)],
    }


def _ece_bins_update(
    bins: Dict[str, List[float]],
    confidence: float,
    correct: float,
) -> None:
    n_bins = len(bins["n"])
    c = float(confidence)
    a = float(correct)
    if c < 0.0:
        c = 0.0
    if c > 1.0:
        c = 1.0
    b = min(int(c * n_bins), n_bins - 1)
    bins["n"][b] += 1.0
    bins["conf_sum"][b] += c
    bins["acc_sum"][b] += a


def _ece_bins_finalize(bins: Dict[str, List[float]]) -> Optional[float]:
    total = sum(bins["n"])
    if total <= 0:
        return None
    ece = 0.0
    for n, cs, ac in zip(bins["n"], bins["conf_sum"], bins["acc_sum"]):
        if n <= 0:
            continue
        conf = cs / n
        acc = ac / n
        ece += (n / total) * abs(acc - conf)
    return float(ece)


def _rankdata_avg(values: List[float]) -> List[float]:
    """Ranks with average ranks for ties (1..n)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearmanr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx)
    deny = sum((b - my) ** 2 for b in ry)
    den = (denx * deny) ** 0.5
    if den == 0.0:
        return None
    return float(num / den)


# ----------------------------
# Task-switch baseline (optional)
# ----------------------------


@torch.inference_mode()
def cola_grammatical_baseline(
    hypotheses: Sequence[str],
    device: torch.device,
    model_id: str = "textattack/roberta-base-CoLA",
) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_id)
    clf = AutoModelForSequenceClassification.from_pretrained(model_id).to(
        device
    )
    clf.eval()

    out: List[str] = []
    bs = 32
    for i in range(0, len(hypotheses), bs):
        batch = hypotheses[i : i + bs]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        logits = clf(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        acceptable = probs[:, 1] >= 0.5
        out.extend(["yes" if bool(x) else "no" for x in acceptable])
    return out


# ----------------------------
# Main
# ----------------------------


def safe_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_") or "run"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", required=True, help="Comma-separated HF model ids."
    )
    ap.add_argument("--dataset", default="rte", choices=["rte", "cb", "mnli"])
    ap.add_argument("--prompt_csv", default="data/binary_NLI_prompts.csv")
    ap.add_argument(
        "--experiment_name",
        default=None,
        help="Filter CSV rows by experiment tag (e.g., sec4, sec5).",
    )
    ap.add_argument(
        "--template_categories",
        default="instructive,irrelevant,misleading-moderate,misleading-extreme,null",
    )
    ap.add_argument("--templates_per_category", type=int, default=5)

    ap.add_argument(
        "--modes", default="plain", help="Comma-separated: plain,chat."
    )
    ap.add_argument("--system_prompt", default=None)

    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--shots_list", default="0,4,8,16,32,64,128,256")
    ap.add_argument("--seeds", default="1,2,3,4")

    ap.add_argument("--device_map", default="auto")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size over evaluation examples (per process).",
    )
    ap.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation hint for HF models (best on H100: flash_attention_2).",
    )
    ap.add_argument(
        "--compile",
        default="none",
        choices=["none", "reduce-overhead", "max-autotune"],
        help="Optional torch.compile mode for the LM forward pass.",
    )
    ap.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul on NVIDIA GPUs (good default for H100).",
    )
    ap.add_argument(
        "--length_sort",
        action="store_true",
        help="Sort eval shard by approximate length to reduce padding waste.",
    )

    ap.add_argument("--out_dir", default="runs/causal_spec_eval")
    ap.add_argument(
        "--write_examples",
        dest="write_examples",
        action="store_true",
        default=True,
        help="Write per-example JSONL shards (default: enabled).",
    )
    ap.add_argument(
        "--no_write_examples",
        dest="write_examples",
        action="store_false",
        help="Disable per-example JSONL writing (write only summary.jsonl).",
    )

    ap.add_argument(
        "--experiments",
        default="template,flip",
        help="Comma-separated: template,flip,flip_conflict,flip_demos_only,flip_instruction_only,task_switch.",
    )

    # Default binary verbalizers if the CSV does not specify label words (sec4).
    ap.add_argument("--yes_word", default="yes")
    ap.add_argument("--no_word", default="no")

    # Strictness controls
    ap.add_argument("--require_single_token_label_words", action="store_true")

    args = ap.parse_args()

    rank_env, world_size_env = _get_distributed_env()
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    use_distributed = world_size_env > 1

    accelerator = None
    if use_distributed and _HAS_ACCELERATE:
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = accelerator.device
    else:
        rank = rank_env
        world_size = world_size_env
        if (
            use_distributed
            and dist.is_available()
            and not dist.is_initialized()
        ):
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

    is_main = rank == 0
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    shots_list = [int(x) for x in args.shots_list.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    template_categories = [
        c.strip() for c in args.template_categories.split(",") if c.strip()
    ]
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    if is_main and (0 in shots_list) and (len(seeds) > 1):
        print(
            f"[info] shots=0 requested; running only the first seed for zero-shot (seed={seeds[0]})."
        )

    if args.dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, args.dtype)

    if device.type == "cuda":
        if args.dtype == "float16":
            amp_dtype = torch.float16
        else:
            # On H100, bf16 is typically the best default.
            amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, label_names, premise_key, hypothesis_key = (
        load_nli_splits(args.dataset)
    )
    n_eval = min(args.max_eval, len(val_ds))

    prompt_rows = load_prompt_rows_from_csv(
        csv_path=args.prompt_csv,
        categories=template_categories,
        templates_per_category=args.templates_per_category,
        experiment_name=args.experiment_name,
    )
    if not prompt_rows:
        raise RuntimeError(
            "No prompt rows loaded. Check --prompt_csv, --experiment_name, and category filters."
        )

    # Precompute nested few-shot indices per seed (stable within this script)
    max_shots = max(shots_list) if shots_list else 0
    indices_by_seed = {
        seed: sample_indices_nested(len(train_ds), seed=seed, max_k=max_shots)
        for seed in seeds
    }

    # For binary-only pieces (flip, few-shot demo building)
    if len(label_names) != 2:
        if (
            "flip" in experiments
            or "flip_conflict" in experiments
            or "flip_demos_only" in experiments
            or "flip_instruction_only" in experiments
            or any(s > 0 for s in shots_list)
        ):
            raise ValueError(
                f"Dataset {args.dataset} has {len(label_names)} labels ({label_names}). "
                "This script supports few-shot demos and flip only for binary labels."
            )

    idx_ent = find_entailment_label_index(label_names) if label_names else None
    if len(label_names) == 2 and idx_ent is None:
        raise RuntimeError(
            f"Could not identify entailment label from label_names={label_names}"
        )

    # Summary rows
    summary_rows: List[Dict[str, object]] = []

    for model_id in models:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        # Ensure we can pad for batched scoring.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Use left padding to keep the last tokens aligned across the batch.
        # This enables models that support `logits_to_keep` (e.g., Qwen2.*) to avoid
        # materializing huge [B, L, V] logits for long sequences.
        tokenizer.padding_side = "left"

        attn_impl = (
            None
            if args.attn_implementation == "auto"
            else args.attn_implementation
        )

        # In multi-process mode, each process should own exactly one GPU.
        effective_device_map = None if use_distributed else args.device_map

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=effective_device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )

        if effective_device_map is None:
            model = model.to(device)

        model.eval()

        if (
            args.compile != "none"
            and hasattr(torch, "compile")
            and device.type == "cuda"
        ):
            model = torch.compile(model, mode=args.compile)

        if accelerator is not None:
            model = accelerator.prepare(model)

        # Resolve device after wrapping
        model_device = device
        eval_all = list(range(n_eval))
        eval_indices = _iter_shard(eval_all, rank=rank, world_size=world_size)

        for mode in modes:
            if mode not in {"plain", "chat"}:
                raise ValueError(f"Unknown mode: {mode}")
            as_chat = mode == "chat"

            if (
                as_chat
                and is_main
                and ("instruct" not in model_id.lower())
                and ("chat" not in model_id.lower())
            ):
                print(
                    f"[warn] mode=chat with base-ish model_id={model_id!r}; "
                    "if results look degenerate, try --modes plain or an Instruct model."
                )

            # Task-switch gold can be computed per-shard to avoid broadcasting.
            if "task_switch" in experiments:
                hypotheses_eval = [
                    val_ds[i][hypothesis_key] for i in eval_indices
                ]
                task_switch_gold = cola_grammatical_baseline(
                    hypotheses_eval, device=model_device
                )
            else:
                task_switch_gold = None

            for seed in seeds:
                seed_idxs = indices_by_seed[seed]

                for shots in shots_list:
                    if shots == 0 and seeds and seed != seeds[0]:
                        continue
                    fewshot_examples = [train_ds[i] for i in seed_idxs[:shots]]

                    run_slug = safe_slug(
                        f"{model_id}__{mode}__{args.dataset}__shots{shots}__seed{seed}"
                    )
                    out_jsonl = None
                    if args.write_examples:
                        out_jsonl = os.path.join(
                            args.out_dir,
                            f"{run_slug}.jsonl"
                            if is_main
                            else f"{run_slug}.rank{rank}.jsonl",
                        )

                    acc_by_cat: Dict[str, List[int]] = {}
                    flip_acc_by_cat: Dict[str, List[int]] = {}
                    flip_dir_by_cat: Dict[str, List[int]] = {}
                    flip_conflict_acc_by_cat: Dict[str, List[int]] = {}
                    flip_conflict_dir_by_cat: Dict[str, List[int]] = {}
                    flip_demos_only_acc_by_cat: Dict[str, List[int]] = {}
                    flip_demos_only_dir_by_cat: Dict[str, List[int]] = {}
                    flip_instruction_only_acc_by_cat: Dict[str, List[int]] = {}
                    flip_instruction_only_dir_by_cat: Dict[str, List[int]] = {}

                    task_switch_correct = 0
                    task_switch_total = 0

                    label_words_seen: Dict[str, set] = {
                        ln: set() for ln in label_names
                    }
                    label_word_token_lens_seen: Dict[str, set] = {
                        ln: set() for ln in label_names
                    }
                    all_single_token_label_words = True

                    # Extra diagnostics (aggregated per category, per run).
                    cats = list(template_categories)
                    cat_to_i = {c: i for i, c in enumerate(cats)}
                    num_cats = len(cats)
                    ece_bins = 10

                    # Length-bucket diagnostics (token length of the *input prompt*).
                    # Buckets are inclusive of the lower bound.
                    length_edges = [0, 256, 512, 1024, 2048, 4096, 8192]
                    length_bucket_names = [
                        f"{length_edges[i]}-{length_edges[i+1]-1}"
                        for i in range(len(length_edges) - 1)
                    ] + [f"{length_edges[-1]}+"]
                    num_len_buckets = len(length_bucket_names)

                    def _len_bucket(n_tok: int) -> int:
                        for bi in range(len(length_edges) - 1):
                            if length_edges[bi] <= n_tok < length_edges[bi + 1]:
                                return bi
                        return len(length_edges) - 1

                    # Cache prior logprobs for a given (mode, system_prompt, label_words).
                    prior_cache: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, float]] = {}

                    # Per-condition, per-category aggregates.
                    # Keys are condition names; values are lists/arrays indexed by cat_to_i.
                    cond_names = ["base"]
                    for cname in [
                        "flip",
                        "flip_conflict",
                        "flip_demos_only",
                        "flip_instruction_only",
                    ]:
                        if cname in experiments:
                            cond_names.append(cname)

                    # Per-template accuracies (for template variance + correlations).
                    # Stored only on main rank to avoid duplication.
                    template_acc: Dict[str, Dict[str, Dict[str, float]]] = {
                        c: {} for c in cond_names
                    }

                    # Calibration-like stats are computed over semantic entailment (not flip-correctness).
                    stats_n = {c: [0.0] * num_cats for c in cond_names}
                    stats_abs_margin = {c: [0.0] * num_cats for c in cond_names}
                    stats_entropy = {c: [0.0] * num_cats for c in cond_names}
                    stats_brier = {c: [0.0] * num_cats for c in cond_names}
                    stats_margin_lift = {c: [0.0] * num_cats for c in cond_names}
                    ece_count = {c: [[0.0] * ece_bins for _ in range(num_cats)] for c in cond_names}
                    ece_sum_conf = {c: [[0.0] * ece_bins for _ in range(num_cats)] for c in cond_names}
                    ece_sum_acc = {c: [[0.0] * ece_bins for _ in range(num_cats)] for c in cond_names}

                    # Length-bucket aggregates per condition.
                    len_n = {c: [0.0] * num_len_buckets for c in cond_names}
                    len_correct = {c: [0.0] * num_len_buckets for c in cond_names}
                    len_abs_margin = {c: [0.0] * num_len_buckets for c in cond_names}
                    len_entropy = {c: [0.0] * num_len_buckets for c in cond_names}

                    # Confusion (semantic entailment) for the base condition.
                    base_tp = [0.0] * num_cats
                    base_tn = [0.0] * num_cats
                    base_fp = [0.0] * num_cats
                    base_fn = [0.0] * num_cats

                    # Flip-conditional accuracy and semantic invariance.
                    flip_cond = {
                        c: {
                            "base_correct": [0.0] * num_cats,
                            "base_wrong": [0.0] * num_cats,
                            "flip_correct_and_base_correct": [0.0] * num_cats,
                            "flip_correct_and_base_wrong": [0.0] * num_cats,
                            "invariance_agree": [0.0] * num_cats,
                            "invariance_total": [0.0] * num_cats,
                        }
                        for c in cond_names
                        if c != "base"
                    }

                    # Conflict preference (only meaningful for flip_conflict).
                    conflict_pref = {
                        "demo_consistent": [0.0] * num_cats,
                        "instruction_consistent": [0.0] * num_cats,
                        "total": [0.0] * num_cats,
                    }

                    # Prior margin (per-template prior repeated per-example) for reporting.
                    prior_margin_sum = [0.0] * num_cats
                    prior_margin_n = [0.0] * num_cats

                    # Only write per-run JSONL when --write_examples is enabled.
                    # In distributed mode, non-main ranks write shard files.
                    if out_jsonl is None:
                        f_out_ctx = None
                    else:
                        f_out_ctx = open(out_jsonl, "w", encoding="utf-8")

                    if f_out_ctx is None:
                        f_out = None
                    else:
                        f_out = f_out_ctx

                    try:
                        if args.write_examples:
                            meta = {
                                "model": model_id,
                                "mode": mode,
                                "dataset": args.dataset,
                                "shots": shots,
                                "seed": seed,
                                "n_eval": n_eval,
                                "prompt_csv": args.prompt_csv,
                                "experiment_name": args.experiment_name,
                                "template_categories": template_categories,
                                "templates_per_category": args.templates_per_category,
                                "label_names": label_names,
                                "experiments": experiments,
                                "rank": rank,
                                "world_size": world_size,
                            }
                            if f_out is not None:
                                f_out.write(
                                    json.dumps(
                                        {"__meta__": meta}, ensure_ascii=False
                                    )
                                    + "\n"
                                )

                        for pr in prompt_rows:
                            cat = pr.category

                            pr_eval_indices = eval_indices
                            if args.length_sort:
                                pr_eval_indices = sorted(
                                    eval_indices,
                                    key=lambda i: (
                                        len(val_ds[i][premise_key])
                                        + len(val_ds[i][hypothesis_key])
                                    ),
                                )

                            # Determine label words for this template row
                            row_lw = extract_label_words_from_row(
                                pr.raw, label_names
                            )

                            if row_lw is None:
                                if len(label_names) != 2:
                                    raise ValueError(
                                        "No label words in CSV, and dataset is not binary."
                                    )
                                # Default: entailment vs not-entailment map to yes/no words
                                name_ent = label_names[idx_ent]
                                name_not = label_names[1 - idx_ent]
                                row_lw = {
                                    name_ent: args.yes_word,
                                    name_not: args.no_word,
                                }

                            # Enforce full coverage of label words (sec5 safety).
                            # Allow binary fallback for missing labels using default yes/no mapping.
                            row_lw = {
                                ln: row_lw[ln]
                                for ln in label_names
                                if ln in row_lw
                            }
                            missing = [
                                ln for ln in label_names if ln not in row_lw
                            ]
                            if missing:
                                if len(label_names) == 2:
                                    name_ent = label_names[idx_ent]
                                    name_not = label_names[1 - idx_ent]
                                    defaults = {
                                        name_ent: args.yes_word,
                                        name_not: args.no_word,
                                    }
                                    for ln in missing:
                                        row_lw[ln] = defaults[ln]
                                else:
                                    raise ValueError(
                                        f"Missing label words for {missing} in template {pr.template_id}"
                                    )

                            # Canonicalize
                            row_lw = {
                                k: canonicalize_label_word(v)
                                for k, v in row_lw.items()
                            }

                            # Pre-tokenize candidate label words once per template row.
                            candidate_token_seqs: Dict[str, List[int]] = {}
                            for k, v in row_lw.items():
                                ids = tokenizer(
                                    v, add_special_tokens=False
                                ).input_ids
                                candidate_token_seqs[k] = list(ids)

                            for ln in label_names:
                                if ln in row_lw:
                                    label_words_seen[ln].add(str(row_lw[ln]))
                                if ln in candidate_token_seqs:
                                    label_word_token_lens_seen[ln].add(
                                        int(len(candidate_token_seqs[ln]))
                                    )
                            if not _all_candidates_single_token(
                                candidate_token_seqs
                            ):
                                all_single_token_label_words = False

                            if args.require_single_token_label_words:
                                for k, v in row_lw.items():
                                    ids = tokenizer(
                                        v, add_special_tokens=False
                                    ).input_ids
                                    if len(ids) != 1:
                                        raise ValueError(
                                            f"Label word for {k} is not single-token: {v!r} -> {ids}"
                                        )

                            # Minimal-prompt prior (cached) for label-word bias diagnostics.
                            prior_margin = 0.0
                            cat_i = cat_to_i.get(cat)
                            if (
                                cat_i is not None
                                and len(label_names) == 2
                                and idx_ent is not None
                            ):
                                prior_key = (
                                    mode,
                                    (args.system_prompt or ""),
                                    tuple(str(row_lw[ln]) for ln in label_names),
                                )
                                prior_logps = prior_cache.get(prior_key)
                                if prior_logps is None:
                                    prior_logps = compute_labelword_prior_logprobs(
                                        model=model,
                                        tokenizer=tokenizer,
                                        label_names=label_names,
                                        row_lw=row_lw,
                                        as_chat=as_chat,
                                        system_prompt=args.system_prompt,
                                        device=model_device,
                                        amp_dtype=amp_dtype,
                                    )
                                    prior_cache[prior_key] = prior_logps
                                name_ent = label_names[idx_ent]
                                name_not = label_names[1 - idx_ent]
                                prior_margin = float(
                                    prior_logps[name_ent] - prior_logps[name_not]
                                )

                            # Few-shot prefix in baseline mapping (yes_means_entailment=True)
                            fewshot_prefix_base = ""
                            fewshot_prefix_flip = ""
                            if shots > 0:
                                fewshot_prefix_base = build_fewshot_prefix(
                                    examples=fewshot_examples,
                                    template_text=pr.template,
                                    label_words=row_lw,
                                    label_names=label_names,
                                    premise_key=premise_key,
                                    hypothesis_key=hypothesis_key,
                                    yes_means_entailment=True,
                                )
                                # For the flip condition, make demonstrations consistent with the
                                # flipped instruction (avoid teaching the opposite mapping).
                                fewshot_prefix_flip = build_fewshot_prefix(
                                    examples=fewshot_examples,
                                    template_text=pr.template,
                                    label_words=row_lw,
                                    label_names=label_names,
                                    premise_key=premise_key,
                                    hypothesis_key=hypothesis_key,
                                    yes_means_entailment=False,
                                )

                            correct = 0
                            total = 0
                            correct_flip = 0
                            total_flip = 0
                            correct_flip_dir = 0
                            total_flip_dir = 0
                            correct_flip_conflict = 0
                            total_flip_conflict = 0
                            correct_flip_conflict_dir = 0
                            total_flip_conflict_dir = 0
                            correct_flip_demos_only = 0
                            total_flip_demos_only = 0
                            correct_flip_demos_only_dir = 0
                            total_flip_demos_only_dir = 0
                            correct_flip_instruction_only = 0
                            total_flip_instruction_only = 0
                            correct_flip_instruction_only_dir = 0
                            total_flip_instruction_only_dir = 0

                            labels_order = list(label_names)
                            single_tok = _all_candidates_single_token(
                                candidate_token_seqs
                            )
                            if single_tok:
                                candidate_token_ids = {
                                    k: int(v[0])
                                    for k, v in candidate_token_seqs.items()
                                }

                            bs = max(1, int(args.batch_size))
                            steps = range(0, len(pr_eval_indices), bs)
                            for ofs in tqdm(
                                steps,
                                total=(len(pr_eval_indices) + bs - 1) // bs,
                                desc=f"{pr.template_id} ({run_slug})",
                                leave=False,
                                disable=(not is_main),
                            ):
                                batch_indices = pr_eval_indices[ofs : ofs + bs]
                                batch_ex = [val_ds[i] for i in batch_indices]

                                raw_prompts: List[str] = []
                                raw_prompts_nodemo: List[str] = []
                                raw_prompts_flip: List[str] = []
                                for ex in batch_ex:
                                    base = pr.template.format(
                                        premise=ex[premise_key],
                                        hypothesis=ex[hypothesis_key],
                                    )
                                    base = ensure_answer_anchor(base)
                                    raw_prompts.append(
                                        fewshot_prefix_base + base
                                    )
                                    raw_prompts_nodemo.append(base)
                                    raw_prompts_flip.append(
                                        fewshot_prefix_flip + base
                                    )

                                prompt_texts = [
                                    format_prompt(
                                        tokenizer,
                                        rp,
                                        as_chat,
                                        args.system_prompt,
                                    )
                                    for rp in raw_prompts
                                ]

                                enc = tokenizer(
                                    prompt_texts,
                                    return_tensors="pt",
                                    padding=True,
                                    add_special_tokens=False,
                                    pad_to_multiple_of=8,
                                )
                                enc = {
                                    k: v.to(model_device)
                                    for k, v in enc.items()
                                }

                                base_lens = (
                                    enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
                                )

                                autocast_enabled = model_device.type == "cuda"
                                with torch.autocast(
                                    device_type="cuda",
                                    dtype=amp_dtype,
                                    enabled=autocast_enabled,
                                ):
                                    if single_tok:
                                        scores_t = (
                                            score_candidates_lastlogits_batch(
                                                model,
                                                enc["input_ids"],
                                                enc["attention_mask"],
                                                candidate_token_ids,
                                                tokenizer.padding_side,
                                            )
                                        )
                                    else:
                                        scores_t = (
                                            score_candidates_fullseq_batch(
                                                model,
                                                enc["input_ids"],
                                                enc["attention_mask"],
                                                candidate_token_seqs,
                                                tokenizer.padding_side,
                                                int(tokenizer.pad_token_id),
                                            )
                                        )

                                preds = _pick_preds_from_scores(
                                    scores_t, labels_order
                                )

                                for j, ex in enumerate(batch_ex):
                                    gold = ex.get("label")
                                    if gold is None:
                                        continue

                                    total += 1
                                    gold_name = label_names[int(gold)]
                                    if preds[j] == gold_name:
                                        correct += 1

                                    # Aggregate diagnostics per condition/category.
                                    if cat_i is None:
                                        continue

                                    # Prior margin (binary only) is per-template; count per example.
                                    if (
                                        len(label_names) == 2
                                        and idx_ent is not None
                                        and prior_margin is not None
                                    ):
                                        prior_margin_sum[cat_i] += float(
                                            prior_margin
                                        )
                                        prior_margin_n[cat_i] += 1.0

                                    # Compute base-only semantic confusion (binary entailment).
                                    if len(label_names) == 2 and idx_ent is not None:
                                        name_ent = label_names[idx_ent]
                                        pred_ent = preds[j] == name_ent
                                        gold_ent = is_entailment_label(
                                            int(gold), label_names
                                        )
                                        if gold_ent is not None:
                                            if bool(pred_ent) and bool(gold_ent):
                                                base_tp[cat_i] += 1.0
                                            elif (not bool(pred_ent)) and (
                                                not bool(gold_ent)
                                            ):
                                                base_tn[cat_i] += 1.0
                                            elif bool(pred_ent) and (
                                                not bool(gold_ent)
                                            ):
                                                base_fp[cat_i] += 1.0
                                            elif (not bool(pred_ent)) and bool(
                                                gold_ent
                                            ):
                                                base_fn[cat_i] += 1.0

                                    # Base condition calibration-like stats.
                                    logps = torch.stack(
                                        [
                                            scores_t[k][j]
                                            for k in labels_order
                                        ]
                                    ).float()
                                    probs = torch.softmax(logps, dim=0)
                                    conf = float(probs.max().item())
                                    ent = float(
                                        (
                                            -probs
                                            * torch.log(
                                                probs.clamp_min(1e-12)
                                            )
                                        ).sum().item()
                                    )
                                    # Margin: binary uses logprob margin; multi-class uses max-second-max logprob.
                                    if len(labels_order) == 2:
                                        abs_margin = float(
                                            abs((logps[0] - logps[1]).item())
                                        )
                                    else:
                                        top2 = torch.topk(logps, k=2).values
                                        abs_margin = float(
                                            abs((top2[0] - top2[1]).item())
                                        )
                                    # Brier score on semantic gold label.
                                    y = torch.zeros_like(probs)
                                    gold_idx = labels_order.index(gold_name)
                                    y[gold_idx] = 1.0
                                    brier = float(((probs - y) ** 2).sum().item())

                                    stats_n["base"][cat_i] += 1.0
                                    stats_abs_margin["base"][cat_i] += abs_margin
                                    stats_entropy["base"][cat_i] += ent
                                    stats_brier["base"][cat_i] += brier

                                    # Length-bucket stats (semantic correctness).
                                    try:
                                        n_tok = int(base_lens[j])
                                        bi = _len_bucket(n_tok)
                                        len_n["base"][bi] += 1.0
                                        len_correct["base"][bi] += (
                                            1.0 if preds[j] == gold_name else 0.0
                                        )
                                        len_abs_margin["base"][bi] += abs_margin
                                        len_entropy["base"][bi] += ent
                                    except Exception:
                                        pass

                                    # Margin lift uses binary logprob margin minus prior margin.
                                    if (
                                        len(label_names) == 2
                                        and idx_ent is not None
                                        and prior_margin is not None
                                    ):
                                        name_ent = label_names[idx_ent]
                                        name_not = label_names[1 - idx_ent]
                                        logp_ent = float(scores_t[name_ent][j].item())
                                        logp_not = float(scores_t[name_not][j].item())
                                        stats_margin_lift["base"][cat_i] += (
                                            (logp_ent - logp_not)
                                            - float(prior_margin)
                                        )

                                    # ECE bins (semantic correctness).
                                    is_correct = 1.0 if preds[j] == gold_name else 0.0
                                    bin_i = min(
                                        int(conf * ece_bins), ece_bins - 1
                                    )
                                    ece_count["base"][cat_i][bin_i] += 1.0
                                    ece_sum_conf["base"][cat_i][bin_i] += conf
                                    ece_sum_acc["base"][cat_i][bin_i] += is_correct

                                # Flip experiment (binary only)
                                flip_preds: Optional[List[str]] = None
                                flip_conflict_preds: Optional[List[str]] = None
                                flip_demos_only_preds: Optional[List[str]] = (
                                    None
                                )
                                flip_instruction_only_preds: Optional[
                                    List[str]
                                ] = None

                                flip_scores_t = None
                                flip_conflict_scores_t = None
                                flip_demos_only_scores_t = None
                                flip_instruction_only_scores_t = None
                                do_flip = "flip" in experiments
                                do_flip_conflict = (
                                    "flip_conflict" in experiments
                                )
                                do_flip_demos_only = (
                                    "flip_demos_only" in experiments
                                )
                                do_flip_instruction_only = (
                                    "flip_instruction_only" in experiments
                                )
                                if (
                                    do_flip
                                    or do_flip_conflict
                                    or do_flip_demos_only
                                    or do_flip_instruction_only
                                ):
                                    name_ent = label_names[idx_ent]
                                    name_not = label_names[1 - idx_ent]
                                    w_ent = row_lw[name_ent]
                                    w_not = row_lw[name_not]

                                    # flip = instruction + supporting demos
                                    if do_flip:
                                        flip_texts = [
                                            format_prompt(
                                                tokenizer,
                                                add_flip_instruction(
                                                    rp,
                                                    w_ent=w_ent,
                                                    w_not=w_not,
                                                ),
                                                as_chat,
                                                args.system_prompt,
                                            )
                                            for rp in raw_prompts_flip
                                        ]

                                        enc_f = tokenizer(
                                            flip_texts,
                                            return_tensors="pt",
                                            padding=True,
                                            add_special_tokens=False,
                                            pad_to_multiple_of=8,
                                        )
                                        enc_f = {
                                            k: v.to(model_device)
                                            for k, v in enc_f.items()
                                        }

                                        flip_lens = (
                                            enc_f["attention_mask"].sum(dim=1).detach().cpu().tolist()
                                        )

                                        with torch.autocast(
                                            device_type="cuda",
                                            dtype=amp_dtype,
                                            enabled=autocast_enabled,
                                        ):
                                            if single_tok:
                                                flip_scores_t = score_candidates_lastlogits_batch(
                                                    model,
                                                    enc_f["input_ids"],
                                                    enc_f["attention_mask"],
                                                    candidate_token_ids,
                                                    tokenizer.padding_side,
                                                )
                                            else:
                                                flip_scores_t = score_candidates_fullseq_batch(
                                                    model,
                                                    enc_f["input_ids"],
                                                    enc_f["attention_mask"],
                                                    candidate_token_seqs,
                                                    tokenizer.padding_side,
                                                    int(
                                                        tokenizer.pad_token_id
                                                    ),
                                                )

                                        flip_preds = _pick_preds_from_scores(
                                            flip_scores_t, labels_order
                                        )

                                    # flip_conflict = instruction + baseline demos (contradictory)
                                    if do_flip_conflict:
                                        if do_flip and shots == 0:
                                            # No demonstrations: prompts are identical, reuse.
                                            flip_conflict_preds = flip_preds
                                            flip_conflict_scores_t = flip_scores_t
                                        else:
                                            flip_conflict_texts = [
                                                format_prompt(
                                                    tokenizer,
                                                    add_flip_instruction(
                                                        rp,
                                                        w_ent=w_ent,
                                                        w_not=w_not,
                                                    ),
                                                    as_chat,
                                                    args.system_prompt,
                                                )
                                                for rp in raw_prompts
                                            ]

                                            enc_fc = tokenizer(
                                                flip_conflict_texts,
                                                return_tensors="pt",
                                                padding=True,
                                                add_special_tokens=False,
                                                pad_to_multiple_of=8,
                                            )
                                            enc_fc = {
                                                k: v.to(model_device)
                                                for k, v in enc_fc.items()
                                            }

                                            flip_conflict_lens = (
                                                enc_fc["attention_mask"].sum(dim=1).detach().cpu().tolist()
                                            )

                                            with torch.autocast(
                                                device_type="cuda",
                                                dtype=amp_dtype,
                                                enabled=autocast_enabled,
                                            ):
                                                if single_tok:
                                                    flip_conflict_scores_t = score_candidates_lastlogits_batch(
                                                        model,
                                                        enc_fc["input_ids"],
                                                        enc_fc[
                                                            "attention_mask"
                                                        ],
                                                        candidate_token_ids,
                                                        tokenizer.padding_side,
                                                    )
                                                else:
                                                    flip_conflict_scores_t = score_candidates_fullseq_batch(
                                                        model,
                                                        enc_fc["input_ids"],
                                                        enc_fc[
                                                            "attention_mask"
                                                        ],
                                                        candidate_token_seqs,
                                                        tokenizer.padding_side,
                                                        int(
                                                            tokenizer.pad_token_id
                                                        ),
                                                    )

                                            flip_conflict_preds = (
                                                _pick_preds_from_scores(
                                                    flip_conflict_scores_t,
                                                    labels_order,
                                                )
                                            )

                                    # flip_demos_only = flipped demos, no instruction
                                    if do_flip_demos_only:
                                        if shots == 0:
                                            # No demos: this condition is just the baseline prompt.
                                            flip_demos_only_preds = preds
                                            flip_demos_only_scores_t = scores_t
                                        else:
                                            flip_demos_only_texts = [
                                                format_prompt(
                                                    tokenizer,
                                                    rp,
                                                    as_chat,
                                                    args.system_prompt,
                                                )
                                                for rp in raw_prompts_flip
                                            ]

                                            enc_fdo = tokenizer(
                                                flip_demos_only_texts,
                                                return_tensors="pt",
                                                padding=True,
                                                add_special_tokens=False,
                                                pad_to_multiple_of=8,
                                            )
                                            enc_fdo = {
                                                k: v.to(model_device)
                                                for k, v in enc_fdo.items()
                                            }

                                            flip_demos_only_lens = (
                                                enc_fdo["attention_mask"].sum(dim=1).detach().cpu().tolist()
                                            )

                                            with torch.autocast(
                                                device_type="cuda",
                                                dtype=amp_dtype,
                                                enabled=autocast_enabled,
                                            ):
                                                if single_tok:
                                                    flip_demos_only_scores_t = score_candidates_lastlogits_batch(
                                                        model,
                                                        enc_fdo["input_ids"],
                                                        enc_fdo[
                                                            "attention_mask"
                                                        ],
                                                        candidate_token_ids,
                                                        tokenizer.padding_side,
                                                    )
                                                else:
                                                    flip_demos_only_scores_t = score_candidates_fullseq_batch(
                                                        model,
                                                        enc_fdo["input_ids"],
                                                        enc_fdo[
                                                            "attention_mask"
                                                        ],
                                                        candidate_token_seqs,
                                                        tokenizer.padding_side,
                                                        int(
                                                            tokenizer.pad_token_id
                                                        ),
                                                    )

                                            flip_demos_only_preds = (
                                                _pick_preds_from_scores(
                                                    flip_demos_only_scores_t,
                                                    labels_order,
                                                )
                                            )

                                    # flip_instruction_only = instruction, no demos (even if shots>0)
                                    if do_flip_instruction_only:
                                        if (
                                            do_flip or do_flip_conflict
                                        ) and shots == 0:
                                            # In zero-shot, all flip-instruction prompts are identical; reuse.
                                            flip_instruction_only_preds = (
                                                flip_preds
                                                if flip_preds is not None
                                                else flip_conflict_preds
                                            )
                                            flip_instruction_only_scores_t = (
                                                flip_scores_t
                                                if flip_scores_t is not None
                                                else flip_conflict_scores_t
                                            )
                                        else:
                                            flip_io_texts = [
                                                format_prompt(
                                                    tokenizer,
                                                    add_flip_instruction(
                                                        rp,
                                                        w_ent=w_ent,
                                                        w_not=w_not,
                                                    ),
                                                    as_chat,
                                                    args.system_prompt,
                                                )
                                                for rp in raw_prompts_nodemo
                                            ]

                                            enc_fio = tokenizer(
                                                flip_io_texts,
                                                return_tensors="pt",
                                                padding=True,
                                                add_special_tokens=False,
                                                pad_to_multiple_of=8,
                                            )
                                            enc_fio = {
                                                k: v.to(model_device)
                                                for k, v in enc_fio.items()
                                            }

                                            flip_instruction_only_lens = (
                                                enc_fio["attention_mask"].sum(dim=1).detach().cpu().tolist()
                                            )

                                            with torch.autocast(
                                                device_type="cuda",
                                                dtype=amp_dtype,
                                                enabled=autocast_enabled,
                                            ):
                                                if single_tok:
                                                    flip_instruction_only_scores_t = score_candidates_lastlogits_batch(
                                                        model,
                                                        enc_fio["input_ids"],
                                                        enc_fio[
                                                            "attention_mask"
                                                        ],
                                                        candidate_token_ids,
                                                        tokenizer.padding_side,
                                                    )
                                                else:
                                                    flip_instruction_only_scores_t = score_candidates_fullseq_batch(
                                                        model,
                                                        enc_fio["input_ids"],
                                                        enc_fio[
                                                            "attention_mask"
                                                        ],
                                                        candidate_token_seqs,
                                                        tokenizer.padding_side,
                                                        int(
                                                            tokenizer.pad_token_id
                                                        ),
                                                    )

                                            flip_instruction_only_preds = _pick_preds_from_scores(
                                                flip_instruction_only_scores_t,
                                                labels_order,
                                            )

                                    for j, ex in enumerate(batch_ex):
                                        gold = ex.get("label")
                                        gold_name = (
                                            None
                                            if gold is None
                                            else label_names[int(gold)]
                                        )
                                        gold_ent = (
                                            None
                                            if gold is None
                                            else is_entailment_label(
                                                int(gold), label_names
                                            )
                                        )
                                        flip_gold_name = None
                                        if gold_ent is not None:
                                            flip_gold_name = (
                                                name_not
                                                if bool(gold_ent)
                                                else name_ent
                                            )

                                        expected = (
                                            name_not
                                            if preds[j] == name_ent
                                            else name_ent
                                        )

                                        if do_flip and flip_preds is not None:
                                            # Calibration-like stats for flip condition (semantic gold).
                                            if (
                                                flip_scores_t is not None
                                                and gold_name is not None
                                                and cat_i is not None
                                            ):
                                                logps_f = torch.stack(
                                                    [
                                                        flip_scores_t[k][j]
                                                        for k in labels_order
                                                    ]
                                                ).float()
                                                probs_f = torch.softmax(
                                                    logps_f, dim=0
                                                )
                                                conf_f = float(
                                                    probs_f.max().item()
                                                )
                                                ent_f = float(
                                                    (
                                                        -probs_f
                                                        * torch.log(
                                                            probs_f.clamp_min(
                                                                1e-12
                                                            )
                                                        )
                                                    ).sum().item()
                                                )
                                                if len(labels_order) == 2:
                                                    abs_margin_f = float(
                                                        abs(
                                                            (
                                                                logps_f[0]
                                                                - logps_f[1]
                                                            ).item()
                                                        )
                                                    )
                                                else:
                                                    top2 = torch.topk(
                                                        logps_f, k=2
                                                    ).values
                                                    abs_margin_f = float(
                                                        abs(
                                                            (
                                                                top2[0]
                                                                - top2[1]
                                                            ).item()
                                                        )
                                                    )
                                                y = torch.zeros_like(probs_f)
                                                gold_idx = labels_order.index(
                                                    gold_name
                                                )
                                                y[gold_idx] = 1.0
                                                brier_f = float(
                                                    ((probs_f - y) ** 2)
                                                    .sum()
                                                    .item()
                                                )
                                                stats_n["flip"][cat_i] += 1.0
                                                stats_abs_margin["flip"][
                                                    cat_i
                                                ] += abs_margin_f
                                                stats_entropy["flip"][
                                                    cat_i
                                                ] += ent_f
                                                stats_brier["flip"][cat_i] += (
                                                    brier_f
                                                )

                                                # Length-bucket stats (semantic correctness).
                                                try:
                                                    n_tok = int(flip_lens[j])
                                                    bi = _len_bucket(n_tok)
                                                    len_n["flip"][bi] += 1.0
                                                    len_correct["flip"][bi] += (
                                                        1.0
                                                        if flip_preds[j]
                                                        == gold_name
                                                        else 0.0
                                                    )
                                                    len_abs_margin["flip"][bi] += abs_margin_f
                                                    len_entropy["flip"][bi] += ent_f
                                                except Exception:
                                                    pass
                                                if prior_margin is not None and len(label_names) == 2:
                                                    logp_ent = float(
                                                        flip_scores_t[
                                                            name_ent
                                                        ][j].item()
                                                    )
                                                    logp_not = float(
                                                        flip_scores_t[
                                                            name_not
                                                        ][j].item()
                                                    )
                                                    stats_margin_lift[
                                                        "flip"
                                                    ][cat_i] += (
                                                        (logp_ent - logp_not)
                                                        - float(prior_margin)
                                                    )
                                                is_correct_sem = (
                                                    1.0
                                                    if flip_preds[j]
                                                    == gold_name
                                                    else 0.0
                                                )
                                                bin_i = min(
                                                    int(conf_f * ece_bins),
                                                    ece_bins - 1,
                                                )
                                                ece_count["flip"][cat_i][
                                                    bin_i
                                                ] += 1.0
                                                ece_sum_conf["flip"][cat_i][
                                                    bin_i
                                                ] += conf_f
                                                ece_sum_acc["flip"][cat_i][
                                                    bin_i
                                                ] += is_correct_sem

                                            if flip_gold_name is not None:
                                                total_flip += 1
                                                if (
                                                    flip_preds[j]
                                                    == flip_gold_name
                                                ):
                                                    correct_flip += 1

                                                # Conditional flip accuracy.
                                                base_is_correct = (
                                                    gold_name is not None
                                                    and preds[j] == gold_name
                                                )
                                                flip_is_correct = (
                                                    flip_preds[j]
                                                    == flip_gold_name
                                                )
                                                key = "flip"
                                                if key in flip_cond:
                                                    if base_is_correct:
                                                        flip_cond[key][
                                                            "base_correct"
                                                        ][cat_i] += 1.0
                                                        if flip_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_correct"
                                                            ][cat_i] += 1.0
                                                    else:
                                                        flip_cond[key][
                                                            "base_wrong"
                                                        ][cat_i] += 1.0
                                                        if flip_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_wrong"
                                                            ][cat_i] += 1.0

                                                # Semantic invariance: map flip prediction back to semantic label.
                                                try:
                                                    base_pred_idx = labels_order.index(
                                                        preds[j]
                                                    )
                                                    flip_pred_idx = labels_order.index(
                                                        flip_preds[j]
                                                    )
                                                    flip_sem_idx = (
                                                        1 - flip_pred_idx
                                                        if len(labels_order)
                                                        == 2
                                                        else flip_pred_idx
                                                    )
                                                    if key in flip_cond:
                                                        flip_cond[key][
                                                            "invariance_total"
                                                        ][cat_i] += 1.0
                                                        if (
                                                            flip_sem_idx
                                                            == base_pred_idx
                                                        ):
                                                            flip_cond[key][
                                                                "invariance_agree"
                                                            ][cat_i] += 1.0
                                                except Exception:
                                                    pass
                                            total_flip_dir += 1
                                            if flip_preds[j] == expected:
                                                correct_flip_dir += 1

                                        if (
                                            do_flip_conflict
                                            and flip_conflict_preds is not None
                                        ):
                                            if (
                                                flip_conflict_scores_t
                                                is not None
                                                and gold_name is not None
                                                and cat_i is not None
                                            ):
                                                logps_fc = torch.stack(
                                                    [
                                                        flip_conflict_scores_t[k][j]
                                                        for k in labels_order
                                                    ]
                                                ).float()
                                                probs_fc = torch.softmax(
                                                    logps_fc, dim=0
                                                )
                                                conf_fc = float(
                                                    probs_fc.max().item()
                                                )
                                                ent_fc = float(
                                                    (
                                                        -probs_fc
                                                        * torch.log(
                                                            probs_fc.clamp_min(
                                                                1e-12
                                                            )
                                                        )
                                                    ).sum().item()
                                                )
                                                if len(labels_order) == 2:
                                                    abs_margin_fc = float(
                                                        abs(
                                                            (
                                                                logps_fc[0]
                                                                - logps_fc[1]
                                                            ).item()
                                                        )
                                                    )
                                                else:
                                                    top2 = torch.topk(
                                                        logps_fc, k=2
                                                    ).values
                                                    abs_margin_fc = float(
                                                        abs(
                                                            (
                                                                top2[0]
                                                                - top2[1]
                                                            ).item()
                                                        )
                                                    )
                                                y = torch.zeros_like(probs_fc)
                                                gold_idx = labels_order.index(
                                                    gold_name
                                                )
                                                y[gold_idx] = 1.0
                                                brier_fc = float(
                                                    ((probs_fc - y) ** 2)
                                                    .sum()
                                                    .item()
                                                )
                                                stats_n[
                                                    "flip_conflict"
                                                ][cat_i] += 1.0
                                                stats_abs_margin[
                                                    "flip_conflict"
                                                ][cat_i] += abs_margin_fc
                                                stats_entropy[
                                                    "flip_conflict"
                                                ][cat_i] += ent_fc
                                                stats_brier[
                                                    "flip_conflict"
                                                ][cat_i] += brier_fc

                                                # Length-bucket stats (semantic correctness).
                                                try:
                                                    n_tok = int(flip_conflict_lens[j])
                                                    bi = _len_bucket(n_tok)
                                                    len_n["flip_conflict"][bi] += 1.0
                                                    len_correct["flip_conflict"][bi] += (
                                                        1.0
                                                        if flip_conflict_preds[j]
                                                        == gold_name
                                                        else 0.0
                                                    )
                                                    len_abs_margin["flip_conflict"][bi] += abs_margin_fc
                                                    len_entropy["flip_conflict"][bi] += ent_fc
                                                except Exception:
                                                    pass
                                                if prior_margin is not None and len(label_names) == 2:
                                                    logp_ent = float(
                                                        flip_conflict_scores_t[
                                                            name_ent
                                                        ][j].item()
                                                    )
                                                    logp_not = float(
                                                        flip_conflict_scores_t[
                                                            name_not
                                                        ][j].item()
                                                    )
                                                    stats_margin_lift[
                                                        "flip_conflict"
                                                    ][cat_i] += (
                                                        (logp_ent - logp_not)
                                                        - float(prior_margin)
                                                    )
                                                is_correct_sem = (
                                                    1.0
                                                    if flip_conflict_preds[j]
                                                    == gold_name
                                                    else 0.0
                                                )
                                                bin_i = min(
                                                    int(conf_fc * ece_bins),
                                                    ece_bins - 1,
                                                )
                                                ece_count[
                                                    "flip_conflict"
                                                ][cat_i][bin_i] += 1.0
                                                ece_sum_conf[
                                                    "flip_conflict"
                                                ][cat_i][bin_i] += conf_fc
                                                ece_sum_acc[
                                                    "flip_conflict"
                                                ][cat_i][bin_i] += is_correct_sem

                                            if flip_gold_name is not None:
                                                total_flip_conflict += 1
                                                if (
                                                    flip_conflict_preds[j]
                                                    == flip_gold_name
                                                ):
                                                    correct_flip_conflict += 1

                                                # Conditional flip accuracy + invariance.
                                                base_is_correct = (
                                                    gold_name is not None
                                                    and preds[j] == gold_name
                                                )
                                                fc_is_correct = (
                                                    flip_conflict_preds[j]
                                                    == flip_gold_name
                                                )
                                                key = "flip_conflict"
                                                if key in flip_cond:
                                                    if base_is_correct:
                                                        flip_cond[key][
                                                            "base_correct"
                                                        ][cat_i] += 1.0
                                                        if fc_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_correct"
                                                            ][cat_i] += 1.0
                                                    else:
                                                        flip_cond[key][
                                                            "base_wrong"
                                                        ][cat_i] += 1.0
                                                        if fc_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_wrong"
                                                            ][cat_i] += 1.0
                                                    try:
                                                        base_pred_idx = labels_order.index(
                                                            preds[j]
                                                        )
                                                        fc_pred_idx = labels_order.index(
                                                            flip_conflict_preds[j]
                                                        )
                                                        fc_sem_idx = (
                                                            1 - fc_pred_idx
                                                            if len(labels_order)
                                                            == 2
                                                            else fc_pred_idx
                                                        )
                                                        flip_cond[key][
                                                            "invariance_total"
                                                        ][cat_i] += 1.0
                                                        if (
                                                            fc_sem_idx
                                                            == base_pred_idx
                                                        ):
                                                            flip_cond[key][
                                                                "invariance_agree"
                                                            ][cat_i] += 1.0
                                                    except Exception:
                                                        pass

                                                # Conflict preference: demos (gold) vs instruction (flip_gold).
                                                if (
                                                    gold_name is not None
                                                    and cat_i is not None
                                                ):
                                                    conflict_pref["total"][
                                                        cat_i
                                                    ] += 1.0
                                                    if (
                                                        flip_conflict_preds[
                                                            j
                                                        ]
                                                        == gold_name
                                                    ):
                                                        conflict_pref[
                                                            "demo_consistent"
                                                        ][cat_i] += 1.0
                                                    elif (
                                                        flip_conflict_preds[
                                                            j
                                                        ]
                                                        == flip_gold_name
                                                    ):
                                                        conflict_pref[
                                                            "instruction_consistent"
                                                        ][cat_i] += 1.0
                                            total_flip_conflict_dir += 1
                                            if (
                                                flip_conflict_preds[j]
                                                == expected
                                            ):
                                                correct_flip_conflict_dir += 1

                                        if (
                                            do_flip_demos_only
                                            and flip_demos_only_preds
                                            is not None
                                        ):
                                            if (
                                                flip_demos_only_scores_t
                                                is not None
                                                and gold_name is not None
                                                and cat_i is not None
                                            ):
                                                logps_fdo = torch.stack(
                                                    [
                                                        flip_demos_only_scores_t[k][j]
                                                        for k in labels_order
                                                    ]
                                                ).float()
                                                probs_fdo = torch.softmax(
                                                    logps_fdo, dim=0
                                                )
                                                conf_fdo = float(
                                                    probs_fdo.max().item()
                                                )
                                                ent_fdo = float(
                                                    (
                                                        -probs_fdo
                                                        * torch.log(
                                                            probs_fdo.clamp_min(
                                                                1e-12
                                                            )
                                                        )
                                                    ).sum().item()
                                                )
                                                if len(labels_order) == 2:
                                                    abs_margin_fdo = float(
                                                        abs(
                                                            (
                                                                logps_fdo[0]
                                                                - logps_fdo[1]
                                                            ).item()
                                                        )
                                                    )
                                                else:
                                                    top2 = torch.topk(
                                                        logps_fdo, k=2
                                                    ).values
                                                    abs_margin_fdo = float(
                                                        abs(
                                                            (
                                                                top2[0]
                                                                - top2[1]
                                                            ).item()
                                                        )
                                                    )
                                                y = torch.zeros_like(probs_fdo)
                                                gold_idx = labels_order.index(
                                                    gold_name
                                                )
                                                y[gold_idx] = 1.0
                                                brier_fdo = float(
                                                    ((probs_fdo - y) ** 2)
                                                    .sum()
                                                    .item()
                                                )
                                                stats_n[
                                                    "flip_demos_only"
                                                ][cat_i] += 1.0
                                                stats_abs_margin[
                                                    "flip_demos_only"
                                                ][cat_i] += abs_margin_fdo
                                                stats_entropy[
                                                    "flip_demos_only"
                                                ][cat_i] += ent_fdo
                                                stats_brier[
                                                    "flip_demos_only"
                                                ][cat_i] += brier_fdo

                                                # Length-bucket stats (semantic correctness).
                                                try:
                                                    n_tok = int(flip_demos_only_lens[j])
                                                    bi = _len_bucket(n_tok)
                                                    len_n["flip_demos_only"][bi] += 1.0
                                                    len_correct["flip_demos_only"][bi] += (
                                                        1.0
                                                        if flip_demos_only_preds[j]
                                                        == gold_name
                                                        else 0.0
                                                    )
                                                    len_abs_margin["flip_demos_only"][bi] += abs_margin_fdo
                                                    len_entropy["flip_demos_only"][bi] += ent_fdo
                                                except Exception:
                                                    pass
                                                if prior_margin is not None and len(label_names) == 2:
                                                    logp_ent = float(
                                                        flip_demos_only_scores_t[
                                                            name_ent
                                                        ][j].item()
                                                    )
                                                    logp_not = float(
                                                        flip_demos_only_scores_t[
                                                            name_not
                                                        ][j].item()
                                                    )
                                                    stats_margin_lift[
                                                        "flip_demos_only"
                                                    ][cat_i] += (
                                                        (logp_ent - logp_not)
                                                        - float(prior_margin)
                                                    )
                                                is_correct_sem = (
                                                    1.0
                                                    if flip_demos_only_preds[j]
                                                    == gold_name
                                                    else 0.0
                                                )
                                                bin_i = min(
                                                    int(conf_fdo * ece_bins),
                                                    ece_bins - 1,
                                                )
                                                ece_count[
                                                    "flip_demos_only"
                                                ][cat_i][bin_i] += 1.0
                                                ece_sum_conf[
                                                    "flip_demos_only"
                                                ][cat_i][bin_i] += conf_fdo
                                                ece_sum_acc[
                                                    "flip_demos_only"
                                                ][cat_i][bin_i] += is_correct_sem

                                            if flip_gold_name is not None:
                                                total_flip_demos_only += 1
                                                if (
                                                    flip_demos_only_preds[j]
                                                    == flip_gold_name
                                                ):
                                                    correct_flip_demos_only += 1

                                                base_is_correct = (
                                                    gold_name is not None
                                                    and preds[j] == gold_name
                                                )
                                                fdo_is_correct = (
                                                    flip_demos_only_preds[j]
                                                    == flip_gold_name
                                                )
                                                key = "flip_demos_only"
                                                if key in flip_cond:
                                                    if base_is_correct:
                                                        flip_cond[key][
                                                            "base_correct"
                                                        ][cat_i] += 1.0
                                                        if fdo_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_correct"
                                                            ][cat_i] += 1.0
                                                    else:
                                                        flip_cond[key][
                                                            "base_wrong"
                                                        ][cat_i] += 1.0
                                                        if fdo_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_wrong"
                                                            ][cat_i] += 1.0
                                                    try:
                                                        base_pred_idx = labels_order.index(
                                                            preds[j]
                                                        )
                                                        fdo_pred_idx = labels_order.index(
                                                            flip_demos_only_preds[j]
                                                        )
                                                        # No instruction => no semantic unflip.
                                                        fdo_sem_idx = fdo_pred_idx
                                                        flip_cond[key][
                                                            "invariance_total"
                                                        ][cat_i] += 1.0
                                                        if (
                                                            fdo_sem_idx
                                                            == base_pred_idx
                                                        ):
                                                            flip_cond[key][
                                                                "invariance_agree"
                                                            ][cat_i] += 1.0
                                                    except Exception:
                                                        pass
                                            total_flip_demos_only_dir += 1
                                            if (
                                                flip_demos_only_preds[j]
                                                == expected
                                            ):
                                                correct_flip_demos_only_dir += 1

                                        if (
                                            do_flip_instruction_only
                                            and flip_instruction_only_preds
                                            is not None
                                        ):
                                            if (
                                                flip_instruction_only_scores_t
                                                is not None
                                                and gold_name is not None
                                                and cat_i is not None
                                            ):
                                                logps_fio = torch.stack(
                                                    [
                                                        flip_instruction_only_scores_t[k][j]
                                                        for k in labels_order
                                                    ]
                                                ).float()
                                                probs_fio = torch.softmax(
                                                    logps_fio, dim=0
                                                )
                                                conf_fio = float(
                                                    probs_fio.max().item()
                                                )
                                                ent_fio = float(
                                                    (
                                                        -probs_fio
                                                        * torch.log(
                                                            probs_fio.clamp_min(
                                                                1e-12
                                                            )
                                                        )
                                                    ).sum().item()
                                                )
                                                if len(labels_order) == 2:
                                                    abs_margin_fio = float(
                                                        abs(
                                                            (
                                                                logps_fio[0]
                                                                - logps_fio[1]
                                                            ).item()
                                                        )
                                                    )
                                                else:
                                                    top2 = torch.topk(
                                                        logps_fio, k=2
                                                    ).values
                                                    abs_margin_fio = float(
                                                        abs(
                                                            (
                                                                top2[0]
                                                                - top2[1]
                                                            ).item()
                                                        )
                                                    )
                                                y = torch.zeros_like(probs_fio)
                                                gold_idx = labels_order.index(
                                                    gold_name
                                                )
                                                y[gold_idx] = 1.0
                                                brier_fio = float(
                                                    ((probs_fio - y) ** 2)
                                                    .sum()
                                                    .item()
                                                )
                                                stats_n[
                                                    "flip_instruction_only"
                                                ][cat_i] += 1.0
                                                stats_abs_margin[
                                                    "flip_instruction_only"
                                                ][cat_i] += abs_margin_fio
                                                stats_entropy[
                                                    "flip_instruction_only"
                                                ][cat_i] += ent_fio
                                                stats_brier[
                                                    "flip_instruction_only"
                                                ][cat_i] += brier_fio

                                                # Length-bucket stats (semantic correctness).
                                                try:
                                                    n_tok = int(flip_instruction_only_lens[j])
                                                    bi = _len_bucket(n_tok)
                                                    len_n["flip_instruction_only"][bi] += 1.0
                                                    len_correct["flip_instruction_only"][bi] += (
                                                        1.0
                                                        if flip_instruction_only_preds[j]
                                                        == gold_name
                                                        else 0.0
                                                    )
                                                    len_abs_margin["flip_instruction_only"][bi] += abs_margin_fio
                                                    len_entropy["flip_instruction_only"][bi] += ent_fio
                                                except Exception:
                                                    pass
                                                if prior_margin is not None and len(label_names) == 2:
                                                    logp_ent = float(
                                                        flip_instruction_only_scores_t[
                                                            name_ent
                                                        ][j].item()
                                                    )
                                                    logp_not = float(
                                                        flip_instruction_only_scores_t[
                                                            name_not
                                                        ][j].item()
                                                    )
                                                    stats_margin_lift[
                                                        "flip_instruction_only"
                                                    ][cat_i] += (
                                                        (logp_ent - logp_not)
                                                        - float(prior_margin)
                                                    )
                                                is_correct_sem = (
                                                    1.0
                                                    if flip_instruction_only_preds[j]
                                                    == gold_name
                                                    else 0.0
                                                )
                                                bin_i = min(
                                                    int(conf_fio * ece_bins),
                                                    ece_bins - 1,
                                                )
                                                ece_count[
                                                    "flip_instruction_only"
                                                ][cat_i][bin_i] += 1.0
                                                ece_sum_conf[
                                                    "flip_instruction_only"
                                                ][cat_i][bin_i] += conf_fio
                                                ece_sum_acc[
                                                    "flip_instruction_only"
                                                ][cat_i][bin_i] += is_correct_sem

                                            if flip_gold_name is not None:
                                                total_flip_instruction_only += 1
                                                if (
                                                    flip_instruction_only_preds[
                                                        j
                                                    ]
                                                    == flip_gold_name
                                                ):
                                                    correct_flip_instruction_only += 1

                                                base_is_correct = (
                                                    gold_name is not None
                                                    and preds[j] == gold_name
                                                )
                                                fio_is_correct = (
                                                    flip_instruction_only_preds[j]
                                                    == flip_gold_name
                                                )
                                                key = "flip_instruction_only"
                                                if key in flip_cond:
                                                    if base_is_correct:
                                                        flip_cond[key][
                                                            "base_correct"
                                                        ][cat_i] += 1.0
                                                        if fio_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_correct"
                                                            ][cat_i] += 1.0
                                                    else:
                                                        flip_cond[key][
                                                            "base_wrong"
                                                        ][cat_i] += 1.0
                                                        if fio_is_correct:
                                                            flip_cond[key][
                                                                "flip_correct_and_base_wrong"
                                                            ][cat_i] += 1.0
                                                    try:
                                                        base_pred_idx = labels_order.index(
                                                            preds[j]
                                                        )
                                                        fio_pred_idx = labels_order.index(
                                                            flip_instruction_only_preds[j]
                                                        )
                                                        fio_sem_idx = (
                                                            1 - fio_pred_idx
                                                            if len(labels_order)
                                                            == 2
                                                            else fio_pred_idx
                                                        )
                                                        flip_cond[key][
                                                            "invariance_total"
                                                        ][cat_i] += 1.0
                                                        if (
                                                            fio_sem_idx
                                                            == base_pred_idx
                                                        ):
                                                            flip_cond[key][
                                                                "invariance_agree"
                                                            ][cat_i] += 1.0
                                                    except Exception:
                                                        pass
                                            total_flip_instruction_only_dir += 1
                                            if (
                                                flip_instruction_only_preds[j]
                                                == expected
                                            ):
                                                correct_flip_instruction_only_dir += 1

                                if args.write_examples and f_out is not None:
                                    # Materialize per-example records for this batch
                                    for j, ex in enumerate(batch_ex):
                                        gold = ex.get("label")
                                        rec = {
                                            "model": model_id,
                                            "mode": mode,
                                            "dataset": args.dataset,
                                            "shots": shots,
                                            "seed": seed,
                                            "template_id": pr.template_id,
                                            "category": cat,
                                            "example_index": int(
                                                batch_indices[j]
                                            ),
                                            "premise": ex[premise_key],
                                            "hypothesis": ex[hypothesis_key],
                                            "gold": None
                                            if gold is None
                                            else label_names[int(gold)],
                                            "pred": preds[j],
                                            "label_words": row_lw,
                                            "rank": rank,
                                        }
                                        rec["scores"] = {
                                            k: float(scores_t[k][j].item())
                                            for k in labels_order
                                        }
                                        if (
                                            "flip" in experiments
                                            and flip_preds is not None
                                        ):
                                            rec["flip_pred"] = flip_preds[j]
                                            if flip_scores_t is not None:
                                                rec["scores_flip"] = {
                                                    k: float(
                                                        flip_scores_t[k][j].item()
                                                    )
                                                    for k in labels_order
                                                }
                                        if (
                                            "flip_conflict" in experiments
                                            and flip_conflict_preds is not None
                                        ):
                                            rec["flip_conflict_pred"] = (
                                                flip_conflict_preds[j]
                                            )
                                            if (
                                                flip_conflict_scores_t
                                                is not None
                                            ):
                                                rec[
                                                    "scores_flip_conflict"
                                                ] = {
                                                    k: float(
                                                        flip_conflict_scores_t[k][j].item()
                                                    )
                                                    for k in labels_order
                                                }
                                        if (
                                            "flip_demos_only" in experiments
                                            and flip_demos_only_preds
                                            is not None
                                        ):
                                            rec["flip_demos_only_pred"] = (
                                                flip_demos_only_preds[j]
                                            )
                                            if (
                                                flip_demos_only_scores_t
                                                is not None
                                            ):
                                                rec[
                                                    "scores_flip_demos_only"
                                                ] = {
                                                    k: float(
                                                        flip_demos_only_scores_t[k][j].item()
                                                    )
                                                    for k in labels_order
                                                }
                                        if (
                                            "flip_instruction_only"
                                            in experiments
                                            and flip_instruction_only_preds
                                            is not None
                                        ):
                                            rec[
                                                "flip_instruction_only_pred"
                                            ] = flip_instruction_only_preds[j]
                                            if (
                                                flip_instruction_only_scores_t
                                                is not None
                                            ):
                                                rec[
                                                    "scores_flip_instruction_only"
                                                ] = {
                                                    k: float(
                                                        flip_instruction_only_scores_t[k][j].item()
                                                    )
                                                    for k in labels_order
                                                }
                                        f_out.write(
                                            json.dumps(rec, ensure_ascii=False)
                                            + "\n"
                                        )

                            # Reduce counts across ranks
                            counts = torch.tensor(
                                [
                                    correct,
                                    total,
                                    correct_flip,
                                    total_flip,
                                    correct_flip_dir,
                                    total_flip_dir,
                                    correct_flip_conflict,
                                    total_flip_conflict,
                                    correct_flip_conflict_dir,
                                    total_flip_conflict_dir,
                                    correct_flip_demos_only,
                                    total_flip_demos_only,
                                    correct_flip_demos_only_dir,
                                    total_flip_demos_only_dir,
                                    correct_flip_instruction_only,
                                    total_flip_instruction_only,
                                    correct_flip_instruction_only_dir,
                                    total_flip_instruction_only_dir,
                                ],
                                device=model_device,
                                dtype=torch.long,
                            )
                            if accelerator is not None:
                                counts = accelerator.reduce(
                                    counts, reduction="sum"
                                )
                            elif use_distributed and dist.is_initialized():
                                dist.all_reduce(counts, op=dist.ReduceOp.SUM)

                            correct = int(counts[0].item())
                            total = int(counts[1].item())
                            correct_flip = int(counts[2].item())
                            total_flip = int(counts[3].item())
                            correct_flip_dir = int(counts[4].item())
                            total_flip_dir = int(counts[5].item())
                            correct_flip_conflict = int(counts[6].item())
                            total_flip_conflict = int(counts[7].item())
                            correct_flip_conflict_dir = int(counts[8].item())
                            total_flip_conflict_dir = int(counts[9].item())
                            correct_flip_demos_only = int(counts[10].item())
                            total_flip_demos_only = int(counts[11].item())
                            correct_flip_demos_only_dir = int(
                                counts[12].item()
                            )
                            total_flip_demos_only_dir = int(counts[13].item())
                            correct_flip_instruction_only = int(
                                counts[14].item()
                            )
                            total_flip_instruction_only = int(
                                counts[15].item()
                            )
                            correct_flip_instruction_only_dir = int(
                                counts[16].item()
                            )
                            total_flip_instruction_only_dir = int(
                                counts[17].item()
                            )

                            # Record per-template accuracies (after reduction) for variance/correlation.
                            if is_main:
                                if total:
                                    template_acc.setdefault("base", {}).setdefault(
                                        cat, {}
                                    )[str(pr.template_id)] = float(correct / total)
                                if "flip" in experiments and total_flip:
                                    template_acc.setdefault("flip", {}).setdefault(
                                        cat, {}
                                    )[str(pr.template_id)] = float(
                                        correct_flip / total_flip
                                    )
                                if "flip_conflict" in experiments and total_flip_conflict:
                                    template_acc.setdefault(
                                        "flip_conflict", {}
                                    ).setdefault(cat, {})[str(pr.template_id)] = float(
                                        correct_flip_conflict
                                        / total_flip_conflict
                                    )
                                if "flip_demos_only" in experiments and total_flip_demos_only:
                                    template_acc.setdefault(
                                        "flip_demos_only", {}
                                    ).setdefault(cat, {})[str(pr.template_id)] = float(
                                        correct_flip_demos_only
                                        / total_flip_demos_only
                                    )
                                if "flip_instruction_only" in experiments and total_flip_instruction_only:
                                    template_acc.setdefault(
                                        "flip_instruction_only", {}
                                    ).setdefault(cat, {})[str(pr.template_id)] = float(
                                        correct_flip_instruction_only
                                        / total_flip_instruction_only
                                    )

                            acc_by_cat.setdefault(cat, [0, 0])
                            acc_by_cat[cat][0] += correct
                            acc_by_cat[cat][1] += total

                            if "flip" in experiments:
                                flip_acc_by_cat.setdefault(cat, [0, 0])
                                flip_acc_by_cat[cat][0] += correct_flip
                                flip_acc_by_cat[cat][1] += total_flip

                                flip_dir_by_cat.setdefault(cat, [0, 0])
                                flip_dir_by_cat[cat][0] += correct_flip_dir
                                flip_dir_by_cat[cat][1] += total_flip_dir

                            if "flip_conflict" in experiments:
                                flip_conflict_acc_by_cat.setdefault(
                                    cat, [0, 0]
                                )
                                flip_conflict_acc_by_cat[cat][0] += (
                                    correct_flip_conflict
                                )
                                flip_conflict_acc_by_cat[cat][1] += (
                                    total_flip_conflict
                                )

                                flip_conflict_dir_by_cat.setdefault(
                                    cat, [0, 0]
                                )
                                flip_conflict_dir_by_cat[cat][0] += (
                                    correct_flip_conflict_dir
                                )
                                flip_conflict_dir_by_cat[cat][1] += (
                                    total_flip_conflict_dir
                                )

                            if "flip_demos_only" in experiments:
                                flip_demos_only_acc_by_cat.setdefault(
                                    cat, [0, 0]
                                )
                                flip_demos_only_acc_by_cat[cat][0] += (
                                    correct_flip_demos_only
                                )
                                flip_demos_only_acc_by_cat[cat][1] += (
                                    total_flip_demos_only
                                )

                                flip_demos_only_dir_by_cat.setdefault(
                                    cat, [0, 0]
                                )
                                flip_demos_only_dir_by_cat[cat][0] += (
                                    correct_flip_demos_only_dir
                                )
                                flip_demos_only_dir_by_cat[cat][1] += (
                                    total_flip_demos_only_dir
                                )

                            if "flip_instruction_only" in experiments:
                                flip_instruction_only_acc_by_cat.setdefault(
                                    cat, [0, 0]
                                )
                                flip_instruction_only_acc_by_cat[cat][0] += (
                                    correct_flip_instruction_only
                                )
                                flip_instruction_only_acc_by_cat[cat][1] += (
                                    total_flip_instruction_only
                                )

                                flip_instruction_only_dir_by_cat.setdefault(
                                    cat, [0, 0]
                                )
                                flip_instruction_only_dir_by_cat[cat][0] += (
                                    correct_flip_instruction_only_dir
                                )
                                flip_instruction_only_dir_by_cat[cat][1] += (
                                    total_flip_instruction_only_dir
                                )

                        # Task-switch override (separate prompt, binary yes/no only)
                        if "task_switch" in experiments:
                            ts_candidate_token_seqs = {
                                "yes": tokenizer(
                                    " yes", add_special_tokens=False
                                ).input_ids,
                                "no": tokenizer(
                                    " no", add_special_tokens=False
                                ).input_ids,
                            }
                            ts_labels = ["yes", "no"]
                            ts_single_tok = _all_candidates_single_token(
                                ts_candidate_token_seqs
                            )
                            if ts_single_tok:
                                ts_candidate_token_ids = {
                                    k: int(v[0])
                                    for k, v in ts_candidate_token_seqs.items()
                                }

                            bs = max(1, int(args.batch_size))
                            steps = range(0, len(eval_indices), bs)
                            for bidx, ofs in enumerate(
                                tqdm(
                                    steps,
                                    total=(len(eval_indices) + bs - 1) // bs,
                                    desc=f"task_switch ({run_slug})",
                                    leave=False,
                                    disable=(not is_main),
                                )
                            ):
                                batch_indices = eval_indices[ofs : ofs + bs]
                                batch_ex = [val_ds[i] for i in batch_indices]

                                raw_ts_list: List[str] = []
                                for ex in batch_ex:
                                    base = (
                                        f"Premise: {ex[premise_key]}\n"
                                        f"Hypothesis: {ex[hypothesis_key]}\n"
                                        "Question: Is the hypothesis grammatical English?\n"
                                    )
                                    base = ensure_answer_anchor(base)
                                    raw_ts_list.append(
                                        add_task_switch_instruction(base)
                                    )

                                ts_texts = [
                                    format_prompt(
                                        tokenizer,
                                        rt,
                                        as_chat,
                                        args.system_prompt,
                                    )
                                    for rt in raw_ts_list
                                ]

                                enc_ts = tokenizer(
                                    ts_texts,
                                    return_tensors="pt",
                                    padding=True,
                                    add_special_tokens=False,
                                    pad_to_multiple_of=8,
                                )
                                enc_ts = {
                                    k: v.to(model_device)
                                    for k, v in enc_ts.items()
                                }

                                autocast_enabled = model_device.type == "cuda"
                                with torch.autocast(
                                    device_type="cuda",
                                    dtype=amp_dtype,
                                    enabled=autocast_enabled,
                                ):
                                    if ts_single_tok:
                                        ts_scores_t = (
                                            score_candidates_lastlogits_batch(
                                                model,
                                                enc_ts["input_ids"],
                                                enc_ts["attention_mask"],
                                                ts_candidate_token_ids,
                                                tokenizer.padding_side,
                                            )
                                        )
                                    else:
                                        ts_scores_t = score_candidates_fullseq_batch(
                                            model,
                                            enc_ts["input_ids"],
                                            enc_ts["attention_mask"],
                                            {
                                                k: list(v)
                                                for k, v in ts_candidate_token_seqs.items()
                                            },
                                            tokenizer.padding_side,
                                            int(tokenizer.pad_token_id),
                                        )

                                ts_preds = _pick_preds_from_scores(
                                    ts_scores_t, ts_labels
                                )
                                for j in range(len(batch_ex)):
                                    gold_ts = (
                                        task_switch_gold[ofs + j]
                                        if task_switch_gold is not None
                                        else None
                                    )
                                    if gold_ts is not None:
                                        task_switch_total += 1
                                        if ts_preds[j] == gold_ts:
                                            task_switch_correct += 1

                            ts_counts = torch.tensor(
                                [task_switch_correct, task_switch_total],
                                device=model_device,
                                dtype=torch.long,
                            )
                            if accelerator is not None:
                                ts_counts = accelerator.reduce(
                                    ts_counts, reduction="sum"
                                )
                            elif use_distributed and dist.is_initialized():
                                dist.all_reduce(
                                    ts_counts, op=dist.ReduceOp.SUM
                                )

                            task_switch_correct = int(ts_counts[0].item())
                            task_switch_total = int(ts_counts[1].item())

                    finally:
                        if f_out_ctx is not None:
                            f_out_ctx.close()

                    # Reduce aggregated diagnostic arrays across ranks.
                    # Everything is summed (not averaged) to keep determinism.
                    def _pack_diag() -> List[float]:
                        flat: List[float] = []
                        for c in cond_names:
                            flat.extend(stats_n[c])
                            flat.extend(stats_abs_margin[c])
                            flat.extend(stats_entropy[c])
                            flat.extend(stats_brier[c])
                            flat.extend(stats_margin_lift[c])
                            for ci in range(num_cats):
                                flat.extend(ece_count[c][ci])
                                flat.extend(ece_sum_conf[c][ci])
                                flat.extend(ece_sum_acc[c][ci])

                        for c in cond_names:
                            flat.extend(len_n[c])
                            flat.extend(len_correct[c])
                            flat.extend(len_abs_margin[c])
                            flat.extend(len_entropy[c])

                        flat.extend(base_tp)
                        flat.extend(base_tn)
                        flat.extend(base_fp)
                        flat.extend(base_fn)

                        for c in cond_names:
                            if c == "base":
                                continue
                            flat.extend(flip_cond[c]["base_correct"])
                            flat.extend(flip_cond[c]["base_wrong"])
                            flat.extend(
                                flip_cond[c][
                                    "flip_correct_and_base_correct"
                                ]
                            )
                            flat.extend(
                                flip_cond[c]["flip_correct_and_base_wrong"]
                            )
                            flat.extend(flip_cond[c]["invariance_agree"])
                            flat.extend(flip_cond[c]["invariance_total"])

                        flat.extend(conflict_pref["demo_consistent"])
                        flat.extend(conflict_pref["instruction_consistent"])
                        flat.extend(conflict_pref["total"])

                        flat.extend(prior_margin_sum)
                        flat.extend(prior_margin_n)
                        return flat

                    def _unpack_diag(flat: List[float]) -> None:
                        idx = 0
                        for c in cond_names:
                            stats_n[c] = flat[idx : idx + num_cats]
                            idx += num_cats
                            stats_abs_margin[c] = flat[idx : idx + num_cats]
                            idx += num_cats
                            stats_entropy[c] = flat[idx : idx + num_cats]
                            idx += num_cats
                            stats_brier[c] = flat[idx : idx + num_cats]
                            idx += num_cats
                            stats_margin_lift[c] = flat[idx : idx + num_cats]
                            idx += num_cats
                            for ci in range(num_cats):
                                ece_count[c][ci] = flat[idx : idx + ece_bins]
                                idx += ece_bins
                                ece_sum_conf[c][ci] = flat[
                                    idx : idx + ece_bins
                                ]
                                idx += ece_bins
                                ece_sum_acc[c][ci] = flat[
                                    idx : idx + ece_bins
                                ]
                                idx += ece_bins

                        for c in cond_names:
                            len_n[c] = flat[idx : idx + num_len_buckets]
                            idx += num_len_buckets
                            len_correct[c] = flat[idx : idx + num_len_buckets]
                            idx += num_len_buckets
                            len_abs_margin[c] = flat[idx : idx + num_len_buckets]
                            idx += num_len_buckets
                            len_entropy[c] = flat[idx : idx + num_len_buckets]
                            idx += num_len_buckets

                        nonlocal_base_tp = flat[idx : idx + num_cats]
                        idx += num_cats
                        nonlocal_base_tn = flat[idx : idx + num_cats]
                        idx += num_cats
                        nonlocal_base_fp = flat[idx : idx + num_cats]
                        idx += num_cats
                        nonlocal_base_fn = flat[idx : idx + num_cats]
                        idx += num_cats
                        for ci in range(num_cats):
                            base_tp[ci] = nonlocal_base_tp[ci]
                            base_tn[ci] = nonlocal_base_tn[ci]
                            base_fp[ci] = nonlocal_base_fp[ci]
                            base_fn[ci] = nonlocal_base_fn[ci]

                        for c in cond_names:
                            if c == "base":
                                continue
                            flip_cond[c]["base_correct"] = flat[
                                idx : idx + num_cats
                            ]
                            idx += num_cats
                            flip_cond[c]["base_wrong"] = flat[
                                idx : idx + num_cats
                            ]
                            idx += num_cats
                            flip_cond[c][
                                "flip_correct_and_base_correct"
                            ] = flat[idx : idx + num_cats]
                            idx += num_cats
                            flip_cond[c][
                                "flip_correct_and_base_wrong"
                            ] = flat[idx : idx + num_cats]
                            idx += num_cats
                            flip_cond[c]["invariance_agree"] = flat[
                                idx : idx + num_cats
                            ]
                            idx += num_cats
                            flip_cond[c]["invariance_total"] = flat[
                                idx : idx + num_cats
                            ]
                            idx += num_cats

                        conflict_pref["demo_consistent"] = flat[
                            idx : idx + num_cats
                        ]
                        idx += num_cats
                        conflict_pref["instruction_consistent"] = flat[
                            idx : idx + num_cats
                        ]
                        idx += num_cats
                        conflict_pref["total"] = flat[idx : idx + num_cats]
                        idx += num_cats

                        prior_margin_sum[:] = flat[idx : idx + num_cats]
                        idx += num_cats
                        prior_margin_n[:] = flat[idx : idx + num_cats]
                        idx += num_cats

                    diag_flat = _pack_diag()
                    if world_size > 1:
                        diag_t = torch.tensor(
                            diag_flat,
                            device=model_device,
                            dtype=torch.float32,
                        )
                        if accelerator is not None:
                            diag_t = accelerator.reduce(
                                diag_t, reduction="sum"
                            )
                        elif use_distributed and dist.is_initialized():
                            dist.all_reduce(diag_t, op=dist.ReduceOp.SUM)
                        _unpack_diag(diag_t.detach().cpu().tolist())

                    def _acc(pair: Optional[List[int]]) -> Optional[float]:
                        if not pair:
                            return None
                        c, t = pair
                        return (c / t) if t else None

                    summary = {
                        "model": model_id,
                        "mode": mode,
                        "dataset": args.dataset,
                        "shots": shots,
                        "seed": seed,
                        "n_eval": n_eval,
                        "prompt_csv": args.prompt_csv,
                        "experiment_name": args.experiment_name,
                        "templates_per_category": args.templates_per_category,
                        "world_size": world_size,
                        "all_single_token_label_words": bool(
                            all_single_token_label_words
                        ),
                        "label_words_by_label": {
                            ln: sorted(label_words_seen.get(ln, set()))
                            for ln in label_names
                        },
                        "label_word_token_lens_by_label": {
                            ln: sorted(
                                int(x)
                                for x in label_word_token_lens_seen.get(
                                    ln, set()
                                )
                            )
                            for ln in label_names
                        },
                        "acc_by_category": {
                            k: _acc(v) for k, v in sorted(acc_by_cat.items())
                        },
                        "acc_flip_by_category": {
                            k: _acc(v)
                            for k, v in sorted(flip_acc_by_cat.items())
                        },
                        "flip_directional_compliance_by_category": {
                            k: _acc(v)
                            for k, v in sorted(flip_dir_by_cat.items())
                        },
                        "acc_flip_conflict_by_category": {
                            k: _acc(v)
                            for k, v in sorted(
                                flip_conflict_acc_by_cat.items()
                            )
                        },
                        "flip_conflict_directional_compliance_by_category": {
                            k: _acc(v)
                            for k, v in sorted(
                                flip_conflict_dir_by_cat.items()
                            )
                        },
                        "acc_flip_demos_only_by_category": {
                            k: _acc(v)
                            for k, v in sorted(
                                flip_demos_only_acc_by_cat.items()
                            )
                        },
                        "flip_demos_only_directional_compliance_by_category": {
                            k: _acc(v)
                            for k, v in sorted(
                                flip_demos_only_dir_by_cat.items()
                            )
                        },
                        "acc_flip_instruction_only_by_category": {
                            k: _acc(v)
                            for k, v in sorted(
                                flip_instruction_only_acc_by_cat.items()
                            )
                        },
                        "flip_instruction_only_directional_compliance_by_category": {
                            k: _acc(v)
                            for k, v in sorted(
                                flip_instruction_only_dir_by_cat.items()
                            )
                        },
                        "task_switch_acc": (
                            task_switch_correct / task_switch_total
                        )
                        if task_switch_total
                        else None,
                    }

                    # Attach diagnostics.
                    def _mean(sum_v: float, n_v: float) -> Optional[float]:
                        return (sum_v / n_v) if n_v else None

                    mean_abs_margin_by_cat = {
                        c: {
                            cats[i]: _mean(
                                stats_abs_margin[c][i], stats_n[c][i]
                            )
                            for i in range(num_cats)
                        }
                        for c in cond_names
                    }
                    mean_entropy_by_cat = {
                        c: {
                            cats[i]: _mean(
                                stats_entropy[c][i], stats_n[c][i]
                            )
                            for i in range(num_cats)
                        }
                        for c in cond_names
                    }
                    mean_brier_by_cat = {
                        c: {
                            cats[i]: _mean(
                                stats_brier[c][i], stats_n[c][i]
                            )
                            for i in range(num_cats)
                        }
                        for c in cond_names
                    }
                    mean_margin_lift_by_cat = {
                        c: {
                            cats[i]: _mean(
                                stats_margin_lift[c][i], stats_n[c][i]
                            )
                            for i in range(num_cats)
                        }
                        for c in cond_names
                    }

                    ece_by_cat = {}
                    for c in cond_names:
                        ece_by_cat[c] = {}
                        for i in range(num_cats):
                            n_v = stats_n[c][i]
                            if not n_v:
                                ece_by_cat[c][cats[i]] = None
                                continue
                            ece = 0.0
                            for b in range(ece_bins):
                                cnt = ece_count[c][i][b]
                                if not cnt:
                                    continue
                                conf_b = ece_sum_conf[c][i][b] / cnt
                                acc_b = ece_sum_acc[c][i][b] / cnt
                                ece += (cnt / n_v) * abs(acc_b - conf_b)
                            ece_by_cat[c][cats[i]] = float(ece)

                    prior_margin_by_cat = {
                        cats[i]: _mean(prior_margin_sum[i], prior_margin_n[i])
                        for i in range(num_cats)
                    }

                    base_confusion_by_cat = None
                    if len(label_names) == 2 and idx_ent is not None:
                        base_confusion_by_cat = {}
                        for i in range(num_cats):
                            tp = base_tp[i]
                            tn = base_tn[i]
                            fp = base_fp[i]
                            fn = base_fn[i]
                            tpr = tp / (tp + fn) if (tp + fn) else None
                            tnr = tn / (tn + fp) if (tn + fp) else None
                            fpr = fp / (fp + tn) if (fp + tn) else None
                            fnr = fn / (fn + tp) if (fn + tp) else None
                            base_confusion_by_cat[cats[i]] = {
                                "tp": tp,
                                "tn": tn,
                                "fp": fp,
                                "fn": fn,
                                "tpr": tpr,
                                "tnr": tnr,
                                "fpr": fpr,
                                "fnr": fnr,
                            }

                    flip_conditional_by_cat = {}
                    for c in cond_names:
                        if c == "base":
                            continue
                        flip_conditional_by_cat[c] = {}
                        for i in range(num_cats):
                            bc = flip_cond[c]["base_correct"][i]
                            bw = flip_cond[c]["base_wrong"][i]
                            fcbc = flip_cond[c][
                                "flip_correct_and_base_correct"
                            ][i]
                            fcbw = flip_cond[c][
                                "flip_correct_and_base_wrong"
                            ][i]
                            inv_a = flip_cond[c]["invariance_agree"][i]
                            inv_t = flip_cond[c]["invariance_total"][i]
                            flip_conditional_by_cat[c][cats[i]] = {
                                "flip_acc_given_base_correct": (
                                    (fcbc / bc) if bc else None
                                ),
                                "flip_acc_given_base_wrong": (
                                    (fcbw / bw) if bw else None
                                ),
                                "semantic_invariance": (
                                    (inv_a / inv_t) if inv_t else None
                                ),
                            }

                    conflict_preference_by_cat = None
                    if "flip_conflict" in experiments:
                        conflict_preference_by_cat = {}
                        for i in range(num_cats):
                            tot = conflict_pref["total"][i]
                            conflict_preference_by_cat[cats[i]] = {
                                "demo_consistent": (
                                    conflict_pref["demo_consistent"][i]
                                    / tot
                                    if tot
                                    else None
                                ),
                                "instruction_consistent": (
                                    conflict_pref[
                                        "instruction_consistent"
                                    ][i]
                                    / tot
                                    if tot
                                    else None
                                ),
                            }

                    summary["diagnostics"] = {
                        "mean_abs_margin_by_category": mean_abs_margin_by_cat,
                        "mean_entropy_by_category": mean_entropy_by_cat,
                        "mean_brier_by_category": mean_brier_by_cat,
                        "ece_by_category": ece_by_cat,
                        "mean_margin_lift_by_category": mean_margin_lift_by_cat,
                        "prior_margin_by_category": prior_margin_by_cat,
                        "base_confusion_by_category": base_confusion_by_cat,
                        "flip_conditionals_by_category": flip_conditional_by_cat,
                        "flip_conflict_preference_by_category": conflict_preference_by_cat,
                        "length_buckets": {
                            "bucket_names": list(length_bucket_names),
                            "by_condition": {
                                c: {
                                    "n": list(len_n[c]),
                                    "acc": [
                                        (len_correct[c][i] / len_n[c][i])
                                        if len_n[c][i]
                                        else None
                                        for i in range(num_len_buckets)
                                    ],
                                    "mean_abs_margin": [
                                        (len_abs_margin[c][i] / len_n[c][i])
                                        if len_n[c][i]
                                        else None
                                        for i in range(num_len_buckets)
                                    ],
                                    "mean_entropy": [
                                        (len_entropy[c][i] / len_n[c][i])
                                        if len_n[c][i]
                                        else None
                                        for i in range(num_len_buckets)
                                    ],
                                }
                                for c in cond_names
                            },
                        },
                    }

                    # Template variance + Spearman correlations.
                    if is_main:
                        def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
                            if not vals:
                                return None, None
                            m = sum(vals) / float(len(vals))
                            if len(vals) <= 1:
                                return float(m), 0.0
                            v = sum((x - m) ** 2 for x in vals) / float(len(vals))
                            return float(m), float(v ** 0.5)

                        template_stats = {
                            "per_condition": {},
                            "spearman_vs_base": {},
                        }

                        # Per condition/category variance.
                        for c in cond_names:
                            template_stats["per_condition"][c] = {}
                            for cat_name in cats:
                                d = template_acc.get(c, {}).get(cat_name, {})
                                vals = list(d.values())
                                mean_v, std_v = _mean_std(vals)
                                template_stats["per_condition"][c][cat_name] = {
                                    "n_templates": int(len(vals)),
                                    "mean": mean_v,
                                    "std": std_v,
                                    "min": (min(vals) if vals else None),
                                    "max": (max(vals) if vals else None),
                                }

                        # Spearman vs base across templates (paired by template_id).
                        base_by_cat = template_acc.get("base", {})
                        for c in cond_names:
                            if c == "base":
                                continue
                            template_stats["spearman_vs_base"][c] = {
                                "by_category": {},
                                "overall": None,
                            }
                            overall_x: List[float] = []
                            overall_y: List[float] = []
                            for cat_name in cats:
                                base_d = base_by_cat.get(cat_name, {})
                                other_d = template_acc.get(c, {}).get(cat_name, {})
                                common = [
                                    tid
                                    for tid in base_d.keys()
                                    if tid in other_d
                                ]
                                x = [base_d[tid] for tid in common]
                                y = [other_d[tid] for tid in common]
                                rho = _spearmanr(x, y) if len(common) >= 2 else None
                                template_stats["spearman_vs_base"][c][
                                    "by_category"
                                ][cat_name] = {
                                    "n_templates": int(len(common)),
                                    "rho": rho,
                                }
                                overall_x.extend(x)
                                overall_y.extend(y)
                            template_stats["spearman_vs_base"][c][
                                "overall"
                            ] = (
                                _spearmanr(overall_x, overall_y)
                                if len(overall_x) >= 2
                                else None
                            )

                        summary["diagnostics"]["template_stats"] = template_stats
                    if len(label_names) == 2 and idx_ent is not None:
                        name_ent = label_names[idx_ent]
                        name_not = label_names[1 - idx_ent]
                        summary["label_words_entailment"] = sorted(
                            label_words_seen.get(name_ent, set())
                        )
                        summary["label_words_not_entailment"] = sorted(
                            label_words_seen.get(name_not, set())
                        )
                    if is_main:
                        summary_rows.append(summary)

                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                    elif use_distributed and dist.is_initialized():
                        dist.barrier()

                    if args.write_examples and use_distributed and is_main:
                        manifest_path = os.path.join(
                            args.out_dir, f"{run_slug}.shards.json"
                        )
                        shards = [
                            os.path.join(args.out_dir, f"{run_slug}.jsonl")
                        ] + [
                            os.path.join(
                                args.out_dir, f"{run_slug}.rank{r}.jsonl"
                            )
                            for r in range(1, int(world_size))
                        ]
                        with open(manifest_path, "w", encoding="utf-8") as mf:
                            mf.write(
                                json.dumps(
                                    {
                                        "run_slug": run_slug,
                                        "world_size": int(world_size),
                                        "shards": shards,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                    if not is_main and (not args.write_examples):
                        # No per-rank output was written.
                        pass

    if is_main:
        summary_path = os.path.join(args.out_dir, "summary.jsonl")
        with open(summary_path, "w", encoding="utf-8") as f:
            for r in summary_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Seed stability summary (aggregated over seeds for the same config).
        stability_path = os.path.join(args.out_dir, "seed_stability.jsonl")

        def _mean_std_ignore_none(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
            xs = [v for v in vals if v is not None]
            if not xs:
                return None, None
            m = sum(xs) / float(len(xs))
            if len(xs) <= 1:
                return float(m), 0.0
            v = sum((x - m) ** 2 for x in xs) / float(len(xs))
            return float(m), float(v ** 0.5)

        def _agg_metric(rows: List[Dict[str, object]], key: str) -> Dict[str, object]:
            # key points to a dict-of-category -> acc
            cats_all = set()
            for r in rows:
                d = r.get(key)
                if isinstance(d, dict):
                    cats_all.update(d.keys())
            out = {}
            for c in sorted(cats_all):
                vals = []
                for r in rows:
                    d = r.get(key)
                    vals.append(d.get(c) if isinstance(d, dict) else None)
                mean_v, std_v = _mean_std_ignore_none(vals)
                out[c] = {"mean": mean_v, "std": std_v, "n": len([v for v in vals if v is not None])}
            return out

        groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
        for r in summary_rows:
            gk = (
                r.get("model"),
                r.get("mode"),
                r.get("dataset"),
                r.get("shots"),
                r.get("n_eval"),
                r.get("prompt_csv"),
                r.get("experiment_name"),
                r.get("templates_per_category"),
                r.get("world_size"),
            )
            groups.setdefault(gk, []).append(r)

        with open(stability_path, "w", encoding="utf-8") as f:
            for gk, rows in groups.items():
                seeds = [rr.get("seed") for rr in rows]
                out = {
                    "model": gk[0],
                    "mode": gk[1],
                    "dataset": gk[2],
                    "shots": gk[3],
                    "n_eval": gk[4],
                    "prompt_csv": gk[5],
                    "experiment_name": gk[6],
                    "templates_per_category": gk[7],
                    "world_size": gk[8],
                    "seeds": seeds,
                    "n_seeds": len(rows),
                    "acc_by_category": _agg_metric(rows, "acc_by_category"),
                    "acc_flip_by_category": _agg_metric(rows, "acc_flip_by_category"),
                    "acc_flip_conflict_by_category": _agg_metric(rows, "acc_flip_conflict_by_category"),
                    "acc_flip_demos_only_by_category": _agg_metric(rows, "acc_flip_demos_only_by_category"),
                    "acc_flip_instruction_only_by_category": _agg_metric(rows, "acc_flip_instruction_only_by_category"),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        if args.write_examples:
            print(
                f"Wrote {summary_path} and per-run example JSONL files under {args.out_dir}"
            )
        else:
            print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
