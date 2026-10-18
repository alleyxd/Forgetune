"""Check a dataset before spending GPU time on it.

Finds rows that will fail or train on nothing (missing fields, empty targets, a chosen answer equal to the
rejected one), duplicate and contradictory prompts, rows that will be cut off at --max-len, and test prompts
that also appear in the training data, which quietly inflates every score measured on that test set.
"""
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import data


def fields(row: dict) -> tuple[str, str, str | None, str | None]:
    """(kind, prompt, target, rejected) for one row; kind is 'sft', 'dpo' or 'bad'."""
    if row.get("messages"):
        msgs = row["messages"]
        user = "\n\n".join(m.get("content") or "" for m in msgs[:-1] if m.get("role") == "user")
        target = msgs[-1].get("content") if msgs[-1].get("role") == "assistant" else None
        return "sft", user, target, None
    if row.get("chosen") is not None or row.get("rejected") is not None:
        return "dpo", row.get("prompt") or "", row.get("chosen"), row.get("rejected")
    if row.get("prompt") is not None or row.get("completion") is not None:
        return "sft", row.get("prompt") or "", row.get("completion"), None
    return "bad", "", None, None


def load(source: str, limit: int | None = None) -> list[dict]:
    # JSONL is read row by row rather than through a Dataset, which would give every row the first row's columns
    if Path(source).exists():
        lines = [l for l in Path(source).read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines[:limit]]
    else:
        rows = list(data.read(source, limit=limit))
    return rows


def key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check(source: str, against: str | None = None, model: str | None = None, max_len: int = 1024,
          limit: int | None = None) -> dict:
    rows = load(source, limit)
    problems = defaultdict(list)
    kinds = Counter()
    by_prompt = defaultdict(set)
    lengths = []
    for i, row in enumerate(rows):
        kind, prompt, target, rejected = fields(row)
        kinds[kind] += 1
        if kind == "bad":
            problems["unrecognised format"].append(i)
            continue
        if not prompt.strip():
            problems["empty prompt"].append(i)
        if not (target or "").strip():
            problems["empty or missing target"].append(i)
        if kind == "dpo" and target is not None and key(target) == key(rejected or ""):
            problems["chosen equals rejected"].append(i)
        by_prompt[key(prompt)].add(key(target or ""))
        lengths.append((i, prompt + (target or "")))

    dup_rows = len(rows) - kinds["bad"] - len(by_prompt)
    contradictory = sum(1 for targets in by_prompt.values() if len(targets) > 1)
    report = {"source": source, "rows": len(rows), "kinds": dict(kinds),
              "problems": {k: {"count": len(v), "rows": v[:10]} for k, v in problems.items()},
              "duplicate_prompts": dup_rows, "prompts_with_different_targets": contradictory}

    if lengths:
        if model:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model)
            sizes = [(i, len(tok(text, add_special_tokens=False).input_ids)) for i, text in lengths]
            unit = "tokens"
        else:
            sizes = [(i, len(text)) for i, text in lengths]
            unit = "chars"
        values = [n for _, n in sizes]
        report["length"] = {"unit": unit, "min": min(values), "median": int(statistics.median(values)),
                            "max": max(values)}
        if model:
            # the chat template adds a few dozen tokens on top, so this is a lower bound
            over = [i for i, n in sizes if n > max_len]
            report["length"]["over_max_len"] = {"max_len": max_len, "count": len(over), "rows": over[:10]}

    if against:
        test = load(against)
        seen = set(by_prompt)
        leaked = [i for i, r in enumerate(test) if fields(r)[0] != "bad" and key(fields(r)[1]) in seen]
        report["overlap_with_test"] = {"against": against, "test_rows": len(test), "count": len(leaked),
                                       "rows": leaked[:10]}
    return report


def failed(report: dict) -> bool:
    """True when the data has problems that will break or poison training."""
    return bool(report["problems"]) or report.get("overlap_with_test", {}).get("count", 0) > 0
