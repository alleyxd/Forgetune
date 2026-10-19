"""Compare two saved evaluations of the same test set, example by example.

An accuracy change hides how it happened: 45% to 83% could mean nothing broke, or that many answers broke while more
were fixed. Saved results (forgetune eval --save) keep every prediction, so this lines them up by prompt and counts
what each run fixed and what it broke.
"""
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(before: str, after: str, show: int = 3) -> dict:
    a, b = load(before), load(after)
    rows_a = {r["prompt"]: r for r in a["rows"]}
    rows_b = {r["prompt"]: r for r in b["rows"]}
    shared = [p for p in rows_a if p in rows_b]
    fixed = [p for p in shared if not rows_a[p]["ok"] and rows_b[p]["ok"]]
    broken = [p for p in shared if rows_a[p]["ok"] and not rows_b[p]["ok"]]

    def example(p):
        return {"prompt": p[:300], "gold": rows_a[p]["gold"], "before": rows_a[p]["pred"], "after": rows_b[p]["pred"]}

    acc = lambda rows: round(sum(rows[p]["ok"] for p in shared) / max(1, len(shared)), 4)
    return {"before": before, "after": after, "compared": len(shared),
            "only_in_before": len(rows_a) - len(shared), "only_in_after": len(rows_b) - len(shared),
            "accuracy_before": acc(rows_a), "accuracy_after": acc(rows_b),
            "fixed": len(fixed), "broken": len(broken),
            "examples_fixed": [example(p) for p in fixed[:show]],
            "examples_broken": [example(p) for p in broken[:show]]}
