"""Preference pairs from a model's own mistakes: run the model on training prompts, keep the ones it gets wrong,
and pair the gold SQL (chosen) with its wrong answer (rejected). Writes examples/sql_dpo.jsonl.

python examples/make_dpo_data.py --model Qwen/Qwen3-1.7B [--adapter runs/sql-1.7b] --n 4000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgetune import data  # noqa: E402
from forgetune.evaluate import generate, load, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter")
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--out", default=str(Path(__file__).parent / "sql_dpo.jsonl"))
args = ap.parse_args()

rows = [json.loads(l) for l in open(Path(__file__).parent / "sql_train.jsonl", encoding="utf-8")][-args.n:]
model, tok = load(args.model, args.adapter)
preds = generate(model, tok, [data.render_prompt(tok, r["prompt"]) for r in rows])
pairs = [{"prompt": r["prompt"], "chosen": r["completion"], "rejected": p.strip()}
         for r, p in zip(rows, preds) if norm(p) != norm(r["completion"]) and p.strip()]
with open(args.out, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"{len(pairs)} preference pairs from {len(rows)} prompts -> {args.out}")
