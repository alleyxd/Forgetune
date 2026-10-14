import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data


def norm(s: str) -> str:
    s = s.strip().strip("`").strip()
    s = re.sub(r"^sql\s*", "", s, flags=re.I)
    s = s.replace('"', "'").rstrip(";")
    return re.sub(r"\s+", " ", s).strip().lower()


def load(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int = 160, batch: int = 32) -> list[str]:
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out


def exact_match(base: str, eval_data: str, adapter: str | None = None, system: str | None = None,
                limit: int | None = None) -> dict:
    model, tok = load(base, adapter)
    ds = data.read(eval_data, limit=limit)
    prompts = [data.render_prompt(tok, r["prompt"], system) for r in ds]
    t = time.time()
    preds = generate(model, tok, prompts)
    secs = time.time() - t
    rows = [{"prompt": r["prompt"], "gold": r["completion"], "pred": p, "ok": norm(p) == norm(r["completion"])}
            for r, p in zip(ds, preds)]
    return {"model": base, "adapter": adapter, "n": len(rows),
            "exact_match": round(sum(r["ok"] for r in rows) / len(rows), 4),
            "ms_per_example": round(secs / len(rows) * 1000, 1), "rows": rows}


def save(result: dict, path: str):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
