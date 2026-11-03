"""Loading training data.

Accepted JSONL rows (or Hugging Face datasets with these columns):
  {"prompt": "...", "completion": "..."}                  supervised fine-tuning
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}   supervised, last assistant turn is the target
  {"prompt": "...", "chosen": "...", "rejected": "..."}   preference tuning (DPO)

Prompts are rendered with the model's own chat template here, once, so training and generation see exactly
the same text.
"""
import json
from pathlib import Path

from datasets import Dataset, load_dataset


def read(source: str, split: str = "train", limit: int | None = None) -> Dataset:
    if Path(source).exists():
        rows = [json.loads(l) for l in Path(source).read_text(encoding="utf-8").splitlines() if l.strip()]
        # from_list takes its columns from the first row, so give every row every key seen in the file
        keys = list(dict.fromkeys(k for r in rows for k in r))
        ds = Dataset.from_list([{k: r.get(k) for k in keys} for r in rows])
    else:
        name, _, config = source.partition(":")
        ds = load_dataset(name, config or None, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def render_messages(tok, msgs: list[dict], system: str | None = None) -> str:
    # a system message in the data wins over the default one
    if system and not (msgs and msgs[0]["role"] == "system"):
        msgs = [{"role": "system", "content": system}] + msgs
    # enable_thinking=False keeps Qwen3-style models from emitting a reasoning block; other templates ignore it
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def render_prompt(tok, user: str, system: str | None = None) -> str:
    return render_messages(tok, [{"role": "user", "content": user}], system)


def to_sft(ds: Dataset, tok, system: str | None = None) -> Dataset:
    end = tok.eos_token or ""

    def conv(row):
        if "messages" in row and row["messages"]:
            msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
            # the whole conversation before the last turn is the prompt, earlier assistant turns included
            return {"prompt": render_messages(tok, msgs[:-1], system), "completion": msgs[-1]["content"] + end}
        return {"prompt": render_prompt(tok, row["prompt"], system), "completion": row["completion"] + end}

    return ds.map(conv, remove_columns=ds.column_names)


def to_dpo(ds: Dataset, tok, system: str | None = None) -> Dataset:
    end = tok.eos_token or ""
    return ds.map(lambda r: {"prompt": render_prompt(tok, r["prompt"], system),
                             "chosen": r["chosen"] + end, "rejected": r["rejected"] + end},
                  remove_columns=ds.column_names)
