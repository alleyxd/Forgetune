# Forgetune

[![tests](https://github.com/alleyxd/Forgetune/actions/workflows/tests.yml/badge.svg)](https://github.com/alleyxd/Forgetune/actions/workflows/tests.yml)

Fine-tune a small open model on your own data with one command, and measure whether it helped.
Supervised fine-tuning with LoRA or QLoRA, preference tuning with DPO, held-out evaluation, and adapter merging,
on a single consumer GPU. Built on Hugging Face Transformers, TRL and PEFT.

```bash
forgetune sft  --model Qwen/Qwen3-1.7B --data examples/sql_train.jsonl --out runs/sql-1.7b
forgetune eval --model Qwen/Qwen3-1.7B --adapter runs/sql-1.7b --data examples/sql_test.jsonl
```

## Result: text-to-SQL

Task: given a table schema and a question, write the SQL query
([b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)). 20,000 training examples,
500 held-out test examples, one epoch, LoRA rank 16, RTX 5090. Scored by exact match after normalising case,
whitespace and quote style.

| Model | Exact match | Training time | Peak GPU memory |
|---|---|---|---|
| Qwen3-1.7B, prompted | 45.4% | - | - |
| **Qwen3-1.7B + LoRA** | **82.6%** | 19.5 min | 4.6 GB |
| Qwen3-4B, prompted | 50.2% | - | - |
| **Qwen3-4B + QLoRA (4-bit)** | **83.0%** | 28.9 min | 4.0 GB |

The prompted models write reasonable SQL but miss the dataset's conventions (which column to aggregate, exact
string literals). Twenty minutes of LoRA training teaches those conventions: the tuned 1.7B model beats the
prompted 4B model by more than 30 points.

Two things worth noticing. QLoRA trained the 4B model in less GPU memory (4.0 GB) than plain LoRA needed for the
1.7B one (4.6 GB), because the frozen base weights are stored in 4 bits. And after tuning, the bigger model is
barely better (83.0% against 82.6%) while being slower to serve, so for a narrow task like this the small
model is the one to deploy.

What the remaining errors look like (tuned 1.7B):

```
pred: SELECT attendance FROM table_name_63 WHERE game_site = "meadowlands" AND ...
gold: SELECT attendance FROM table_name_63 WHERE game_site = "the meadowlands" AND ...

pred: SELECT SUM(bronze) FROM table_name_15 WHERE silver = 2 AND total < 5
gold: SELECT AVG(bronze) FROM table_name_15 WHERE silver = 2 AND total < 5
```

Some are genuine mistakes, and some are ambiguous questions where exact match is a strict judge.

## Data formats

JSONL, one example per line, or any Hugging Face dataset with these columns:

```json
{"prompt": "Schema: ... Question: ...", "completion": "SELECT ..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "chosen": "a good answer", "rejected": "a worse answer"}
```

The first two are for `forgetune sft`, the third for `forgetune dpo`. Prompts are rendered once with the model's own
chat template, so training and inference see exactly the same text, and the loss is computed on the completion
only.

## Commands

```bash
forgetune sft   --model MODEL --data train.jsonl --out runs/x [--qlora] [--lora-r 16] [--epochs 1] [--system "..."]
forgetune dpo   --model MODEL --data pairs.jsonl --out runs/y [--beta 0.1]
forgetune check --data train.jsonl [--against test.jsonl] [--model MODEL --max-len 1024]
forgetune eval  --model MODEL [--adapter runs/x] --data test.jsonl [--save results.json]
forgetune merge --adapter runs/x --out merged/x      # one standalone model, faster to serve
forgetune compare base.json tuned.json            # per example: what the run fixed and what it broke
```

`check` reads the data the way training will and reports rows that would fail or teach nothing (missing fields,
empty targets, identical chosen and rejected answers), duplicate and contradictory prompts, rows longer than
`--max-len`, and test prompts that also appear in the training set. It exits with status 1 when it finds any of
these, so it can run in CI before a training job.

Each run writes `forgetune.json` next to the adapter with the settings, the final training loss, the time taken,
and the peak GPU memory.

`examples/make_dpo_data.py` builds preference pairs from a model's own mistakes: it runs the model on training
prompts and pairs each wrong answer (rejected) with the reference (chosen).

## Reproduce

```bash
uv venv && uv pip install -e ".[dev]"
python examples/make_sql_data.py
forgetune eval --model Qwen/Qwen3-1.7B --data examples/sql_test.jsonl
forgetune sft  --model Qwen/Qwen3-1.7B --data examples/sql_train.jsonl --out runs/sql-1.7b
forgetune eval --model Qwen/Qwen3-1.7B --adapter runs/sql-1.7b --data examples/sql_test.jsonl
```

## Tests

```bash
pytest -q
```

## License

MIT
