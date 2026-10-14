"""Text-to-SQL demo data from b-mc2/sql-create-context: schema + question -> SQL.
Writes examples/sql_train.jsonl (20,000) and examples/sql_test.jsonl (500), disjoint."""
import json
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent
PROMPT = "Write one SQLite query that answers the question. Reply with the SQL only.\n\nSchema:\n{context}\n\nQuestion: {question}"

ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=0)
rows = [{"prompt": PROMPT.format(context=r["context"], question=r["question"]), "completion": r["answer"]} for r in ds]
test, train = rows[:500], rows[500:20500]
for name, part in (("sql_train", train), ("sql_test", test)):
    with open(HERE / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
print(len(train), "train,", len(test), "test")
