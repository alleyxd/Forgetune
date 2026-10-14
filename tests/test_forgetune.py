import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgetune import data  # noqa: E402
from forgetune.evaluate import norm  # noqa: E402


class FakeTok:
    eos_token = "</s>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        body = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        return body + ("<assistant>" if add_generation_prompt else "")


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_prompt_completion_rows(tmp_path):
    src = write_jsonl(tmp_path / "d.jsonl", [{"prompt": "2+2?", "completion": "4"}])
    ds = data.to_sft(data.read(src), FakeTok(), system="Be brief.")
    assert ds[0] == {"prompt": "<system>Be brief.<user>2+2?<assistant>", "completion": "4</s>"}


def test_messages_rows_use_last_assistant_turn(tmp_path):
    src = write_jsonl(tmp_path / "d.jsonl", [{"messages": [{"role": "user", "content": "hi"},
                                                           {"role": "assistant", "content": "hello"}]}])
    ds = data.to_sft(data.read(src), FakeTok())
    assert ds[0]["prompt"] == "<user>hi<assistant>" and ds[0]["completion"] == "hello</s>"


def test_preference_rows(tmp_path):
    src = write_jsonl(tmp_path / "d.jsonl", [{"prompt": "q", "chosen": "good", "rejected": "bad"}])
    ds = data.to_dpo(data.read(src), FakeTok())
    assert ds[0] == {"prompt": "<user>q<assistant>", "chosen": "good</s>", "rejected": "bad</s>"}


def test_limit(tmp_path):
    src = write_jsonl(tmp_path / "d.jsonl", [{"prompt": str(i), "completion": "x"} for i in range(10)])
    assert len(data.read(src, limit=3)) == 3


@pytest.mark.parametrize("a,b", [
    ("SELECT a FROM t WHERE x = 'Athens';", 'SELECT a FROM t WHERE x = "athens"'),
    ("```sql\nSELECT  a\nFROM t```", "SELECT a FROM t"),
])
def test_sql_normalisation(a, b):
    assert norm(a) == norm(b)
