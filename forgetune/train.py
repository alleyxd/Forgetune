import json
import time
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import data

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(name: str, qlora: bool):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if qlora:
        # 4-bit NF4 base weights, bf16 compute: roughly a quarter of the memory of the full model
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(name, **kw), tok


def lora(r: int):
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")


def run(method: str, model_name: str, train: str, out: str, *, eval_data: str | None = None, epochs: float = 1,
        lr: float = 2e-4, batch: int = 8, accum: int = 2, max_len: int = 1024, r: int = 16, qlora: bool = False,
        system: str | None = None, limit: int | None = None, beta: float = 0.1):
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    model, tok = load_model(model_name, qlora)
    ds = data.read(train, limit=limit)
    ev = data.read(eval_data) if eval_data else None
    common = dict(output_dir=out, num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=batch,
                  gradient_accumulation_steps=accum, lr_scheduler_type="cosine", warmup_steps=0.03,
                  logging_steps=20, save_strategy="no", bf16=True, gradient_checkpointing=True, report_to="none")
    t = time.time()
    if method == "sft":
        args = SFTConfig(**common, max_length=max_len, completion_only_loss=True)
        trainer = SFTTrainer(model=model, args=args, train_dataset=data.to_sft(ds, tok, system),
                             eval_dataset=data.to_sft(ev, tok, system) if ev else None,
                             processing_class=tok, peft_config=lora(r))
    elif method == "dpo":
        args = DPOConfig(**common, max_length=max_len, beta=beta)
        trainer = DPOTrainer(model=model, args=args, train_dataset=data.to_dpo(ds, tok, system),
                             eval_dataset=data.to_dpo(ev, tok, system) if ev else None,
                             processing_class=tok, peft_config=lora(r))
    else:
        raise ValueError(method)
    result = trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    info = {"method": method, "base": model_name, "examples": len(ds), "epochs": epochs, "lora_r": r,
            "qlora": qlora, "system": system, "train_loss": round(result.training_loss, 4),
            "minutes": round((time.time() - t) / 60, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    Path(out, "forgetune.json").write_text(json.dumps(info, indent=1))
    return info


def merge(adapter: str, out: str):
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(adapter, dtype=torch.bfloat16)
    model.merge_and_unload().save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
