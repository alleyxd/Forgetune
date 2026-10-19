import argparse
import json


def main():
    ap = argparse.ArgumentParser(prog="forgetune", description="Fine-tune a small model with LoRA in one command.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("sft", "supervised fine-tuning"), ("dpo", "preference tuning")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--model", required=True, help="Hugging Face model id")
        p.add_argument("--data", required=True, help="JSONL file or Hugging Face dataset id")
        p.add_argument("--eval-data")
        p.add_argument("--out", required=True)
        p.add_argument("--epochs", type=float, default=1)
        p.add_argument("--lr", type=float, default=2e-4 if name == "sft" else 5e-6)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--accum", type=int, default=2)
        p.add_argument("--max-len", type=int, default=1024)
        p.add_argument("--lora-r", type=int, default=16)
        p.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (QLoRA)")
        p.add_argument("--system", help="system prompt added to every example")
        p.add_argument("--limit", type=int)
        if name == "dpo":
            p.add_argument("--beta", type=float, default=0.1)
    e = sub.add_parser("eval", help="exact-match accuracy on prompt/completion data")
    e.add_argument("--model", required=True)
    e.add_argument("--adapter")
    e.add_argument("--data", required=True)
    e.add_argument("--system")
    e.add_argument("--limit", type=int)
    e.add_argument("--save")
    c = sub.add_parser("check", help="find problems in a dataset before training on it")
    c.add_argument("--data", required=True)
    c.add_argument("--against", help="test set to check for prompts that also appear in --data")
    c.add_argument("--model", help="count tokens with this model's tokenizer instead of characters")
    c.add_argument("--max-len", type=int, default=1024)
    c.add_argument("--limit", type=int)
    m = sub.add_parser("merge", help="merge a LoRA adapter into its base model")
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd in ("sft", "dpo"):
        from .train import run
        extra = {"beta": args.beta} if args.cmd == "dpo" else {}
        info = run(args.cmd, args.model, args.data, args.out, eval_data=args.eval_data, epochs=args.epochs,
                   lr=args.lr, batch=args.batch, accum=args.accum, max_len=args.max_len, r=args.lora_r,
                   qlora=args.qlora, system=args.system, limit=args.limit, **extra)
        print(json.dumps(info, indent=1))
    elif args.cmd == "eval":
        from .evaluate import exact_match, save
        res = exact_match(args.model, args.data, args.adapter, args.system, args.limit)
        if args.save:
            save(res, args.save)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    elif args.cmd == "check":
        from .check import check, failed
        report = check(args.data, args.against, args.model, args.max_len, args.limit)
        print(json.dumps(report, indent=1, ensure_ascii=False))
        raise SystemExit(1 if failed(report) else 0)
    else:
        from .train import merge
        merge(args.adapter, args.out)
        print("merged into", args.out)


if __name__ == "__main__":
    main()
