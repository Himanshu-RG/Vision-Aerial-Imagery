"""H0 check: does an "all classes in one prompt" input exceed Grounding DINO's 256-token text limit?

Builds the prompt the way the HF Grounding DINO examples do ("name1. name2. ... nameN.", lower-case)
and counts BERT tokens, including [CLS] and [SEP]. Only tokenizer files are needed.

Usage:
    python scripts/check_prompt_length.py --classes configs/lae80c_classes_raw.txt
    python scripts/check_prompt_length.py --classes FILE --tokenizer <store>/checkpoints/grounding-dino-tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_TEXT_LEN = 256  # GroundingDinoConfig.max_text_len for IDEA-Research/grounding-dino-tiny (verify in config.json)


def build_prompt(names: list[str]) -> str:
    return " ".join(f"{n.strip().lower()}." for n in names)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def max_classes_that_fit(tokenizer, names: list[str], limit: int) -> int:
    for k in range(1, len(names) + 1):
        if count_tokens(tokenizer, build_prompt(names[:k])) > limit:
            return k - 1
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=Path, required=True, help="one class name per line")
    parser.add_argument("--tokenizer", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--revision", default=None, help="hub commit to use when --tokenizer is a repo id")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    names = [ln.strip() for ln in args.classes.read_text(encoding="utf-8").splitlines() if ln.strip()]
    prompt = build_prompt(names)
    n_tokens = count_tokens(tok, prompt)
    per_class = [count_tokens(tok, build_prompt([n])) - 2 for n in names]  # minus [CLS]/[SEP]
    result = {
        "n_classes": len(names),
        "prompt_tokens_with_special": n_tokens,
        "limit": MAX_TEXT_LEN,
        "exceeds_limit": n_tokens > MAX_TEXT_LEN,
        "classes_that_fit_in_one_prompt": max_classes_that_fit(tok, names, MAX_TEXT_LEN),
        "tokens_per_class_incl_period": {"min": min(per_class), "max": max(per_class),
                                         "mean": round(sum(per_class) / len(per_class), 2)},
        "longest_classes": sorted(zip(per_class, names), reverse=True)[:5],
        "tokenizer": args.tokenizer,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
