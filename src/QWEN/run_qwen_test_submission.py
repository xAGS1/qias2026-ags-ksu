"""
run_qwen_test_submission.py
===========================
Run fine-tuned Qwen2.5-3B on the 500 test examples and produce
a CodaBench-ready submission.json.

Usage (from project root):
    python qwen/run_qwen_test_submission.py

Output:
    configs/output/Qwen-3B-Mawarith/submission.json  ← upload this to CodaBench
    configs/output/Qwen-3B-Mawarith/pred_qwen.json   ← full predictions with metadata
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
TEST_FILE    = Path(r"qias2025_almawarith_test_id_question.json")
ADAPTER_PATH = Path(r"qwen_mawarith")
OUT_DIR      = Path(r"Qwen-3B-Mawarith")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE       = OUT_DIR / "pred_qwen.json"
SUBMISSION_FILE = OUT_DIR / "submission.json"

MAX_NEW_TOKENS = 1200
RESUME         = True   # skip already-done ids if pred_qwen.json exists


# ── load model ─────────────────────────────────────────────────────────────
def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    base_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Qwen] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[Qwen] Loading base model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    print(f"[Qwen] Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(
        base, str(ADAPTER_PATH),
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    print("[Qwen] Model ready!\n")
    return model, tokenizer


# ── build prompt ────────────────────────────────────────────────────────────
# Import from your existing prompts module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
USE_PIPELINE_PROMPT = False
print("[INFO] Using single-shot fallback prompt")


FALLBACK_PROMPT = """\
أنت خبير في علم الفرائض. أجب بـ JSON فقط، بدون أي تفكير أو شرح أو مقدمة.

السؤال: {question}

```json
{{
  "heirs": [{{"heir": "...", "count": 1}}],
  "blocked": [{{"heir": "...", "count": 1}}],
  "shares": [{{"heir": "...", "count": 1, "fraction": "1/6"}}],
  "awl_or_radd": null,
  "awl_stage": null,
  "post_tasil": {{
    "total_shares": 6,
    "distribution": [{{"heir": "...", "count": 1, "per_head_shares": "1/6", "per_head_percent": 16.67}}]
  }}
}}
```
"""


def build_prompt(question: str) -> str:
    if USE_PIPELINE_PROMPT:
        return build_structured_extraction_prompt(question)
    return FALLBACK_PROMPT.format(question=question)


# ── JSON extraction ──────────────────────────────────────────────────────────
def extract_json(text: str) -> dict | None:
    # Strip <think> block
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):]

    for ch in ("\ufeff", "\u200f", "\u200e", "\u202a", "\u202b", "\u202c"):
        text = text.replace(ch, "")
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE)

    candidates = []
    for m in re.finditer(r"{", text):
        start = m.start()
        brace, in_string = 0, False
        for i in range(start, len(text)):
            c = text[i]
            if c == '"' and (i == 0 or text[i - 1] != "\\"):
                in_string = not in_string
            if not in_string:
                if c == "{":
                    brace += 1
                elif c == "}":
                    brace -= 1
            if brace == 0:
                chunk = text[start:i + 1]
                try:
                    candidates.append(json.loads(chunk))
                except Exception:
                    pass
                break

    if not candidates:
        return None
    return max(candidates, key=lambda d: len(json.dumps(d)))


# ── inference ────────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, question: str) -> tuple[dict | None, str]:
    import torch
    prompt = build_prompt(question)
    messages = [
        {"role": "system", "content": "أنت خبير في علم الفرائض الإسلامية. تجيب دائماً بـ JSON فقط بدون أي تفكير أو شرح."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # disables <think> block for Qwen3
    )
    # Also prepend the opening brace to force JSON immediately
    text += "```json\n{"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    # Prepend the forced opening we added
    raw = "{" + raw
    parsed = extract_json(raw)
    return parsed, raw


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # Load test set
    examples = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    print(f"[INFO] Test set: {len(examples)} examples")

    # Resume support
    done_ids = set()
    all_preds = []
    if RESUME and PRED_FILE.exists():
        all_preds = json.loads(PRED_FILE.read_text(encoding="utf-8"))
        done_ids  = {p["id"] for p in all_preds}
        print(f"[INFO] Resuming — {len(done_ids)} already done")

    remaining = [ex for ex in examples if ex["id"] not in done_ids]
    print(f"[INFO] Remaining: {len(remaining)} examples\n")

    if not remaining:
        print("[INFO] All done! Building submission...")
    else:
        model, tokenizer = load_model()
        for i, ex in enumerate(remaining):
            t0 = time.time()
            parsed, raw = run_inference(model, tokenizer, ex["question"])
            elapsed = time.time() - t0

            status = "✅" if parsed else "❌"
            print(f"[{i+1}/{len(remaining)}] {ex['id']} {status}  ({elapsed:.1f}s)")

            # Debug: save full raw for failed cases
            if not parsed:
                debug_file = OUT_DIR / f"debug_{ex['id']}.txt"
                debug_file.write_text(raw, encoding="utf-8")
                print(f"         Raw saved → {debug_file}")

            all_preds.append({
                "id":       ex["id"],
                "question": ex["question"],
                "output":   parsed or {},
                "raw":      raw[:800],
            })

            # Save after every example (safe resume)
            PRED_FILE.write_text(
                json.dumps(all_preds, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        json_ok  = sum(1 for p in all_preds if p["output"])
        json_fail = len(all_preds) - json_ok
        print(f"\n[INFO] JSON parsed: {json_ok}/{len(all_preds)}  failed: {json_fail}")

    # Build submission.json
    submission = []
    for p in all_preds:
        submission.append({
            "id":       p["id"],
            "question": p["question"],
            "output":   p["output"],
        })

    SUBMISSION_FILE.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n[INFO] submission.json saved → {SUBMISSION_FILE}")
    print(f"[INFO] Total entries: {len(submission)}")
    print(f"\n✅ Upload {SUBMISSION_FILE} to CodaBench (zip it first!)")


if __name__ == "__main__":
    main()
