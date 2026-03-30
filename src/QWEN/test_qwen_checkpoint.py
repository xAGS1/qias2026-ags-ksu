"""
test_qwen_checkpoint.py — Test a QLoRA checkpoint on a few dev examples
Runs on CPU so it doesn't interfere with training on GPU

Usage:
    python test_qwen_checkpoint.py --checkpoint ./qwen_mawarith/checkpoint-300
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./qwen_mawarith/checkpoint-300")
    parser.add_argument("--data", default=r"dev\qias2025_almawarith_part1.json")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to test")
    parser.add_argument("--out", default="./qwen_test_output.json", help="Save results to JSON file")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
    checkpoint = Path(args.checkpoint)

    print(f"[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"[INFO] Loading base model on CPU (won't interfere with training)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",        # CPU only — no GPU conflict
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Resolve absolute path — PEFT needs local_files_only for local dirs
    checkpoint_abs = checkpoint.resolve()
    print(f"[INFO] Loading LoRA adapter from {checkpoint_abs}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_abs),
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    print("[INFO] Model ready")

    # Load dev examples
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    examples = data[:args.n]

    SYSTEM = """أنت فقيه متخصص في علم الفرائض والمواريث الإسلامية.
مهمتك حل مسائل الإرث وإعطاء الإجابة بصيغة JSON منضبطة تمامًا."""

    results = []

    for i, ex in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"Example {i+1}: {ex['id']}")
        print(f"Question: {ex['question'][:100]}...")

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")

        import time
        print("[INFO] Generating (CPU, may take 1-2 min)...")
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        n_new = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"[INFO] Generated {n_new} tokens in {elapsed:.1f}s ({n_new/elapsed:.1f} tok/s)")

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print(f"\nModel output:\n{response[:800]}")

        # Try to parse JSON
        parsed_output = None
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_output = json.loads(json_match.group(0))
                print(f"\n✅ Valid JSON with keys: {list(parsed_output.keys())}")
            else:
                print("\n⚠️ No JSON found in output")
        except Exception as e:
            print(f"\n⚠️ JSON parse error: {e}")

        # Compare with gold
        gold = ex.get("output", {})
        if gold:
            gold_heirs = [h["heir"] for h in gold.get("heirs", [])]
            print(f"\nGold heirs: {gold_heirs}")

        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "raw_output": response,
            "parsed_output": parsed_output,
            "gold_output": gold,
        })

    # Save results to file
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] Results saved to: {out_path.resolve()}")
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
