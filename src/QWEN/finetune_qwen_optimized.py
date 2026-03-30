"""
finetune_qwen_optimized.py — Fine-tune Qwen2.5-3B-Instruct on QIAS 2026 Mawarith data
using QLoRA (4-bit) on RTX 4060 (8GB VRAM)

Usage:
    python finetune_qwen_optimized.py --train_dir train --output_dir ./qwen_mawarith --epochs 3
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any


SYSTEM_PROMPT = """أنت فقيه متخصص في علم الفرائض والمواريث الإسلامية. 
مهمتك حل مسائل الإرث وإعطاء الإجابة بصيغة JSON منضبطة تمامًا.
يجب أن يحتوي JSON على: heirs, blocked, shares, awl_or_radd, post_tasil."""


def load_train_files(train_dir: Path, max_examples: int = 2000) -> List[Dict[str, Any]]:
    examples = []
    files = sorted(train_dir.glob("*.json"))
    print(f"[INFO] Found {len(files)} training files")
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            examples.extend(data)
    random.seed(42)
    random.shuffle(examples)
    examples = examples[:max_examples]
    print(f"[INFO] Training examples: {len(examples)}")
    return examples


def format_example(ex: Dict[str, Any]) -> Dict[str, str]:
    question    = ex["question"]
    target_json = json.dumps(ex["output"], ensure_ascii=False, indent=2)
    raw_answer  = ex.get("answer", "")
    think_match = re.search(r"<think>(.*?)</think>", raw_answer, re.DOTALL)
    reasoning   = think_match.group(1).strip() if think_match else ""
    full_answer = f"<think>\n{reasoning}\n</think>\n\n{target_json}" if reasoning else target_json
    return {"system": SYSTEM_PROMPT, "question": question, "answer": full_answer}


def build_chat_text(ex: Dict[str, str], tokenizer) -> str:
    messages = [
        {"role": "system",    "content": ex["system"]},
        {"role": "user",      "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",  required=True)
    parser.add_argument("--output_dir", default="./qwen_mawarith")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--max_length", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=2e-4)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"[INFO] CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw      = load_train_files(Path(args.train_dir))
    texts    = [build_chat_text(format_example(ex), tokenizer) for ex in raw]
    dataset  = Dataset.from_dict({"text": texts})
    print(f"[INFO] Dataset ready: {len(dataset)} examples | max_length: {args.max_length}")

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        packing=False,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    print("[INFO] Starting fine-tuning...")
    trainer.train()

    print(f"[INFO] Saving to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
