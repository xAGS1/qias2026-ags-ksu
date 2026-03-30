""" March 2026
run_gpt_submission.py
=====================
GPT-5.4 via OpenAI API.

Uses embedded prompt (rules + dev examples) as described.

Setup:
    pip install requests
    set OPENAI_API_KEY=sk-your-key-here   (Windows)
    export OPENAI_API_KEY=sk-your-key-here (Mac/Linux)

Usage:
    python run_gpt_submission.py
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests

# ── Edit these paths ────────────────────────────────────────────────────────
DEV_FILE_1 = Path("data/dev/qias2025_almawarith_part1.json")
DEV_FILE_2 = Path("data/dev/qias2025_almawarith_part61.json")
TEST_FILE = Path("data/test/qias2025_almawarith_test_id_question.json")
OUT_FILE = Path("prediction.json")
RESUME_FILE = Path("prediction_progress.json")
STATS_FILE = Path("run_stats.json")

OPENAI_API_URL = os.getenv("OPENAI_API_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

# Paper-matching defaults.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
DEV_PER_BATCH = int(os.getenv("DEV_PER_BATCH", "20"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

# Safe runtime controls. These do not change the paper method.
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "3"))
REQUESTS_PER_DAY = int(os.getenv("REQUESTS_PER_DAY", "200"))
TOKENS_PER_DAY = int(os.getenv("TOKENS_PER_DAY", "850000"))
MIN_SECONDS_BETWEEN_REQUESTS = float(os.getenv("MIN_SECONDS_BETWEEN_REQUESTS", "35"))

RESUME = True
HTTP_TIMEOUT = 300
MAX_RETRIES = 5

# ── System prompt (rules + hard cases) ──────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in Islamic inheritance law (علم الفرائض والمواريث الإسلامية) implementing the Almawarith calculator logic (جمهور school - Shafi'i/Maliki opinion).

Pay EXTRA attention to these hard families:

A) الجد مع الإخوة (grandfather + siblings):
   - Grandfather gets the BETTER of: his Quranic share OR equal share with brothers as asaba
   - Use muqasama, thuluth, or sudus — whichever is best for grandfather
   - Grandfather NEVER gets less than 1/6

B) الأخت مع البنات/بنات الابن (sister becomes عصبة مع الغير):
   - Full sister or half sister (لأب) becomes residual عصبة when there is a daughter or granddaughter
   - She takes the remainder after fixed shares, NOT her Quranic 1/2 or 2/3

C) الرد مع الزوج/الزوجة (radd with spouse):
   - Spouse does NOT participate in radd (جمهور opinion)
   - Radd is distributed only among non-spouse heirs with fixed shares
   - Spouse keeps their fixed share

D) تعدد الجدات (multiple grandmothers):
   - Closer grandmother blocks farther one from SAME side
   - Maternal grandmother (أم الأم) blocks all grandmothers on both sides if closer

E) الأكدريّة: Husband + mother + grandfather + full sister
   - Sister gets 1/2, then shares with grandfather

F) المشتركة/الحمارية: Husband + mother + 2+ uterine siblings + full siblings
   - Full siblings share WITH uterine siblings in the 1/3

G) بنات الابن مع ابن الابن في درجتها:
   - Son's son at same level makes son's daughters عصبة, share للذكر مثل حظ الأنثيين

H) العمريّتان: Husband/wife + both parents
   - Mother gets 1/3 of REMAINDER after spouse, not 1/3 of total

CRITICAL RULES:
- awl_or_radd only scored correct if BOTH heirs AND shares are correct first
- fraction format: always "N/D" string e.g. "1/6", never decimals
- ALL heirs from heirs[] must appear in post_tasil distribution[]
- per_head_percent must be correct to 2 decimal places

Output only valid JSON array. No explanation, no markdown."""

USER_PROMPT_TEMPLATE = """\
Here are {n_dev} gold development examples to learn from:

{dev_examples}

---

Now solve these {n_test} test problems.

Output a JSON array only. No markdown, no fences, no explanation.

Each entry:
{{
  "id": "...",
  "question": "...",
  "output": {{
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
}}

Problems:
{problems}
"""


def approx_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 4)


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def load_all_dev_examples(f1: Path, f2: Path) -> list[dict]:
    examples: list[dict] = []
    for f in (f1, f2):
        if f.exists():
            examples.extend(json.loads(f.read_text(encoding="utf-8")))
    return examples


def get_dev_context(all_dev: list[dict], batch_idx: int, examples_per_batch: int) -> str:
    n = len(all_dev)
    if n == 0:
        raise RuntimeError("No dev examples loaded")
    start = (batch_idx * examples_per_batch) % n
    indices = [(start + i) % n for i in range(examples_per_batch)]
    selected = [all_dev[i] for i in indices]
    return json.dumps(selected, ensure_ascii=False, indent=2)


def extract_json_array(text: str) -> list[dict]:
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found")

    depth = 0
    in_string = False
    for i in range(start, len(text)):
        c = text[i]
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        if not in_string:
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
        if depth == 0:
            return json.loads(text[start:i + 1])

    raise ValueError("Unclosed JSON array")


class RateState:
    def __init__(self) -> None:
        self.request_times: deque[float] = deque()
        self.last_request_time: float = 0.0
        self.requests_today: int = 0
        self.tokens_today: int = 0

    def load_from_stats(self, stats: dict[str, Any]) -> None:
        self.requests_today = int(stats.get("requests_today", 0))
        self.tokens_today = int(stats.get("tokens_today", 0))

    def to_stats(self) -> dict[str, Any]:
        return {
            "requests_today": self.requests_today,
            "tokens_today": self.tokens_today,
            "updated_at_unix": time.time(),
        }

    def wait_for_minute_window(self) -> None:
        now = time.time()

        if self.last_request_time > 0:
            elapsed = now - self.last_request_time
            if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
                time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

        now = time.time()
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        if len(self.request_times) >= REQUESTS_PER_MINUTE:
            sleep_for = 60 - (now - self.request_times[0]) + 0.5
            if sleep_for > 0:
                print(f"  Minute window full, sleeping {sleep_for:.1f}s...")
                time.sleep(sleep_for)

        now = time.time()
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

    def register_request(self, used_tokens: int) -> None:
        now = time.time()
        self.request_times.append(now)
        self.last_request_time = now
        self.requests_today += 1
        self.tokens_today += used_tokens

    def can_send(self, estimated_tokens: int) -> tuple[bool, str]:
        if self.requests_today >= REQUESTS_PER_DAY:
            return False, f"soft daily request cap reached: {self.requests_today}/{REQUESTS_PER_DAY}"
        if self.tokens_today + estimated_tokens > TOKENS_PER_DAY:
            return False, f"soft daily token cap would be exceeded: {self.tokens_today}+{estimated_tokens}>{TOKENS_PER_DAY}"
        return True, ""


def build_user_message(problems: list[dict], dev_context: str) -> str:
    problems_text = json.dumps(
        [{"id": p["id"], "question": p["question"]} for p in problems],
        ensure_ascii=False,
        indent=2,
    )
    return USER_PROMPT_TEMPLATE.format(
        n_dev=len(json.loads(dev_context)),
        dev_examples=dev_context,
        n_test=len(problems),
        problems=problems_text,
    )


def call_openai(
    problems: list[dict],
    dev_context: str,
    rate_state: RateState,
) -> tuple[str, dict[str, int]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    user_msg = build_user_message(problems, dev_context)
    est_input_tokens = approx_tokens_from_text(SYSTEM_PROMPT) + approx_tokens_from_text(user_msg)
    est_total_tokens = est_input_tokens + MAX_OUTPUT_TOKENS

    can_send, reason = rate_state.can_send(est_total_tokens)
    if not can_send:
        raise RuntimeError(f"stopping before request because {reason}")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        rate_state.wait_for_minute_window()
        try:
            response = requests.post(
                OPENAI_API_URL,
                json=payload,
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(120, 15 * (2 ** attempt))
                print(f"  Rate limited (429), sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:
                wait = min(120, 10 * (2 ** attempt))
                print(f"  Server error {response.status_code}, sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            message_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", est_input_tokens))
            completion_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

            rate_state.register_request(total_tokens)
            return message_text, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"HTTP request failed after retries: {e}") from e
            wait = min(120, 10 * (2 ** attempt))
            print(f"  Network/transient error, sleeping {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError("Max retries exceeded")


def main() -> None:
    examples = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    print(f"Test set: {len(examples)} examples | Model: {OPENAI_MODEL}")
    print(
        f"Batch size={BATCH_SIZE}, dev per batch={DEV_PER_BATCH}, "
        f"max output tokens={MAX_OUTPUT_TOKENS}"
    )
    print(
        f"Soft caps: {REQUESTS_PER_MINUTE} RPM, {REQUESTS_PER_DAY} RPD, "
        f"{TOKENS_PER_DAY} TPD, min gap {MIN_SECONDS_BETWEEN_REQUESTS}s"
    )

    all_dev = load_all_dev_examples(DEV_FILE_1, DEV_FILE_2)
    print(f"Dev examples loaded: {len(all_dev)} total, {DEV_PER_BATCH} per request (rotating)")

    done: dict[str, dict] = {}
    if RESUME and RESUME_FILE.exists():
        done = {p["id"]: p for p in load_json(RESUME_FILE, [])}
        print(f"Resuming from {len(done)} already done")

    stats = load_json(STATS_FILE, {})
    rate_state = RateState()
    rate_state.load_from_stats(stats)

    remaining = [e for e in examples if e["id"] not in done]
    print(f"Remaining: {len(remaining)}")
    print(f"Usage so far today: requests={rate_state.requests_today}, tokens={rate_state.tokens_today}\n")

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"[{batch_num}/{total_batches}] {batch[0]['id']} ... {batch[-1]['id']}")

        try:
            dev_context = get_dev_context(all_dev, batch_start // BATCH_SIZE, DEV_PER_BATCH)
            t0 = time.time()
            raw, usage = call_openai(batch, dev_context, rate_state)
            results = extract_json_array(raw)

            for result in results:
                done[result["id"]] = result

            for ex in batch:
                if ex["id"] not in done:
                    done[ex["id"]] = {
                        "id": ex["id"],
                        "question": ex["question"],
                        "output": {},
                    }

            dt = time.time() - t0
            print(
                f"  OK: {len(results)} in {dt:.1f}s | "
                f"prompt={usage['prompt_tokens']} completion={usage['completion_tokens']} total={usage['total_tokens']}"
            )

        except Exception as e:
            print(f"  STOPPED/FAILED: {e}")
            for ex in batch:
                if ex["id"] not in done:
                    done[ex["id"]] = {
                        "id": ex["id"],
                        "question": ex["question"],
                        "output": {},
                    }
            save_json(RESUME_FILE, [done[e["id"]] for e in examples if e["id"] in done])
            save_json(STATS_FILE, rate_state.to_stats())
            break

        save_json(RESUME_FILE, [done[e["id"]] for e in examples if e["id"] in done])
        save_json(STATS_FILE, rate_state.to_stats())

    all_preds = [done[e["id"]] for e in examples if e["id"] in done]
    valid = sum(1 for p in all_preds if p.get("output") and "heirs" in p["output"])
    print(f"\nCurrent saved predictions: {len(all_preds)}/{len(examples)} | valid: {valid}")

    submission = []
    for e in examples:
        if e["id"] in done:
            submission.append(
                {"id": done[e["id"]]["id"], "question": done[e["id"]]["question"], "output": done[e["id"]]["output"]}
            )
    save_json(OUT_FILE, submission)
    print(f"Saved to {OUT_FILE}")
    print(f"Saved stats to {STATS_FILE}")


if __name__ == "__main__":
    main()