from __future__ import annotations

import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ======================================================
# Optional backends
# ======================================================
try:
    import google.generativeai as genai

    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False


# ======================================================
# Fanar config
# ======================================================
FANAR_API_URL = "https://api.fanar.qa/v1/chat/completions"
FANAR_MODEL_NAME = os.getenv("FANAR_MODEL_NAME", "Fanar-Sadiq")

FANAR_DEFAULT_PARAMS = {
    "max_tokens": 6000,
    "temperature": 0.0,
}


# ======================================================
# Raw LLM calls
# ======================================================
def call_fanar_raw(prompt: str) -> str:
    api_key = os.getenv("FANAR_API_KEY")
    if not api_key:
        raise RuntimeError("FANAR_API_KEY missing in environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": FANAR_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        **FANAR_DEFAULT_PARAMS,
    }

    r = requests.post(
        FANAR_API_URL,
        json=payload,
        headers=headers,
        timeout=180,
    )
    r.raise_for_status()

    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Invalid Fanar response:\n{r.text}") from e


def call_gemini_raw(prompt: str) -> str:
    if not GEMINI_OK:
        raise RuntimeError("Gemini backend not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    out = model.generate_content(prompt)
    return out.text or ""



def call_openai_raw(prompt: str) -> str:
    """OpenAI API — GPT-4o, GPT-5.4, etc."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.0,
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload, headers=headers, timeout=300,
    )
    r.raise_for_status()
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Invalid OpenAI response:\n{r.text}") from e


def call_deepseek_raw(prompt: str) -> str:
    """DeepSeek API — OpenAI-compatible."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in environment")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
        "temperature": 0.0,
    }
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        json=payload, headers=headers, timeout=300,
    )
    r.raise_for_status()
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Invalid DeepSeek response:\n{r.text}") from e

def call_llm_raw(prompt: str, model_key: str = "fanar") -> str:
    model_key = model_key.lower()

    if model_key == "fanar":
        return call_fanar_raw(prompt)
    if model_key == "gemini":
        return call_gemini_raw(prompt)

    if model_key in ("openai", "gpt", "gpt-4o", "gpt-5.4"):
        return call_openai_raw(prompt)
    if model_key == "deepseek":
        return call_deepseek_raw(prompt)
    raise ValueError(f"Unsupported model_key: {model_key}")


# ======================================================
# Robust JSON extraction
# ======================================================
def _extract_json(text: str) -> str:
    """
    Extract the largest valid JSON object from raw LLM output.
    Handles:
      - invisible Unicode chars
      - markdown fences
      - extra text before/after JSON
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    for ch in ("\ufeff", "\u200f", "\u200e", "\u202a", "\u202b", "\u202c"):
        text = text.replace(ch, "")

    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE)

    candidates: list[str] = []

    for m in re.finditer(r"{", text):
        start = m.start()
        brace = 0
        in_string = False

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
                chunk = text[start : i + 1]
                try:
                    json.loads(chunk)
                    candidates.append(chunk)
                except Exception:
                    pass
                break

    if not candidates:
        return ""

    return max(candidates, key=len)


def call_llm_json(prompt: str, model_key: str = "fanar") -> dict:
    raw = call_llm_raw(prompt, model_key)
    extracted = _extract_json(raw)

    if not extracted:
        raise RuntimeError("Failed to extract JSON from LLM output")

    parsed = json.loads(extracted)
    if not isinstance(parsed, dict):
        raise RuntimeError("Extracted JSON is not an object")
    return parsed


# ======================================================
# Unified client (used by pipeline)
# ======================================================
class LLMClient:
    """Unified LLM client used across the project."""

    def call(self, model_key: str, prompt: str) -> str:
        return call_llm_raw(prompt, model_key)

    def call_text(self, model_key: str, prompt: str) -> str:
        return call_llm_raw(prompt, model_key)

    def call_json(self, model_key: str, prompt: str) -> dict:
        return call_llm_json(prompt, model_key)


# Global singleton
llm_client = LLMClient()