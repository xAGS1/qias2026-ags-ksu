from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List

from .llm_client import llm_client
from .prompts import (
    build_heirs_extraction_prompt,
    build_reasoning_prompt,
    build_retry_structured_prompt,
    build_structured_extraction_prompt,
)


# ============================================================
# Fraction / validation helpers
# ============================================================

def _parse_fraction(frac: str):
    if not isinstance(frac, str):
        return None
    frac = frac.strip()
    if "/" not in frac:
        return None
    try:
        a, b = frac.split("/", 1)
        return Fraction(int(a.strip()), int(b.strip()))
    except Exception:
        return None


def _safe_list(x):
    return x if isinstance(x, list) else []


def validate_structured_answer(structured: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []

    heirs   = _safe_list(structured.get("heirs"))
    blocked = _safe_list(structured.get("blocked"))
    shares  = _safe_list(structured.get("shares"))

    heirs_names   = {h.get("heir") for h in heirs   if isinstance(h, dict)}
    blocked_names = {h.get("heir") for h in blocked if isinstance(h, dict)}
    share_names   = [s.get("heir") for s in shares   if isinstance(s, dict)]

    overlap = blocked_names.intersection(set(share_names))
    if overlap:
        errors.append(f"Blocked heirs also appear in shares: {sorted(overlap)}")

    unknown_share_heirs = [n for n in share_names if n not in heirs_names]
    if unknown_share_heirs:
        errors.append(f"Shares contain heirs not in original heirs list: {unknown_share_heirs}")

    fixed_sum      = Fraction(0, 1)
    residual_count = 0

    for s in shares:
        if not isinstance(s, dict):
            continue
        frac = s.get("fraction")
        if frac in ("باقي التركة", "باقى التركة", "كل التركة"):
            residual_count += 1
            continue
        parsed = _parse_fraction(frac)
        if parsed is None:
            errors.append(f"Invalid fraction format: {frac}")
            continue
        fixed_sum += parsed

    # Note: multiple residual shares are valid in عصبة مع الغير cases
    # e.g. أب الأب + أخوات both share باقي التركة
    # Only flag if there are 3+ residuals which is always wrong
    if residual_count > 2:
        errors.append("More than two independent residual shares found.")

    awl_or_radd = structured.get("awl_or_radd")
    if fixed_sum > 1 and awl_or_radd == "لا":
        errors.append(f"Fixed shares exceed estate ({fixed_sum}) but awl_or_radd is 'لا'.")

    post_tasil = structured.get("post_tasil")
    if isinstance(post_tasil, dict):
        distribution        = _safe_list(post_tasil.get("distribution"))
        total_percent       = 0.0
        valid_percent_count = 0
        for d in distribution:
            if not isinstance(d, dict):
                continue
            p = d.get("per_head_percent")
            if isinstance(p, (int, float)):
                total_percent += float(p)
                valid_percent_count += 1
        if valid_percent_count > 0 and total_percent > 100.5:
            errors.append(f"post_tasil percentages exceed 100% ({total_percent:.2f}).")

    return {
        "valid":          len(errors) == 0,
        "errors":         errors,
        "fixed_sum":      str(fixed_sum),
        "residual_count": residual_count,
    }


def _attach_validation(
    structured: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(structured, dict):
        structured = {}
    structured["_validation"] = validation
    return structured


# ============================================================
# Post-processing enforcers
# ============================================================

def _enforce_heir_names(
    structured: Dict[str, Any],
    extracted_heirs: list,
) -> Dict[str, Any]:
    """
    Fix truncated chain names in structured output.
    e.g. LLM writes 'ابن' when it should be 'ابن ابن ابن'.
    Uses prefix matching; resolves ambiguity by excluding names
    that already appear correctly elsewhere in the output.
    """
    if not isinstance(structured, dict):
        return structured

    allowed = {
        h["heir"]: h["count"]
        for h in extracted_heirs
        if isinstance(h, dict) and "heir" in h
    }

    used_correctly: set = set()
    for field in ("heirs", "blocked", "shares"):
        for item in _safe_list(structured.get(field)):
            if isinstance(item, dict):
                name = item.get("heir")
                if isinstance(name, str) and name in allowed:
                    used_correctly.add(name)

    def _fix(name: str) -> str:
        if name in allowed:
            return name
        candidates = [k for k in allowed if k == name or k.startswith(name + " ")]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            unaccounted = [c for c in candidates if c not in used_correctly]
            if len(unaccounted) == 1:
                return unaccounted[0]
        return name

    for field in ("heirs", "blocked", "shares"):
        for item in _safe_list(structured.get(field)):
            if isinstance(item, dict) and isinstance(item.get("heir"), str):
                fixed = _fix(item["heir"])
                if fixed != item["heir"]:
                    item["heir"]  = fixed
                    item["count"] = allowed[fixed]

    for item in _safe_list((structured.get("post_tasil") or {}).get("distribution")):
        if isinstance(item, dict) and isinstance(item.get("heir"), str):
            fixed = _fix(item["heir"])
            if fixed != item["heir"]:
                item["heir"]  = fixed
                item["count"] = allowed[fixed]

    return structured


def _enforce_heirs_completeness(
    structured: Dict[str, Any],
    extracted_heirs: list,
) -> Dict[str, Any]:
    """
    Overwrite structured['heirs'] with the full extracted_heirs list.
    The LLM sometimes omits blocked heirs from the heirs field.
    """
    if not isinstance(structured, dict):
        return structured
    structured["heirs"] = [
        {"heir": h["heir"], "count": h["count"]}
        for h in extracted_heirs
        if isinstance(h, dict) and "heir" in h
    ]
    return structured


def _recompute_post_tasil_percents(structured: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically recompute per_head_percent and per_head_shares
    from the shares fractions and heir counts using exact Fraction arithmetic.
    The LLM regularly writes the group total instead of the per-person value.
    """
    if not isinstance(structured, dict):
        return structured

    post_tasil = structured.get("post_tasil")
    if not isinstance(post_tasil, dict):
        return structured

    distribution = post_tasil.get("distribution")
    if not isinstance(distribution, list):
        return structured

    share_map: Dict[str, str] = {}
    for s in _safe_list(structured.get("shares")):
        if isinstance(s, dict) and isinstance(s.get("heir"), str):
            share_map[s["heir"]] = s.get("fraction", "")

    # Compute remainder for 'باقي التركة' heirs
    known_sum = Fraction(0)
    for frac_str in share_map.values():
        if isinstance(frac_str, str) and "/" in frac_str:
            try:
                a, b = frac_str.split("/", 1)
                known_sum += Fraction(int(a.strip()), int(b.strip()))
            except Exception:
                pass
    remainder = max(Fraction(0), Fraction(1) - known_sum)

    for entry in distribution:
        if not isinstance(entry, dict):
            continue
        heir  = entry.get("heir")
        count = entry.get("count")
        if not isinstance(heir, str) or not isinstance(count, int) or count <= 0:
            continue

        frac_str = share_map.get(heir, "")
        if frac_str == "كل التركة":
            group_frac = Fraction(1)
        elif frac_str in ("باقي التركة", "باقى التركة"):
            group_frac = remainder
        elif "/" in frac_str:
            try:
                a, b = frac_str.split("/", 1)
                group_frac = Fraction(int(a.strip()), int(b.strip()))
            except Exception:
                continue
        else:
            continue

        per_head_frac             = group_frac / count
        entry["per_head_shares"]  = f"{per_head_frac.numerator}/{per_head_frac.denominator}"
        entry["per_head_percent"] = round(float(per_head_frac) * 100, 4)

    return structured



def _compute_awl_stage(structured: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute awl_stage field from shares when awl_or_radd is عول or رد.
    If the LLM already provided awl_stage with valid data, keep it.
    Otherwise compute it deterministically.
    """
    if not isinstance(structured, dict):
        return structured

    awl_or_radd = structured.get("awl_or_radd", "لا")
    if awl_or_radd not in ("عول", "رد"):
        structured["awl_stage"] = None
        return structured

    # If already present and valid, keep it
    existing = structured.get("awl_stage")
    if isinstance(existing, dict) and existing.get("asl_after_awl") and existing.get("distribution"):
        return structured

    # Compute from shares
    shares = _safe_list(structured.get("shares"))
    if not shares:
        structured["awl_stage"] = None
        return structured

    # Parse all fractions
    fracs = []
    for s in shares:
        if not isinstance(s, dict):
            continue
        f = _parse_fraction(s.get("fraction", ""))
        if f is not None:
            fracs.append((s.get("heir"), s.get("count", 1), f))

    if not fracs:
        structured["awl_stage"] = None
        return structured

    # asl_after_awl = LCM of all denominators
    from math import gcd
    def lcm(a, b):
        return a * b // gcd(a, b)

    denoms = [f.denominator for _, _, f in fracs]
    asl = denoms[0]
    for d in denoms[1:]:
        asl = lcm(asl, d)

    # For عول: sum > 1, asl stays as original denominator
    # For رد: sum < 1, asl = sum of numerators after scaling
    total_num = sum(int(f * asl) for _, _, f in fracs)
    if awl_or_radd == "عول":
        asl_after = total_num  # عول: denominator becomes sum of numerators
    else:  # رد
        asl_after = total_num  # رد: same logic, redistribute among heirs

    distribution = []
    for heir, count, f in fracs:
        numerator = int(f * asl)
        per_head_num = numerator // count if count > 0 else numerator
        distribution.append({
            "heir": heir,
            "count": count,
            "per_head_shares": f"{per_head_num}/{asl_after}",
        })

    structured["awl_stage"] = {
        "asl_after_awl": asl_after,
        "distribution": distribution,
    }
    return structured


# ============================================================
# LLM call with retry
# ============================================================

def _call_json_with_retry(
    model_key: str,
    question: str,
    reasoning: str,
    extracted_heirs: list,
) -> Dict[str, Any]:

    def _post_process(s: Dict[str, Any]) -> Dict[str, Any]:
        s = _enforce_heir_names(s, extracted_heirs)
        s = _enforce_heirs_completeness(s, extracted_heirs)
        s = _recompute_post_tasil_percents(s)
        s = _compute_awl_stage(s)
        return s

    # First attempt
    prompt     = build_structured_extraction_prompt(
        question=question,
        reasoning=reasoning,
        extracted_heirs=extracted_heirs,
    )
    structured = llm_client.call_json(model_key, prompt)
    structured = _post_process(structured)
    validation = validate_structured_answer(structured)

    if validation["valid"]:
        return _attach_validation(structured, validation)

    # One retry with validation errors fed back
    retry_prompt     = build_retry_structured_prompt(
        question=question,
        reasoning=reasoning,
        extracted_heirs=extracted_heirs,
        validation_errors=validation["errors"],
    )
    retry_structured = llm_client.call_json(model_key, retry_prompt)
    retry_structured = _post_process(retry_structured)
    retry_validation = validate_structured_answer(retry_structured)

    return _attach_validation(retry_structured, retry_validation)


# ============================================================
# Segmentation
# ============================================================

def _segment_heirs(question: str) -> list:
    """
    Split question into per-heir-type segments by splitting on ' و '
    after stripping the question prefix ('مات وترك:' etc.).
    """
    text = re.sub(r'^.*?(?:وترك[ت]?|تركت?)\s*[:\-]?\s*', '', question).strip()
    if not text:
        text = question.strip()
    segments = re.split(r'\s+و\s+', text)
    segments = [s.strip() for s in segments if s.strip()]
    return segments if segments else [question.strip()]


# ============================================================
# Main entry point
# ============================================================

def solve_case(question: str, model_key: str) -> Dict[str, Any]:
    # STEP 0: segment question into per-heir phrases
    heir_segments = _segment_heirs(question)

    # STEP 1: extract heir names + counts via LLM
    heirs_prompt    = build_heirs_extraction_prompt(heir_segments)
    heirs_json      = llm_client.call_json(model_key, heirs_prompt)
    extracted_heirs = heirs_json.get("heirs", []) if isinstance(heirs_json, dict) else []

    # STEP 2: fiqh reasoning
    reasoning_prompt = build_reasoning_prompt(question, extracted_heirs)
    reasoning        = llm_client.call_text(model_key, reasoning_prompt)

    # STEP 3: structured JSON extraction + post-processing + validation
    structured = _call_json_with_retry(
        model_key=model_key,
        question=question,
        reasoning=reasoning,
        extracted_heirs=extracted_heirs,
    )

    return {
        "question":          question,
        "heirs":             extracted_heirs,
        "reasoning":         reasoning,
        "answer_structured": structured,
    }
