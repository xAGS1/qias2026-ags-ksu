"""
fix_submission.py
=================
Post-processing script for GPT-5.4 Thinking predictions.
Cleans and validates prediction.json before  submission.

Usage:
    python fix_submission.py --input raw_output.json --output prediction.json
"""
import json
import re
import argparse
from pathlib import Path


def clean_heir_name(name: str) -> str:
    """Remove parenthetical count annotations e.g. 'زوجة (العدد 3)' -> 'زوجة'"""
    if not isinstance(name, str):
        return name
    return re.sub(r'\s*\(العدد\s*\d+\)', '', name).strip()


def fix_entry(entry: dict) -> dict:
    out = entry.get("output", {})
    if not isinstance(out, dict):
        return entry

    # Fix 1: Remove (العدد X) from all heir name fields
    for field in ["heirs", "blocked", "shares"]:
        for item in out.get(field, []):
            if isinstance(item, dict) and "heir" in item:
                item["heir"] = clean_heir_name(item["heir"])

    # Fix post_tasil distribution heir names
    for item in out.get("post_tasil", {}).get("distribution", []):
        if isinstance(item, dict) and "heir" in item:
            item["heir"] = clean_heir_name(item["heir"])

    # Fix 2: Ensure required fields are present
    if "awl_or_radd" not in out:
        out["awl_or_radd"] = None
    if "awl_stage" not in out:
        out["awl_stage"] = None
    if "heirs" not in out:
        out["heirs"] = []
    if "blocked" not in out:
        out["blocked"] = []
    if "shares" not in out:
        out["shares"] = []
    if "post_tasil" not in out:
        out["post_tasil"] = {"total_shares": 0, "distribution": []}

    # Fix 3: Validate all heirs appear in post_tasil distribution
    heir_names = {clean_heir_name(h.get("heir", "")) for h in out.get("heirs", [])
                  if isinstance(h, dict)}
    dist_names = {clean_heir_name(d.get("heir", "")) for d in
                  out.get("post_tasil", {}).get("distribution", [])
                  if isinstance(d, dict)}
    missing = heir_names - dist_names
    if missing:
        print(f"  Warning [{entry.get('id')}]: heirs not in post_tasil: {missing}")

    # Fix 4: Verify fraction strings follow N/D format
    for item in out.get("shares", []):
        if isinstance(item, dict) and "fraction" in item:
            f = item["fraction"]
            if isinstance(f, str) and f not in ("null", "") and "/" not in f:
                print(f"  Warning [{entry.get('id')}]: fraction '{f}' may not be N/D format")

    entry["output"] = out
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="raw_output.json", help="Input prediction file")
    parser.add_argument("--output", default="prediction.json",  help="Output cleaned file")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} entries from {args.input}")

    fixed = [fix_entry(entry) for entry in data]

    # Write clean submission file
    submission = [{"id": e["id"], "question": e["question"], "output": e["output"]}
                  for e in fixed]

    Path(args.output).write_text(
        json.dumps(submission, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(submission)} entries to {args.output}")
    print("Done. Ready for CodaBench submission.")


if __name__ == "__main__":
    main()
