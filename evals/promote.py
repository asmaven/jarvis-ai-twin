"""Turn a visitor's question (flagged in a grounding review) into a test case, so the fix stays fixed.

    ./venv/bin/python evals/promote.py <answer-id>                       # judge-only case
    ./venv/bin/python evals/promote.py <answer-id> --include "Maven" --exclude "Amazon product"

The id comes from a reviews/report_*.md entry. The case keeps any earlier turns of that conversation, so
follow-up questions are tested in context. Every case is also checked by the grounding judge and the length
limit; --include/--exclude add exact-phrase checks (repeat them as needed, case-insensitive).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
from review_grounding import load_judgments, load_records  # noqa: E402

QUESTIONS = HERE / "questions.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answer_id")
    ap.add_argument("--include", action="append", default=[], help="phrase the answer must contain")
    ap.add_argument("--exclude", action="append", default=[], help="phrase the answer must not contain")
    ap.add_argument("--local", action="store_true", help="look in this Mac's logs instead of the live dataset")
    args = ap.parse_args()

    record = next((r for r in load_records(args.local) if r["id"] == args.answer_id), None)
    if not record:
        sys.exit(f"No answer with id {args.answer_id} (try --local for answers from this Mac).")
    judgment = load_judgments().get(args.answer_id, {})

    cases = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    if any(c.get("source_id") == args.answer_id for c in cases):
        sys.exit("That answer is already in the test set.")
    n = 1 + max([int(c["id"][1:]) for c in cases if c["id"].startswith("p")] or [0])
    case = {
        "id": f"p{n:02d}",
        "type": "decline" if judgment.get("off_limits", "n/a") != "n/a" else "answer",
        "question": record["question"],
        "source_id": args.answer_id,
        "note": f"from a visitor on {record['ts'][:10]}" + (f"; review said: {judgment['summary']}" if judgment else ""),
    }
    if record.get("context"):
        case["context"] = record["context"]
    if args.include:
        case["must_include_any"] = [[p] for p in args.include]
    if args.exclude:
        case["must_not_include"] = args.exclude

    with QUESTIONS.open("a") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Added {case['id']}: {case['question']}")
    print(f"Run it: ./venv/bin/python evals/run_eval.py --only {case['id']}")


if __name__ == "__main__":
    main()
