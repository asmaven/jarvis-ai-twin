"""Run Jarvis's test set against a running server before anything ships.

Usage (server must be running on :8010):
    ./venv/bin/python evals/run_eval.py                # string checks + grounding judge + length
    ./venv/bin/python evals/run_eval.py --no-judge     # string checks + length only (free, no judge calls)
    ./venv/bin/python evals/run_eval.py --only p01,a05 # just these cases

Each case in questions.jsonl passes only if all of its checks pass:
  - must_include_any: list of groups; each group passes if ANY of its phrases appears (case-insensitive)
  - must_not_include: fails if any of these phrases appears
  - must_not_match: fails if any of these regexes matches (case-insensitive)
  - length: the answer must be MAX_WORDS words or fewer (persona.md: 2 lines)
  - grounding judge (tools/grounding.py): fails on any invented, contradicted, or misused claim, or an
    off-limits topic answered; "minor" persona notes are reported but don't fail the case
questions.jsonl is private (gitignored, it describes Ayaz's career); questions.example.jsonl shows the format and
is used when the private file isn't there.
Cases may carry "context" (earlier turns) to test follow-up questions; evals/promote.py adds visitor questions.

Results go to evals/results_<timestamp>.jsonl, and evals/latest.json records the prompt version that was tested
and whether every case passed. deploy/huggingface/push.sh refuses to upload unless that run passed.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
from grounding import MAX_WORDS, GroundingJudge, load_sources, prompt_version, word_count  # noqa: E402


def score(case: dict, answer: str) -> list:
    text = answer.lower()
    problems = []
    for group in case.get("must_include_any", []):
        if not any(p.lower() in text for p in group):
            problems.append(f"missing one of {group}")
    for phrase in case.get("must_not_include", []):
        if phrase.lower() in text:
            problems.append(f"contains forbidden '{phrase}'")
    for pattern in case.get("must_not_match", []):
        if re.search(pattern, answer, re.IGNORECASE):
            problems.append(f"matches forbidden /{pattern}/")
    if word_count(answer) > MAX_WORDS:
        problems.append(f"too long ({word_count(answer)} words > {MAX_WORDS})")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010")
    ap.add_argument("--no-judge", action="store_true", help="skip the grounding judge (no extra model calls)")
    ap.add_argument("--only", help="comma-separated case ids")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    path = HERE / "questions.jsonl"  # private; the public repo ships questions.example.jsonl
    if not path.exists():
        path = HERE / "questions.example.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    version = prompt_version(*load_sources())
    judge = None if args.no_judge else GroundingJudge()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = HERE / f"results_{stamp}.jsonl"
    passed, latencies, minors = 0, [], 0

    with httpx.Client(timeout=120) as client, out_path.open("w") as out:
        for case in cases:
            messages = case.get("context", []) + [{"role": "user", "content": case["question"]}]
            start = time.perf_counter()
            # session ids starting with "eval" are left out of the visitor review
            resp = client.post(f"{args.base_url}/api/chat", json={"messages": messages, "session_id": f"eval-{stamp}"})
            answer = resp.text if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text}"
            elapsed = time.perf_counter() - start
            problems = score(case, answer)
            verdict = None
            if judge:
                verdict = judge.judge(case["question"], answer, case.get("context", [])).model_dump()
                if verdict["overall"] == "fail":
                    problems.append(f"judge: {verdict['summary']}")
                minors += verdict["overall"] == "minor"
            ok = not problems
            passed += ok
            latencies.append(elapsed)
            out.write(json.dumps({**case, "answer": answer, "passed": ok, "problems": problems,
                                  "judgment": verdict, "seconds": round(elapsed, 2)}, ensure_ascii=False) + "\n")
            note = "; ".join(problems) or (f"minor: {verdict['summary']}" if verdict and verdict["overall"] == "minor" else "")
            print(f"{'PASS' if ok else 'FAIL'} {case['id']} ({elapsed:.1f}s, {word_count(answer)}w) {note}")

    total = len(cases)
    latencies.sort()
    print(f"\nPassed: {passed}/{total} = {passed / total:.0%}" + (f"  ({minors} with minor persona notes)" if judge else ""))
    print(f"Latency (total per answer): median {latencies[total // 2]:.1f}s, max {latencies[-1]:.1f}s")
    print(f"Prompt version tested: {version}")
    print(f"Details: {out_path.relative_to(ROOT)}")

    if not args.only:  # only a full run can clear a deploy
        (HERE / "latest.json").write_text(json.dumps({
            "prompt_version": version, "passed": passed, "total": total, "all_passed": passed == total,
            "judged": judge is not None, "results": out_path.name, "ran_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2) + "\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
