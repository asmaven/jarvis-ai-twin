"""Daily review: judge new visitor answers and write a short summary of what went wrong.

    ./venv/bin/python tools/daily_review.py

Runs tools/review_grounding.py (half-price batch judge), then looks only at what's new since the last run
(reviews/daily_state.json). Answers rated minor or fail are listed for Ayaz to read. Visitor questions never
become tests on their own: the test set is a hand-picked golden set (evals/questions.jsonl), and a fix is
checked against it before it ships.

Prints "NOTHING_NEW" when no visitor asked anything since the last run, otherwise "SUMMARY: <path>" for the
launchd wrapper (kept outside this repo, next to the private rules) to show as a notification.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_grounding import REVIEWS, ROOT, is_owner, load_judgments, load_records, load_visits  # noqa: E402

STATE = REVIEWS / "daily_state.json"
CENTRAL = ZoneInfo("America/Chicago")


def central(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(CENTRAL).strftime("%b %d, %-I:%M %p")


def main():
    subprocess.run([sys.executable, str(HERE / "review_grounding.py")], check=True)

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    since = state.get("last_run", "1970-01-01T00:00:00+00:00")
    now = datetime.now(timezone.utc).isoformat()

    records = [r for r in load_records(False) if not is_owner(r) and r["ts"] > since]
    visits = [v for v in load_visits(False) if not is_owner(v) and v["ts"] > since]
    judgments = load_judgments()
    flagged = [(r, judgments[r["id"]]) for r in records
               if judgments.get(r["id"]) and judgments[r["id"]]["overall"] != "pass"]

    REVIEWS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"last_run": now}))
    if not records:
        print(f"NOTHING_NEW ({len(visits)} page views, no questions since {since[:16]} UTC)")
        return

    counts = {k: sum(1 for r in records if (judgments.get(r["id"]) or {}).get("overall") == k)
              for k in ("pass", "minor", "fail")}
    sessions = list(dict.fromkeys(r["session_id"] for r in records))
    out = [f"Jarvis daily review, {datetime.now(CENTRAL):%a %b %d}", "",
           f"{len(records)} new visitor answers in {len(sessions)} conversations; "
           f"{len(visits)} page views since the last review.",
           f"Judge: {counts['pass']} pass, {counts['minor']} minor, {counts['fail']} fail.", ""]
    if flagged:
        out += [f"Needs a look ({len(flagged)}):", ""]
        out += [f"- [{j['overall']}] \"{r['question']}\": {j['summary']}" for r, j in flagged]
        out.append("")
    out += ["Conversations (Central time):", ""]
    for sid in sessions:
        convo = [r for r in records if r["session_id"] == sid]
        out.append(f"{central(convo[0]['ts'])}, visitor {convo[0].get('visitor')}:")
        for r in convo:
            verdict = (judgments.get(r["id"]) or {}).get("overall", "not judged")
            out += [f"  Q: {r['question']}", f"  A ({verdict}): {r.get('answer') or '(none)'}"]
        out.append("")
    out.append("Ask in Claude Code for a fix if a pattern needs one; it ships only after the golden set passes.")

    path = REVIEWS / f"daily_{datetime.now(CENTRAL):%Y%m%d}.txt"
    path.write_text("\n".join(out))
    print(f"SUMMARY: {path}")


if __name__ == "__main__":
    main()
