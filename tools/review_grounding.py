"""Review what visitors asked Jarvis and how grounded the answers were.

    ./venv/bin/python tools/review_grounding.py            # new conversations from the live site (private dataset)
    ./venv/bin/python tools/review_grounding.py --local    # this Mac's logs/conversations instead
    ./venv/bin/python tools/review_grounding.py --all      # re-judge everything, not just new answers
    (Judging runs through the half-price batch API and usually takes a few minutes; add --now for full-price, instant.)

Each answer is judged once (results kept in reviews/judgments.jsonl) and a Markdown report is written to
reviews/report_<timestamp>.md: failures first, then every conversation. Promote a flagged answer into the test
set with evals/promote.py <id>.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grounding import MAX_WORDS, GroundingJudge, word_count  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REVIEWS = ROOT / "reviews"
JUDGMENTS = REVIEWS / "judgments.jsonl"
# Anonymous visitor IDs that are Ayaz himself (from before owner tagging existed), one per line.
OWNER_VISITORS = REVIEWS / "owner_visitors.txt"
LOG_REPO = "ayazshaik/jarvis-conversations"


def _load_lines(local: bool) -> list:
    if local:
        files = sorted((ROOT / "logs" / "conversations").glob("*.jsonl"))
    else:
        from huggingface_hub import snapshot_download
        repo = os.getenv("TWIN_LOG_REPO", LOG_REPO)
        folder = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns="data/*.jsonl")
        files = sorted(Path(folder, "data").glob("*.jsonl"))
    records = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    # test-set runs (evals/run_eval.py) aren't visitor conversations
    records = [r for r in records if not r.get("session_id", "").startswith("eval")]
    return sorted(records, key=lambda r: r["ts"])


_CACHE = {}


def _all(local: bool) -> list:
    if local not in _CACHE:
        _CACHE[local] = _load_lines(local)
    return _CACHE[local]


def load_records(local: bool) -> list:
    """Answered questions (page-view records live alongside them and are left out here)."""
    return [r for r in _all(local) if r.get("type") != "visit"]


def load_visits(local: bool) -> list:
    """Anonymous page views sent by the page's own script (see /api/visit in app.py)."""
    return [r for r in _all(local) if r.get("type") == "visit"]


def is_owner(r: dict) -> bool:
    """Ayaz testing his own twin: an owner-tagged browser, private mode, or a visitor ID listed in OWNER_VISITORS."""
    owners = OWNER_VISITORS.read_text().split() if OWNER_VISITORS.exists() else []
    return bool(r.get("owner")) or r.get("mode") == "private" or r.get("visitor") in owners


def load_judgments() -> dict:
    if not JUDGMENTS.exists():
        return {}
    return {j["id"]: j for j in map(json.loads, JUDGMENTS.read_text().splitlines()) if j}


def quote(text: str) -> str:
    return "\n".join("> " + line for line in (text or "(no answer)").splitlines()) or "> (no answer)"


def visitor_summary(visits: list, rows: list) -> list:
    """Markdown lines: real (non-owner) visits, by source, and how many went on to ask something."""
    asked = {r["session_id"] for r in rows}
    out = ["## Visitors", ""]
    if not visits:
        return out + ["No page views from other people yet.", ""]
    by_source = Counter(v.get("source", "direct") for v in visits)
    engaged = [v for v in visits if v["session_id"] in asked]
    out += [f"{len(visits)} page views from {len({v.get('visitor') for v in visits})} visitors; "
            f"{len(engaged)} went on to ask Jarvis something.", "", "| Came from | Views |", "|---|---|"]
    out += [f"| {k} | {n} |" for k, n in by_source.most_common()]
    out += ["", "| When (UTC) | Visitor | Came from | Asked? |", "|---|---|---|---|"]
    out += [f"| {v['ts'][:16]} | {v.get('visitor')} | {v.get('source')} | {'yes' if v['session_id'] in asked else 'no'} |"
            for v in sorted(visits, key=lambda v: v["ts"], reverse=True)[:30]]
    return out + [""]


def write_report(rows: list, source: str, visits: list = ()) -> Path:
    judged = [r for r in rows if r.get("judgment")]
    overall = Counter(r["judgment"]["overall"] for r in judged)
    claims = Counter(c["verdict"] for r in judged for c in r["judgment"]["claims"])
    too_long = [r for r in judged if word_count(r["answer"]) > MAX_WORDS]
    sessions = {r["session_id"] for r in rows}
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    out = [f"# Jarvis grounding review, {stamp}", "",
           f"Source: {source}. {len(rows)} answers across {len(sessions)} conversations "
           f"({len({r.get('visitor') for r in rows})} visitors).", "",
           "| Result | Count |", "|---|---|"]
    for k in ("pass", "minor", "fail"):
        out.append(f"| {k} | {overall.get(k, 0)} |")
    out.append(f"| not judged (empty or interrupted) | {len(rows) - len(judged)} |")
    out += ["", "Claims checked: " + ", ".join(f"{v} {k}" for k, v in claims.most_common()) if claims else "",
            f"Answers over {MAX_WORDS} words: {len(too_long)}", ""]

    out += visitor_summary(list(visits), rows)

    flagged = [r for r in judged if r["judgment"]["overall"] != "pass" or r in too_long]
    out += ["## Needs attention", ""] if flagged else ["## Needs attention", "", "Nothing flagged.", ""]
    for r in flagged:
        j = r["judgment"]
        out += [f"### {j['overall'].upper()}: {r['question'][:90]}",
                f"`{r['id']}` · {r['ts'][:16]} · {word_count(r['answer'])} words · prompt {r.get('prompt_version')}",
                "", quote(r["answer"]), "", f"**Why:** {j['summary']}"]
        for c in j["claims"]:
            if c["verdict"] != "supported":
                out.append(f"- **{c['verdict']}**: {c['claim']}" + (f" (KB: \"{c['evidence']}\")" if c["evidence"] else ""))
        for p in j["persona_issues"]:
            out.append(f"- persona: {p}")
        if j["off_limits"] == "answered":
            out.append("- answered an off-limits topic")
        out += ["", f"Add to the test set: `./venv/bin/python evals/promote.py {r['id']}`", ""]

    out += ["## All conversations", ""]
    for sid in dict.fromkeys(r["session_id"] for r in rows):
        convo = [r for r in rows if r["session_id"] == sid]
        out.append(f"### {convo[0]['ts'][:16]} UTC · visitor {convo[0].get('visitor')} · {len(convo)} question{'s' if len(convo) != 1 else ''}")
        for r in convo:
            mark = r["judgment"]["overall"] if r.get("judgment") else ("interrupted" if not r.get("completed") else "-")
            out += [f"- **Q:** {r['question']}", f"  **A** ({mark}): {r['answer'] or '(none)'}"]
        out.append("")

    REVIEWS.mkdir(exist_ok=True)
    path = REVIEWS / f"report_{datetime.now():%Y%m%d_%H%M}.md"
    path.write_text("\n".join(out))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="read this Mac's logs instead of the live dataset")
    ap.add_argument("--all", action="store_true", help="re-judge answers that were already judged")
    ap.add_argument("--include-owner", action="store_true", help="also review Ayaz's own test conversations")
    ap.add_argument("--now", action="store_true", help="judge immediately at full price instead of the half-price batch")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    records = load_records(args.local)
    visits = load_visits(args.local)
    if not args.include_owner:
        visits = [v for v in visits if not is_owner(v)]
        mine = [r for r in records if is_owner(r)]
        records = [r for r in records if not is_owner(r)]
        if mine:
            print(f"Left out {len(mine)} of your own test answers (use --include-owner to keep them).")
    judgments = {} if args.all else load_judgments()
    todo = [r for r in records if r["id"] not in judgments and r.get("answer") and r.get("completed")]
    print(f"{len(records)} answers found, {len(todo)} to judge.")

    if todo:
        judge = GroundingJudge()
        REVIEWS.mkdir(exist_ok=True)
        if args.now:
            results = {r["id"]: judge.judge(r["question"], r["answer"], r.get("context", [])) for r in todo}
        else:  # not urgent, so half price via the batch API
            results = judge.judge_batch([(r["id"], r["question"], r["answer"], r.get("context", [])) for r in todo])
        with JUDGMENTS.open("w" if args.all else "a") as f:
            for r in todo:
                if r["id"] not in results:
                    continue
                verdict = results[r["id"]].model_dump()
                judgments[r["id"]] = {"id": r["id"], "judged_at": datetime.now().isoformat(timespec="seconds"),
                                      "judge_prompt_version": judge.version, **verdict}
                f.write(json.dumps(judgments[r["id"]], ensure_ascii=False) + "\n")
                print(f"  {verdict['overall']:5} {r['question'][:70]}")

    rows = [{**r, "judgment": judgments.get(r["id"])} for r in records]
    path = write_report(rows, "local logs" if args.local else "live site", visits)
    print(f"Visitors: {len(visits)} page views from other people, "
          f"{len({v['session_id'] for v in visits} & {r['session_id'] for r in records})} of them asked something.")
    print(f"Report: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
