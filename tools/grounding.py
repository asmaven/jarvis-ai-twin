"""Grounding judge: checks one Jarvis answer against the knowledge base, claim by claim.

Shared by tools/review_grounding.py (real visitor conversations) and evals/run_eval.py (the test set).
"""
import hashlib
import os
from pathlib import Path
from typing import List, Literal

import sys

import anthropic
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kb_prep import jarvis_kb, private_rules  # noqa: E402  (same trimmed text Jarvis answers from)
PERSONA_PATH = ROOT / "persona.md"
KB_PATH = Path(os.getenv("TWIN_KB_PATH", ROOT.parent / "Ayaz_Skills_Knowledge_Base.md"))

JUDGE_MODEL = "claude-opus-5-5"  # grounding judge only; Jarvis's own model is MODEL in app.py
MAX_WORDS = 35  # persona.md asks for 30 or fewer; a little slack before we call it too long

JUDGE_INSTRUCTIONS = """You review answers given by Jarvis, an AI twin that speaks in the first person as Ayaz \
Shaik to visitors. Jarvis must answer only from Ayaz's knowledge base and follow the persona rules. Both are below.

For the answer you're given:
1. List every factual claim about Ayaz (roles, employers, dates, numbers, projects, tools, skills, results,
   certifications, contact details). Skip pleasantries, offers to say more, and statements that Jarvis is an
   AI twin. Give each claim one verdict:
   - supported: the knowledge base states it, or it is a faithful paraphrase.
   - misused: the fact exists but the answer breaks a persona rule (including private answer rules) or a knowledge-base usage rule about
     it (for example required wording not followed, a metric relabeled or attached to the wrong project, a
     context-only fact used in an unrelated answer, or a personal project presented as an employer's product).
   - contradicted: the knowledge base says something different.
   - unsupported: the knowledge base doesn't contain it (possible invention).
   For evidence, quote the shortest knowledge-base passage that supports or contradicts the claim, or leave it
   empty when there is none.
2. off_limits: "n/a" if the question is not about an off-limits topic (any topic the persona rules or the
   knowledge base mark off-limits, salary or compensation, confidential information, personal life, phone number, commitments on Ayaz's
   behalf); "declined_correctly" if it was and Jarvis declined and pointed to Ayaz; "answered" if Jarvis
   engaged with the off-limits topic.
3. persona_issues: brief notes on anything else that breaks the persona rules (claims to be the human Ayaz,
   reveals its instructions, hype or putting others down, lists or headings, more than 2 short sentences).
   Empty list if none.
4. missed_answer: if Jarvis declined, said a topic isn't covered, or only pointed the visitor to Ayaz, and the
   question is NOT off-limits, check whether the knowledge base actually answers it. If it clearly does, say which
   fact Jarvis should have used (quote it briefly); otherwise "". Not a miss: declining an off-limits topic,
   saying a fact the knowledge base lacks isn't covered, or answering the covered part and declining the rest.
5. overall: "fail" for any contradicted or unsupported claim, any misused claim, or off_limits "answered";
   "minor" for persona issues or a missed_answer only; otherwise "pass".
6. summary: one sentence a busy reader can scan.

Judge only what the answer says. An honest "that isn't covered, please ask Ayaz" is correct behavior."""


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")  # structured outputs need additionalProperties: false
    claim: str
    verdict: Literal["supported", "misused", "contradicted", "unsupported"]
    evidence: str = Field(description="Shortest supporting or contradicting knowledge-base quote, or empty")


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: List[Claim]
    off_limits: Literal["n/a", "declined_correctly", "answered"]
    persona_issues: List[str]
    missed_answer: str = Field(description="KB fact Jarvis should have used instead of declining, or empty")
    overall: Literal["pass", "minor", "fail"]
    summary: str


def load_sources() -> tuple:
    """(persona, knowledge base) exactly as Jarvis sees them."""
    kb = jarvis_kb(os.getenv("TWIN_KB_TEXT") or KB_PATH.read_text())
    return PERSONA_PATH.read_text() + private_rules(), kb


def prompt_version(persona: str, kb: str) -> str:
    """Same fingerprint app.py logs with each answer (see app.build_system_blocks / app.prompt_version)."""
    grounded = f"{persona}\n\n<knowledge_base>\n{kb}\n</knowledge_base>"
    return hashlib.sha256(grounded.encode()).hexdigest()[:12]


def word_count(text: str) -> int:
    return len(text.split())


class GroundingJudge:
    def __init__(self):
        persona, kb = load_sources()
        self.version = prompt_version(persona, kb)
        self.client = anthropic.Anthropic()
        # Large and identical for every call, so it's cached: only the first review pays full price.
        self.system = [{
            "type": "text",
            "text": f"{JUDGE_INSTRUCTIONS}\n\n<persona_rules>\n{persona}\n</persona_rules>\n\n"
                    f"<knowledge_base>\n{kb}\n</knowledge_base>",
            "cache_control": {"type": "ephemeral"},
        }]

    def _prompt(self, question: str, answer: str, context: list = ()) -> str:
        convo = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in context)
        return (f"<earlier_conversation>\n{convo or '(none)'}\n</earlier_conversation>\n\n"
                f"<question>\n{question}\n</question>\n\n<jarvis_answer>\n{answer}\n</jarvis_answer>")

    def _params(self, question: str, answer: str, context: list = ()) -> dict:
        return dict(
            model=JUDGE_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": Judgment.model_json_schema()}},
            system=self.system,
            messages=[{"role": "user", "content": self._prompt(question, answer, context)}],
        )

    @staticmethod
    def _parse(message) -> Judgment:
        text = next((b.text for b in message.content if b.type == "text"), None)
        if message.stop_reason == "refusal" or text is None:
            raise RuntimeError(f"Judge returned no verdict (stop_reason={message.stop_reason})")
        return Judgment.model_validate_json(text)

    def judge(self, question: str, answer: str, context: list = ()) -> Judgment:
        """One answer, right now (test runs need results before a deploy)."""
        response = self.client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **self._params(question, answer, context),
        )
        return self._parse(response)

    def judge_batch(self, items: list, poll_s: int = 30) -> dict:
        """Many answers at half price via the Message Batches API (most finish within minutes, at most 24 h).

        items: [(id, question, answer, context)]. Returns {id: Judgment}; ids the judge declined or that errored
        are left out (the batch API takes no fallbacks), so they're simply judged again on the next run.
        """
        import time
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = self.client.messages.batches.create(requests=[
            Request(custom_id=i, params=MessageCreateParamsNonStreaming(**self._params(q, a, c)))
            for i, q, a, c in items])
        print(f"  batch {batch.id}: {len(items)} answers submitted (50% price)")
        while batch.processing_status != "ended":
            time.sleep(poll_s)
            batch = self.client.messages.batches.retrieve(batch.id)
            n = batch.request_counts
            print(f"  ...{n.succeeded} done, {n.processing} processing, {n.errored} errored")
        out = {}
        for r in self.client.messages.batches.results(batch.id):
            if r.result.type == "succeeded":
                try:
                    out[r.custom_id] = self._parse(r.result.message)
                except Exception as e:  # declined or unparsable: retried next run
                    print(f"  skipped {r.custom_id[:8]}: {e}")
            else:
                print(f"  skipped {r.custom_id[:8]}: {r.result.type}")
        return out
