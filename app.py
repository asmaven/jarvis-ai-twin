"""My AI Twin — a grounded, voice-capable chat twin of Ayaz, served by FastAPI.

Run:  ./venv/bin/uvicorn app:app --reload --port 8010
"""
import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import time
import uuid
import wave
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kb_prep import jarvis_kb, private_rules

load_dotenv()

ROOT = Path(__file__).parent
KB_PATH = Path(os.getenv("TWIN_KB_PATH", ROOT.parent / "Ayaz_Skills_Knowledge_Base.md"))
PERSONA_PATH = ROOT / "persona.md"
LOG_DIR = Path(os.getenv("TWIN_LOG_DIR", ROOT / "logs"))
# Every question and answer is saved here as one JSON line, one file per server run (so restarts never
# overwrite earlier conversations). With TWIN_LOG_REPO set, the folder is also synced to that private
# Hugging Face dataset every few minutes; see tools/review_grounding.py for reading it back.
CONVO_DIR = LOG_DIR / "conversations"
CONVO_PATH = CONVO_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}.jsonl"
# Page views go to their own file in the same folder (and dataset), so they never mix with answers.
VISITS_PATH = CONVO_PATH.with_name("visits_" + CONVO_PATH.name)
CONVO_REPO = os.getenv("TWIN_LOG_REPO", "")  # e.g. ayazshaik/jarvis-conversations
# Visitors are stored as a salted hash, never a raw IP. Set TWIN_LOG_SALT to keep the same visitor ID across
# restarts; otherwise IDs only link visits within one server run.
LOG_SALT = os.getenv("TWIN_LOG_SALT") or secrets.token_hex(16)

MODEL = "claude-opus-5"  # Jarvis's answers. (The grounding judge in tools/grounding.py has its own JUDGE_MODEL.)
# Opus 5.5 was trialed 2026-10-01: no cheaper per visit (it thinks more), slower, and declined "biggest weakness".
MAX_TOKENS = 200        # backstop; persona.md keeps answers to 2 lines
MAX_TURNS = 20            # history kept per request
MAX_USER_CHARS = 2000     # per message
RATE_LIMIT = 30           # requests per IP ...
RATE_WINDOW_S = 600       # ... per 10 minutes
PRIVATE_PASSCODE = os.getenv("TWIN_PRIVATE_PASSCODE", "")
# Link-only access: when set, the page and APIs need ?key=<this> (the shared link). Unset = open (local use).
ACCESS_KEY = os.getenv("TWIN_ACCESS_KEY", "")
# The knowledge base can come from a secret instead of a file, so it isn't visible in a public host's files.
KB_TEXT = os.getenv("TWIN_KB_TEXT", "")
VOICE_PATH = Path(os.getenv("TWIN_VOICE_PATH", ROOT / "voices" / "en_US-hfc_male-medium.onnx"))
MAX_TTS_CHARS = 600       # per sentence/chunk sent to /api/tts

REFUSAL_TEXT = "Sorry, I can't help with that one. Please reach out to Ayaz directly on LinkedIn."

MODE_INSTRUCTIONS = {
    "public": (
        "MODE: PUBLIC. You are talking with a visitor who wants to learn about Ayaz's work. "
        "Answer questions about Ayaz's background as his twin."
    ),
    "private": (
        "MODE: PRIVATE. You are talking with Ayaz himself. Help him draft replies to messages, practice "
        "interview answers in his voice, and check that his answers match the knowledge base. All grounding "
        "rules and off-limits topics still apply."
    ),
}


def prompt_version(system: list[dict]) -> str:
    """Short fingerprint of the persona + knowledge base, so each logged answer names the prompt it came from."""
    return hashlib.sha256(system[0]["text"].encode()).hexdigest()[:12]


def build_system_blocks(mode: str) -> list[dict]:
    """Stable persona + knowledge base first (cached); the mode instruction after the breakpoint."""
    persona = PERSONA_PATH.read_text() + private_rules()
    kb = jarvis_kb(KB_TEXT or KB_PATH.read_text())  # resume-only notes trimmed (kb_prep.py)
    grounded = f"{persona}\n\n<knowledge_base>\n{kb}\n</knowledge_base>"
    return [
        {"type": "text", "text": grounded, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": MODE_INSTRUCTIONS[mode]},
    ]


def _start_log_sync():
    if not CONVO_REPO:
        return None
    try:
        from huggingface_hub import CommitScheduler
        return CommitScheduler(repo_id=CONVO_REPO, repo_type="dataset", private=True, folder_path=CONVO_DIR,
                               path_in_repo="data", every=5)
    except Exception:  # a sync problem must never take the site down; logs still land on local disk
        logging.exception("Conversation sync to %s is off", CONVO_REPO)
        return None


CONVO_DIR.mkdir(parents=True, exist_ok=True)
_log_sync = _start_log_sync()
client = anthropic.AsyncAnthropic()
app = FastAPI(title="My AI Twin")
_requests_by_ip: dict[str, deque] = defaultdict(deque)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    mode: str = Field(default="public", pattern="^(public|private)$")
    passcode: str = ""
    session_id: str = ""
    owner: bool = False  # set by Ayaz's own browsers (see index.html); only used to leave his tests out of reviews


class VisitRequest(BaseModel):
    session_id: str = ""
    owner: bool = False
    referrer: str = Field(default="", max_length=500)


def _visit_source(referrer: str) -> str:
    """Where a visitor came from, by referring site only (never the full URL, which can carry personal data)."""
    from urllib.parse import urlparse
    host = (urlparse(referrer).hostname or "").lower()
    if not host or "jarvis.hf.space" in host:
        return "direct"
    for needle, label in (("linkedin", "linkedin"), ("mail", "email"), ("outlook", "email"), ("google", "search"),
                          ("bing", "search"), ("duckduckgo", "search"), ("huggingface", "huggingface"), ("github", "github")):
        if needle in host:
            return label
    return host[:80]


def _client_ip(request: Request) -> str:
    """The visitor's real IP, even behind a host's proxy or a Cloudflare tunnel.

    Cloudflare sets CF-Connecting-IP itself. Otherwise use the rightmost X-Forwarded-For entry: it's the one the
    host's proxy added, so a visitor can't fake it to dodge the rate limit.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _has_access(key: str) -> bool:
    return not ACCESS_KEY or hmac.compare_digest(key.encode(), ACCESS_KEY.encode())


def _require_access(request: Request) -> None:
    if not _has_access(request.headers.get("x-twin-key", "")):
        raise HTTPException(403, "This demo is invite-only. Please use the full link you were given.")


INVITE_ONLY_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Jarvis - Ayaz's AI Twin</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#1b2a41;color:#fff;
font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;text-align:center;padding:16px}
p{color:#b6c1d1;margin:6px 0 0}a{color:#fff}</style></head><body><div>
<h1 style="font-size:1.4rem;margin:0">Jarvis - Ayaz's AI Twin</h1>
<p>This demo is invite-only. Please open the full link you were given.</p>
<p>To reach Ayaz, visit <a href="https://www.linkedin.com/in/ayazshaik">LinkedIn</a>.</p>
</div></body></html>"""


def _check_rate_limit(ip: str) -> None:
    if ip in ("127.0.0.1", "::1"):  # this machine itself (local use, test runs); hosted visitors never appear as this
        return
    now = time.monotonic()
    hits = _requests_by_ip[ip]
    while hits and now - hits[0] > RATE_WINDOW_S:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Too many questions in a short time. Please try again in a few minutes.")
    hits.append(now)


def _log_turn(record: dict, path: Path = CONVO_PATH) -> None:
    # Logging must never break an answer that already streamed to the visitor.
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        if _log_sync:
            with _log_sync.lock:  # don't let a sync upload read a half-written line
                with path.open("a") as f:
                    f.write(line)
        else:
            with path.open("a") as f:
                f.write(line)
    except OSError as e:
        logging.warning("Could not write chat log: %s", e)


_voice = None


def _get_voice():
    """Load the Piper voice once, on first use."""
    global _voice
    if _voice is None:
        from piper import PiperVoice
        _voice = PiperVoice.load(str(VOICE_PATH))
    return _voice


_SPEECH_REWRITES = [
    (r"\*\*|__|`|#+\s", ""),
    (r"(?m)^\s*[-•]\s+", ""),
    (r"\$(\d+(?:\.\d+)?)\s?MM\+?", r"\1 million dollars plus"),
    (r"\$(\d+(?:\.\d+)?)\s?K\+?", r"\1 thousand dollars"),
    (r"(\d+)\+", r"\1 plus"),
    (r"\bYoY\b", "year over year"),
    (r"≥", "at least "),
    (r"(?i)\be\.g\.", "for example"),
    (r"(?i)\bi\.e\.", "that is"),
    (r"(?i)\bn8n\b", "n eight n"),
    (r"\bSr\.", "Senior"),
    (r"(?i)\bayaz\b", "Ah-yahz"),  # say AH-YAHZ, not EYE-az (also inside the email address)
    (r"(?i)https?://\S+|\b(?:www\.)?linkedin\.com/in/\S+", "my LinkedIn profile"),
    (r"\s[—–]\s|—", ", "),
    (r"\s{2,}", " "),
]


def for_speech(text: str) -> str:
    """Rewrite symbols and markdown so they sound natural when spoken."""
    for pattern, repl in _SPEECH_REWRITES:
        text = re.sub(pattern, repl, text)
    return text.strip()


# Mouth shapes (visemes) the avatar can show, keyed by one letter:
#   R rest/closed · M lips pressed (m, b, p) · A wide open (ah) · O rounded (oo, w)
#   E stretched (ee, s, sh) · F teeth on lip (f, v) · H half open (most other sounds)
_VISEME_OF = {}
for chars, code in (("pbm", "M"), ("fv", "F"), ("uʊoɔwɒ", "O"), ("aɑæʌɐ", "A"),
                    ("iɪeɛjszʃʒᵻɨ", "E"), ("tdnlkgɡŋhɹrɾθðəɚɜxʔ", "H")):
    for ch in chars:
        _VISEME_OF[ch] = code
_PAUSES = set(",;:.!?^$_")  # punctuation and sentence start/end: mouth at rest


def _viseme_timeline(chunks) -> list[list]:
    """Mouth shapes at the exact times Piper speaks each phoneme (from the voice's alignment output).
    Returns [[start_ms, code], ...]."""
    segments, t = [], 0.0  # [start_ms, end_ms, code]
    stress_start = None    # stress marks take time of their own; it belongs to the sound that follows
    for chunk in chunks:
        for al in chunk.phoneme_alignments or []:
            start, t = t, t + al.num_samples * 1000 / chunk.sample_rate
            ph = al.phoneme
            if ph in "ˈˌ":
                stress_start = start if stress_start is None else stress_start
                continue
            if stress_start is not None:
                start, stress_start = stress_start, None
            if (ph == "ː" or (ph == " " and t - start < 120)) and segments:
                segments[-1][1] = t  # long vowel or a normal gap between words: hold the current shape
                continue
            code = "R" if ph in _PAUSES or ph == " " else _VISEME_OF.get(ph, "H")
            if segments and segments[-1][2] == code:
                segments[-1][1] = t
            else:
                segments.append([start, t, code])
    if not segments:
        return []
    return [[int(s), code] for s, _, code in _smooth_segments(segments)] + [[int(t), "R"]]


MIN_SHAPE_MS = 80   # real mouths change ~8-10 times a second; faster than this looks jittery
MIN_PRESS_MS = 70    # a quick m/b/p lip closure still reads as speech


_SHAPE_PRIORITY = {"A": 3, "O": 3, "M": 2, "E": 1, "F": 1, "H": 1, "R": 0}  # how visible each shape is to a viewer


def _smooth_segments(segments: list[list]) -> list[list]:
    """Merge mouth shapes that are too short to see. The more visible shape wins: open vowels (ah, oo), then
    lip closures (m, b, p), then everything else, then rest."""
    def min_ms(seg):
        return MIN_PRESS_MS if seg[2] == "M" else MIN_SHAPE_MS

    def dur(seg):
        return seg[1] - seg[0]

    def rank(seg):
        return _SHAPE_PRIORITY[seg[2]]

    while len(segments) > 1:
        short = [i for i, seg in enumerate(segments) if dur(seg) < min_ms(seg) - 0.5]
        if not short:
            break
        i = min(short, key=lambda j: dur(segments[j]))
        seg, need = segments[i], min_ms(segments[i]) - dur(segments[i])
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(segments)]
        # 1) Borrow time from a less visible neighbour that can spare it.
        donors = [j for j in neighbours if rank(segments[j]) <= rank(seg) and dur(segments[j]) - need >= min_ms(segments[j])]
        if donors:
            j = max(donors, key=lambda n: dur(segments[n]))
            if j < i:
                segments[j][1] -= need
                seg[0] -= need
            else:
                segments[j][0] += need
                seg[1] += need
            continue
        # 2) Otherwise merge with a neighbour; the more visible (then longer) shape takes over the pair.
        j = max(neighbours, key=lambda n: (rank(segments[n]), dur(segments[n])))
        if rank(segments[j]) < rank(seg):
            i, j = j, i
        keep, drop = segments[j], segments[i]
        keep[0], keep[1] = min(keep[0], drop[0]), max(keep[1], drop[1])
        del segments[i]
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg[2] == merged[-1][2]:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        segments = merged
    return segments


def _synthesize_wav(text: str) -> tuple[bytes, list[list]]:
    chunks = list(_get_voice().synthesize(text, include_alignments=True))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setframerate(chunks[0].sample_rate if chunks else 22050)
        wf.setsampwidth(2)
        wf.setnchannels(1)
        for chunk in chunks:
            wf.writeframes(chunk.audio_int16_bytes)
    return buf.getvalue(), _viseme_timeline(chunks)


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def tts(req: TTSRequest, request: Request):
    """Turn one sentence of the twin's answer into natural speech (WAV)."""
    _require_access(request)
    text = for_speech(req.text)[:MAX_TTS_CHARS]
    if not text:
        raise HTTPException(400, "Nothing to say.")
    if not VOICE_PATH.exists():
        raise HTTPException(503, "Voice model not installed.")
    audio, visemes = await asyncio.to_thread(_synthesize_wav, text)
    # The mouth-shape timeline rides along in a header so the page can lip-sync to this exact clip.
    return Response(content=audio, media_type="audio/wav",
                    headers={"X-Visemes": json.dumps(visemes, separators=(",", ":"))})


@app.post("/api/visit")
async def visit(req: VisitRequest, request: Request):
    """One anonymous page view, sent by the page's own script, so link previews and most bots never count."""
    _require_access(request)
    ip = _client_ip(request)
    _check_rate_limit("visit:" + ip)  # separate bucket from chat questions
    _log_turn({
        "id": uuid.uuid4().hex,
        "type": "visit",
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": req.session_id[:64],
        "visitor": hashlib.sha256(f"{LOG_SALT}:{ip}".encode()).hexdigest()[:12],
        "owner": req.owner,
        "source": _visit_source(req.referrer),
    }, VISITS_PATH)
    return {"ok": True}


class EventRequest(BaseModel):
    name: str = Field(default="", max_length=40)
    detail: str = Field(default="", max_length=120)
    session_id: str = Field(default="", max_length=64)


@app.post("/api/event")
async def client_event(req: EventRequest, request: Request):
    """iPhone mic diagnostics (tap, start, error codes) written to the server log; no personal data."""
    _require_access(request)
    _check_rate_limit("event:" + _client_ip(request))
    safe = lambda v: re.sub(r"[^\w=.:-]", "_", v)  # keep the log line plain
    logging.warning("client-event %s %s session=%s", safe(req.name), safe(req.detail), safe(req.session_id[:8]))
    return {"ok": True}


@app.get("/health")
async def health():
    # prompt_version lets a deploy confirm the live knowledge base matches the one the test set passed on
    return {"prompt_version": prompt_version(build_system_blocks("public")), "status": "ok", "kb_loaded": bool(KB_TEXT) or KB_PATH.exists(), "rules_loaded": bool(private_rules()), "voice_installed": VOICE_PATH.exists(),
            "log_sync": bool(_log_sync)}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    _require_access(request)
    _check_rate_limit(_client_ip(request))

    if req.mode == "private" and not (PRIVATE_PASSCODE and hmac.compare_digest(req.passcode, PRIVATE_PASSCODE)):
        raise HTTPException(403, "Private mode needs the correct passcode.")
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(400, "The last message must be from the user.")
    if len(req.messages[-1].content) > MAX_USER_CHARS:
        raise HTTPException(400, f"Please keep questions under {MAX_USER_CHARS} characters.")

    history = [m.model_dump() for m in req.messages[-MAX_TURNS:]]
    if history[0]["role"] != "user":
        history = history[1:]
    system = build_system_blocks(req.mode)

    visitor = hashlib.sha256(f"{LOG_SALT}:{_client_ip(request)}".encode()).hexdigest()[:12]

    async def generate():
        start = time.perf_counter()
        first_token_s = None
        said = []  # everything the visitor saw, including fallback/error messages
        record = {
            "id": uuid.uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": req.session_id[:64],
            "visitor": visitor,
            "turn": sum(m["role"] == "user" for m in history),
            "mode": req.mode,
            "owner": req.owner,
            "model": MODEL,
            "prompt_version": prompt_version(system),
            "question": history[-1]["content"],
            # earlier turns, so an answer that depends on context can be judged fairly
            "context": [{"role": m["role"], "content": m["content"][:1500]} for m in history[:-1][-6:]],
        }
        try:
            async with client.beta.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "medium"},
                system=system,
                messages=history,
            ) as stream:
                async for text in stream.text_stream:
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - start
                    said.append(text)
                    yield text
                final = await stream.get_final_message()

            if final.stop_reason == "refusal":
                said.append(f"\n\n{REFUSAL_TEXT}")
                yield said[-1]
            usage = final.usage
            record.update(
                stop_reason=final.stop_reason,
                served_by=final.model,
                input_tokens=usage.input_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
                output_tokens=usage.output_tokens,
                cache_write_tokens=usage.cache_creation_input_tokens,
            )
            record["completed"] = True
        except anthropic.RateLimitError:
            record["error"] = "rate_limited"
            said.append("I'm getting a lot of questions right now. Please try again in a minute.")
            yield said[-1]
        except anthropic.APIConnectionError:
            record["error"] = "connection"
            said.append("I couldn't reach my brain just now. Please try again.")
            yield said[-1]
        except anthropic.APIStatusError as e:
            logging.exception("Claude API error")
            record["error"] = f"api_{e.status_code}"
            said.append("Something went wrong on my side. Please try again.")
            yield said[-1]
        finally:
            # "completed" stays False if the visitor pressed End Conversation mid-answer
            record.setdefault("completed", False)
            record["answer"] = "".join(said)
            record["first_token_s"] = round(first_token_s, 3) if first_token_s else None
            record["total_s"] = round(time.perf_counter() - start, 3)
            _log_turn(record)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index(key: str = ""):
    if not _has_access(key):
        return HTMLResponse(INVITE_ONLY_PAGE, status_code=403)
    # no-cache: browsers re-check the page every load, so UI edits show up on a normal refresh.
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})
