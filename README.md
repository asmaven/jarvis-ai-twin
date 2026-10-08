# Jarvis: Ayaz's AI Twin

A voice AI twin you can talk to about my work. Ask about my experience in product management, operations, and
building AI agents, and Jarvis answers in my voice, with an animated avatar, from a verified knowledge base of my career.

**Try it:** https://ayazshaik-jarvis.hf.space

Built end to end with [Claude Code](https://claude.com/claude-code) as a personal project.

## The problem

An AI that speaks for a real person has one job it can't fail: never make things up. A confident wrong answer about
someone's experience is worse than no answer. So most of this project isn't the chat. It's the system that keeps
the chat honest.

## How it stays honest

| Layer | What it does |
|---|---|
| **Grounding** | Jarvis answers only from a private knowledge base of my career. If something isn't covered, it says so and points to me instead of guessing. |
| **Guardrails** | Rules for tone, length (two lines max), facts that apply only in certain contexts, and topics it politely declines. |
| **AI judge** | A second model checks each answer claim by claim against the knowledge base and flags anything invented, misused, or off-limits. |
| **Release gate** | A hand-picked golden set of 29 questions runs before every deploy. If one fails, nothing ships, and the deploy script confirms the live site serves the version that passed. |
| **Learning loop** | Real conversations are logged privately (visitors are anonymized) and reviewed by the judge. Every problem becomes a new test, and fixes go into the knowledge base or rules, not the model. |

## What I learned

1. **Tests catch what you don't expect.** Relaxing one style rule quietly made answers grow past the two-line limit.
   The test suite caught it before anyone saw it.
2. **Newer isn't always cheaper.** A newer model with lower prices cost the same per visit because it reasoned longer,
   and it got worse at a question that matters. Measuring beat the spec sheet. Trimming the prompt cut costs about 25%.
3. **Mobile is its own product.** Voice that worked on desktop broke on iPhone in three different ways. On-device
   diagnostics found the causes faster than guessing.

## How it's built

```
Browser (avatar, mic, speaker)  ->  FastAPI server  ->  Claude (Anthropic API)
                                      |-- persona.md (rules) + private knowledge base, prompt-cached
                                      |-- Piper text-to-speech, with lip-sync from phoneme timings
                                      `-- private conversation logs -> AI judge -> review reports -> new tests
```

- **Model:** Claude Opus 5, streaming, with prompt caching.
- **Voice:** speech recognition in the browser (keyboard dictation on iPhone), and a natural voice generated on the server with [Piper](https://github.com/rhasspy/piper).
- **Avatar:** a Memoji whose mouth shapes follow the spoken sounds.
- **Hosting:** a Docker container on Hugging Face Spaces. Secrets, the knowledge base, and a few private answer rules live in the host's encrypted settings, not in this repo.

| Folder | Contents |
|---|---|
| `app.py`, `static/` | Server and web page |
| `persona.md` | Jarvis's rules and guardrails |
| `evals/` | Release-gate runner and sample test questions (the full set is private, like the knowledge base) |
| `tools/` | AI judge and conversation review |
| `deploy/huggingface/` | Container setup and the gated deploy script |

## Run your own

The knowledge base isn't included. Point Jarvis at your own markdown file.

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env    # add your ANTHROPIC_API_KEY, and set TWIN_KB_PATH to your knowledge base
./venv/bin/uvicorn app:app --port 8010    # open http://localhost:8010
```

Server voice is optional (browser voices are the fallback). To add it, download the Piper `en_US-hfc_male-medium`
voice into `voices/` and run `./venv/bin/python -m piper.patch_voice_with_alignment voices/en_US-hfc_male-medium.onnx`
for lip-sync timings.

## Credits

Voice: Piper "hfc_male" (Hi-Fi-CAPTAIN dataset, NICT), CC BY-NC-SA 4.0, used non-commercially with attribution.
