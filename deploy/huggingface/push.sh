#!/bin/zsh
# Stage only what the live site needs and upload it to the Hugging Face Space.
# Usage: deploy/huggingface/push.sh <hf-username>/<space-name>
# Never uploads .env, logs, the venv, the avatar-prep tools, the knowledge base, or the private rules (they go in the TWIN_KB_TEXT and TWIN_RULES_TEXT secrets).
# STAGE_ONLY=1 builds the bundle and prints its folder without uploading (for a local test run).
# Refuses to upload unless evals/run_eval.py (with the grounding judge) fully passed on the current persona.md and
# knowledge base. SKIP_EVAL=1 overrides that, for changes that can't affect answers.
set -euo pipefail
SPACE=${1:?usage: push.sh <hf-username>/<space-name>}
HERE=${0:A:h}
ROOT=${HERE:h:h}
PY="$ROOT/venv/bin/python"

LOCAL_VERSION=$(cd "$ROOT" && "$PY" -c 'import sys; sys.path.insert(0, "tools"); from grounding import load_sources, prompt_version; print(prompt_version(*load_sources()))')
if [[ -z ${STAGE_ONLY:-} && -z ${SKIP_EVAL:-} ]]; then
  "$PY" - "$ROOT/evals/latest.json" "$LOCAL_VERSION" <<'PYEOF' || exit 1
import json, sys
path, version = sys.argv[1], sys.argv[2]
try:
    run = json.load(open(path))
except FileNotFoundError:
    sys.exit("Not deploying: no test run yet. Run ./venv/bin/python evals/run_eval.py first.")
if run["prompt_version"] != version:
    sys.exit(f"Not deploying: persona.md or the knowledge base changed since the last test run "
             f"({run['prompt_version']} -> {version}). Re-run evals/run_eval.py.")
if not (run["all_passed"] and run["judged"]):
    sys.exit(f"Not deploying: last test run passed {run['passed']}/{run['total']}"
             f"{'' if run['judged'] else ' without the grounding judge'}. See evals/{run['results']}.")
print(f"Test set passed {run['passed']}/{run['total']} on prompt {version} ({run['ran_at']}).")
PYEOF
fi
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp "$HERE/Dockerfile" "$HERE/requirements.txt" "$HERE/README.md" "$STAGE/"
cp "$ROOT/app.py" "$ROOT/kb_prep.py" "$ROOT/persona.md" "$STAGE/"
mkdir -p "$STAGE/static" "$STAGE/voices"
cp "$ROOT/static/index.html" "$STAGE/static/"
cp -R "$ROOT/static/avatar_video" "$ROOT/static/avatar" "$STAGE/static/"
cp "$ROOT"/voices/en_US-hfc_male-medium.onnx* "$ROOT/voices/MODEL_CARD" "$STAGE/voices/"
find "$STAGE" -name .DS_Store -delete

if [[ -n ${STAGE_ONLY:-} ]]; then trap - EXIT; echo "$STAGE"; exit 0; fi
echo "Uploading to https://huggingface.co/spaces/$SPACE ..."
"$ROOT/venv/bin/hf" upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy Jarvis (prompt $LOCAL_VERSION)"

# The knowledge base lives in the Space's TWIN_KB_TEXT secret, so check the live site is answering from the same
# persona + knowledge base the tests passed on.
URL="https://${SPACE/\//-}.hf.space/health"
echo "Waiting for the Space to rebuild ..."
sleep 60
for i in {1..30}; do
  LIVE=$(curl -s -m 10 "$URL" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("prompt_version",""))' 2>/dev/null || true)
  if [[ "$LIVE" == "$LOCAL_VERSION" ]]; then echo "Live and matching: prompt $LIVE"; exit 0; fi
  sleep 10
done
echo "WARNING: live prompt is '${LIVE:-unreachable}', expected $LOCAL_VERSION."
echo "If the Space is running, update its TWIN_KB_TEXT secret (knowledge base) and TWIN_RULES_TEXT (Jarvis_Private_Rules.md)."
exit 1
