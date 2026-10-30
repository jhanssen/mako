#!/bin/sh
# Renders an image with mako's /v1/images/generations and saves it as a PNG.
#   render.sh WIDTHxHEIGHT "prompt"
# MAKO_URL (default http://127.0.0.1:8080/v1), MAKO_API_KEY (when the server wants one), MAKO_IMAGE_DIR (default
# ~/Pictures/mako), MAKO_IMAGE_OPEN=0 (not opened in the image viewer). Prints the saved file's path.
set -eu
if [ $# -ne 2 ]; then
    echo "usage: render.sh WIDTHxHEIGHT \"prompt\"" >&2
    exit 2
fi
size=$1
prompt=$2
url=${MAKO_URL:-http://127.0.0.1:8080/v1}
dir=${MAKO_IMAGE_DIR:-$HOME/Pictures/mako}
mkdir -p "$dir"
out="$dir/$(date +%Y%m%d-%H%M%S).png"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
body=$(SIZE=$size PROMPT=$prompt python3 -c '
import json, os
print(json.dumps({"prompt": os.environ["PROMPT"], "size": os.environ["SIZE"], "n": 1, "response_format": "b64_json"}))')
status=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 1800 "$url/images/generations" \
    -H 'Content-Type: application/json' ${MAKO_API_KEY:+-H "Authorization: Bearer $MAKO_API_KEY"} -d "$body")
OUT=$out STATUS=$status python3 - "$tmp" <<'PY'
import base64, json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(f"mako answered HTTP {os.environ['STATUS']} with no JSON")
if os.environ["STATUS"] != "200" or "data" not in d:
    sys.exit(f"mako: HTTP {os.environ['STATUS']}: {d.get('error', {}).get('message', d)}")
open(os.environ["OUT"], "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
PY
if [ "${MAKO_IMAGE_OPEN:-1}" != 0 ]; then
    open "$out"
fi
echo "$out"
