#!/bin/sh
# Lists the LoRAs an image model can render with (GET /v1/images/loras): a name per line, the model's defaults marked
# with the scale they are merged at, then each one's trigger, recommended scales and description when known.
#   loras.sh [MODEL]   (default: the server's default model)
# MAKO_URL (default http://127.0.0.1:8080/v1), MAKO_API_KEY (when the server wants one).
set -eu
url=${MAKO_URL:-http://127.0.0.1:8080/v1}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
query=
if [ $# -ge 1 ]; then
    query="?model=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$1")"
fi
status=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 30 "$url/images/loras$query" \
    ${MAKO_API_KEY:+-H "Authorization: Bearer $MAKO_API_KEY"})
STATUS=$status python3 - "$tmp" <<'PY'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(f"mako answered HTTP {os.environ['STATUS']} with no JSON")
if os.environ["STATUS"] != "200" or "data" not in d:
    sys.exit(f"mako: HTTP {os.environ['STATUS']}: {d.get('error', {}).get('message', d)}")
if not d["data"]:
    print("(no LoRAs)")
for l in d["data"]:
    print(l["name"] + (f"  (default, scale {l['default_scale']:g})" if "default_scale" in l else ""))
    if "trigger" in l:
        print(f"    trigger: {l['trigger']}")
    if "scale_range" in l:
        print(f"    scales: {l['scale_range'][0]:g} to {l['scale_range'][1]:g}")
    if "description" in l:
        print(f"    {l['description']}")
PY
