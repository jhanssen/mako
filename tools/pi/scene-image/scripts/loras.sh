#!/bin/sh
# Lists the LoRAs mako's image model can render with (GET /v1/images/loras): a name per line, the server's defaults
# marked with the scale they are merged at.
# MAKO_URL (default http://127.0.0.1:8080/v1), MAKO_API_KEY (when the server wants one).
set -eu
url=${MAKO_URL:-http://127.0.0.1:8080/v1}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
status=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 30 "$url/images/loras" \
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
PY
