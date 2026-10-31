#!/bin/sh
# Lists the image models mako can render with (GET /v1/images/models): a name per line, the default and the loaded one
# marked, then each one's model family, description, steps and default LoRAs.
# MAKO_URL (default http://127.0.0.1:8080/v1), MAKO_API_KEY (when the server wants one).
set -eu
url=${MAKO_URL:-http://127.0.0.1:8080/v1}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
status=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 30 "$url/images/models" \
    ${MAKO_API_KEY:+-H "Authorization: Bearer $MAKO_API_KEY"})
STATUS=$status python3 - "$tmp" <<'PY'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(f"mako answered HTTP {os.environ['STATUS']} with no JSON")
if os.environ["STATUS"] != "200" or "data" not in d:
    sys.exit(f"mako: HTTP {os.environ['STATUS']}: {d.get('error', {}).get('message', d)}")
for m in d["data"]:
    marks = [k for k in ("default", "loaded") if m.get(k)]
    print(m["id"] + (f"  ({', '.join(marks)})" if marks else ""))
    print(f"    {m.get('family', '')}")
    if "description" in m:
        print(f"    {m['description']}")
    if m.get("loras"):
        print("    default LoRAs: " + ", ".join(f"{l['name']} {l['scale']:g}" for l in m["loras"]))
PY
