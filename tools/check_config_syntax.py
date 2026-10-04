"""Validate the opencode config: brace balance and JSONC parse."""

import json
import sys
from pathlib import Path

P = Path.home() / ".config" / "opencode" / "opencode.jsonc"
raw = P.read_text(encoding="utf-8")

# strip // comments that are not inside a string
out, in_str, esc, i = [], False, False, 0
while i < len(raw):
    c = raw[i]
    if in_str:
        out.append(c)
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = False
    else:
        if c == '"':
            in_str = True
            out.append(c)
        elif c == "/" and i + 1 < len(raw) and raw[i + 1] == "/":
            while i < len(raw) and raw[i] != "\n":
                i += 1
            continue
        else:
            out.append(c)
    i += 1
stripped = "".join(out)

bal = 0
line = 1
minbal = 0
for ch in stripped:
    if ch == "\n":
        line += 1
    elif ch == "{":
        bal += 1
    elif ch == "}":
        bal -= 1
        minbal = min(minbal, bal)
print(f"brace balance at EOF : {bal}   (0 = balanced)")
print(f"min depth during scan: {minbal}   (>=0 = never closed early)")

try:
    cfg = json.loads(stripped)
    print("JSON parse           : OK")
    print(f"top-level keys       : {len(cfg)}")
    bash = cfg.get("permission", {}).get("bash", {})
    print(f"bash rules           : {len(bash)}")
except json.JSONDecodeError as e:
    print(f"JSON parse           : FAILED -> {e}")
    sys.exit(1)

sys.exit(0 if bal == 0 and minbal >= 0 else 1)
