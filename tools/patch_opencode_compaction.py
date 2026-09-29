"""Insert the compaction block and extend instructions in opencode.jsonc.

Written in Python rather than PowerShell because the config is CRLF and a
PowerShell here-string silently fails to match it, which is how the first
attempt at this edit was a no-op.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

P = Path.home() / ".config" / "opencode" / "opencode.jsonc"
raw = P.read_text(encoding="utf-8")

COMPACTION = """  // ---------------------------------------------------------------------
  // Context compaction - keep long sessions alive
  // ---------------------------------------------------------------------
  // opencode already compacts on its own. No third-party context-compression
  // repo was downloaded: the built-in mechanism covers this case, and an
  // unvetted plugin sitting in the context path is a worse risk than a stale
  // context. What the DEFAULTS get wrong is `prune`.
  //
  // `prune` defaults to FALSE, which keeps every old tool output verbatim. In
  // a campaign session the context is dominated not by conversation but by
  // pasted test output, CI logs and file reads - artefacts nobody will read a
  // second time. Pruning reclaims them, which is what buys the session more
  // working turns before a compaction has to happen at all.
  //
  // `auto` stays on. `preserve_recent_tokens` keeps the live working thread
  // verbatim rather than compressing everything, and `reserved` leaves room
  // for the compaction call itself so it cannot overflow mid-operation.
  "compaction": {
    "auto": true,
    "prune": true,
    "preserve_recent_tokens": 30000,
    "reserved": 24000
  },

"""

# 1) drop any previous attempt, so this is idempotent
raw = re.sub(
    r"[ \t]*// -+\n[ \t]*// Context compaction.*?\n[ \t]*\"compaction\":\s*\{[^}]*\}\s*,\n\n",
    "",
    raw,
    flags=re.DOTALL,
)

# 2) insert the compaction block immediately before the instructions comment
marker = re.search(
    r"^([ \t]*// -+\n[ \t]*// Global rules for every project)", raw, flags=re.MULTILINE
)
if not marker:
    raise SystemExit("marker not found: the Global-rules comment block")
raw = raw[: marker.start()] + COMPACTION + raw[marker.start() :]


# 3) extend instructions with the campaign resume file
def fix_instructions(m: re.Match[str]) -> str:
    return (
        "  // plan.md and TASKS.md are the campaign's resume point. Instruction\n"
        "  // files are re-read after any compaction, so putting the recovery\n"
        "  // procedure here is what makes surviving a context reset automatic\n"
        "  // rather than something to be remembered.\n"
        '  "instructions": [\n'
        '    "C:/Users/Илья/.config/opencode/AGENTS.md",\n'
        '    "C:/Users/Илья/.config/opencode/CAMPAIGN-RESUME.md"\n'
        "  ],"
    )


raw, n = re.subn(
    r"^[ \t]*\"instructions\":\s*\[[^\]]*\],",
    fix_instructions,
    raw,
    count=1,
    flags=re.MULTILINE,
)
if n != 1:
    raise SystemExit("instructions line not matched")

P.write_text(raw, encoding="utf-8")

# verify: parse it back
stripped = re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE)
cfg = json.loads(stripped)
print("compaction:", json.dumps(cfg.get("compaction"), indent=2))
print("instructions:", json.dumps(cfg.get("instructions"), indent=2))
print("brace balance:", raw.count("{") - raw.count("}"))
