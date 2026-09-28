"""Build .agent-notes/skills-index.md without reading whole skill bodies.

Only the YAML frontmatter head of each SKILL.md is read, and only the first
line of `description`. That is the point: the index must be cheap enough to
rebuild every tick, and the bodies are what the model loads on demand.
"""

from __future__ import annotations

import re
from pathlib import Path

DEST = Path.home() / ".config" / "opencode" / "skills"
SUPERPOWERS = Path.home() / ".config" / "opencode" / "repos" / "superpowers" / "skills"
AGENTS_LIB = Path.home() / "skill-shop" / "agents-lib"
OUT = Path(".agent-notes/skills-index.md")

NEWLY_INSTALLED = [
    "debugging",
    "code-review",
    "testing",
    "refactoring",
    "version-control",
    "code-documentation",
    "multi-agent-orchestration",
    "agent-evaluation",
    "prompt-injection-defense",
]

SUPERPOWERS_USED = [
    "brainstorming",
    "dispatching-parallel-agents",
    "subagent-driven-development",
    "systematic-debugging",
    "test-driven-development",
    "executing-plans",
    "requesting-code-review",
    "finishing-a-development-branch",
]

# group -> (keyword, ordered candidate stems) for the agents-lib lookup.
# The order is deliberate: a bare substring match picks the wrong file.
# "test" would select hypothesis-testing (statistics, not test authoring) and
# "security" would select backend-security-coder (implementation, not audit).
ROLE_LOOKUP = [
    ("G-PERF", "performance", ["performance-engineer", "performance-optimization", "performance"]),
    ("G-FIX", "debug", ["team-debugger", "team-debug", "debugger", "debug"]),
    (
        "G-COVER",
        "test",
        ["test-automator", "test-writer", "test-engineer", "testing-engineer", "test"],
    ),
    ("G-HYGIENE", "document", ["api-documenter", "technical-writer", "documentation", "document"]),
    ("review", "review", ["team-reviewer", "code-reviewer", "reviewer", "review"]),
    ("security", "security", ["security-auditor", "security", "security-coder"]),
]


def head(path: Path, n: int = 12) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:n])


def field(text: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip().strip("\"'") if m else "-"


def first_desc_line(text: str) -> str:
    """Handle `description: value`, `description: >` and `description: |`."""
    m = re.search(r"^description:\s*(.*)$", text, re.MULTILINE)
    if not m:
        return "(none)"
    inline = m.group(1).strip()
    if inline and inline not in {">", "|", ">-", "|-"}:
        return inline
    nxt = re.search(r"^description:\s*[>|].*?\n\s*(.+)$", text, re.MULTILINE | re.DOTALL)
    return nxt.group(1).strip() if nxt else "(folded, none found)"


def row(skill_md: Path) -> str:
    h = head(skill_md)
    name = field(h, "name")
    desc = first_desc_line(h)
    if len(desc) > 88:
        desc = desc[:85] + "..."
    return f"| `{name}` | {desc} |"


lines: list[str] = []
lines.append("# skills-index")
lines.append("")
lines.append("Rebuild: `python tools/build_skills_index.py`")
lines.append("")

lines.append("## (a) installed this campaign — 9/9")
lines.append("")
lines.append("| name | description (first line of frontmatter) |")
lines.append("|---|---|")
missing_a = []
for s in NEWLY_INSTALLED:
    p = DEST / s / "SKILL.md"
    if p.is_file():
        lines.append(row(p))
    else:
        missing_a.append(s)
        lines.append(f"| `{s}` | **MISSING** |")
lines.append("")

lines.append("## (b) superpowers, reached through the junction into `repos/`")
lines.append("")
lines.append("| name | description (first line of frontmatter) |")
lines.append("|---|---|")
missing_b = []
for s in SUPERPOWERS_USED:
    p = SUPERPOWERS / s / "SKILL.md"
    if p.is_file():
        lines.append(row(p))
    else:
        missing_b.append(s)
        lines.append(f"| `{s}` | **MISSING** |")
lines.append("")

lines.append("## (c) agents-lib role file per group leader")
lines.append("")
lines.append("Role = a file, not a persistent process (see the architecture note in")
lines.append("docs/decisions.md). Memory of a leader is its `board.md`, not its context.")
lines.append("")
lines.append("| group | keyword | chosen file | exact? |")
lines.append("|---|---|---|---|")
lib_files = sorted(AGENTS_LIB.rglob("*.md")) if AGENTS_LIB.is_dir() else []
by_stem: dict[str, list[Path]] = {}
for _f in lib_files:
    by_stem.setdefault(_f.stem, []).append(_f)

for group, keyword, candidates in ROLE_LOOKUP:
    chosen, tag = None, "none"
    for rank, cand in enumerate(candidates):
        if cand in by_stem:
            chosen = by_stem[cand][0]
            tag = "exact" if cand == keyword else f"preference rank {rank + 1}"
            break
    if chosen is None:
        near = [f for f in lib_files if keyword in f.stem]
        if near:
            chosen, tag = near[0], f"substring ({keyword} in name)"
    lines.append(f"| {group} | `{keyword}` | {chosen.name if chosen else chr(45)} | {tag} |")

lines.append("## counts")
lines.append("")
lines.append(
    f"- installed this campaign: {9 - len(missing_a)}/9"
    + (f" — MISSING: {missing_a}" if missing_a else "")
)
lines.append(
    f"- superpowers reachable: {len(SUPERPOWERS_USED) - len(missing_b)}/{len(SUPERPOWERS_USED)}"
    + (f" — MISSING: {missing_b}" if missing_b else "")
)
lines.append(f"- agents-lib files scanned: {len(lib_files)}")
lines.append(
    f"- total skills visible to the agent: {len([d for d in DEST.iterdir() if d.is_dir()])} in `{DEST}`"
)
lines.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
