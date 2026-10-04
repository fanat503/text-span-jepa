"""Run the opencode policy-plugin test suite from the campaign gate.

Why this wrapper exists
-----------------------
`.agent-notes/gate.sh` runs under Git-Bash, and Git-Bash mangles a non-ASCII
path segment: `C:/Users/Илья` arrives as `C:/Users/Èëüÿ`, so a hardcoded
`node "C:/Users/Илья/..."` inside the shell script fails with
`Cannot find module`. Python resolves `Path.home()` correctly and hands node the
proper path, so the shell script only has to call this.

It also fails LOUD. If the policy plugin is missing, or node is unavailable, or
the path cannot be resolved, this exits non-zero. A gate that quietly skips the
one check standing between a confused agent and `shutil.rmtree` on the repo
would be worse than no gate at all, because the owner would believe the repo
is protected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = Path.home() / ".config" / "opencode" / "plugins"
TEST_FILE = PLUGIN_DIR / "policy.test.mjs"
PLUGIN_FILE = PLUGIN_DIR / "policy.js"


def fail(msg: str) -> int:
    print(f"check_policy: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    if not PLUGIN_FILE.is_file():
        return fail(
            f"policy plugin missing at {PLUGIN_FILE}. The agent safety policy is "
            f"NOT installed, so nothing is standing between an agent and a "
            f"recursive delete of the repo."
        )
    if not TEST_FILE.is_file():
        return fail(
            f"policy test missing at {TEST_FILE}. The policy is installed but "
            f"unverified, which is the same as absent for the owner's purposes."
        )

    node = shutil.which("node")
    if node is None:
        return fail("node is not on PATH, so the policy test cannot run")

    # Keep the denial log next to the plugin, and start from a known state so a
    # green run cannot be reporting on a stale log.
    log = Path(os.environ.get("OPENCODE_POLICY_LOG", PLUGIN_DIR / "policy-denials.log"))
    try:
        if log.is_file():
            log.unlink()
    except OSError as exc:  # pragma: no cover
        print(f"check_policy: could not reset {log}: {exc}", file=sys.stderr)

    proc = subprocess.run(
        [node, str(TEST_FILE)],
        cwd=str(HERE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "NODE_NO_WARNINGS": "1"},
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    print(f"check_policy: plugin and test present, {TEST_FILE.name} green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
