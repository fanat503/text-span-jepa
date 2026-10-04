# skills-index

Rebuild: `python tools/build_skills_index.py`

## (a) installed this campaign — 9/9

| name | description (first line of frontmatter) |
|---|---|
| `debugging` | Systematically diagnose and fix software bugs by analyzing error messages, stack trac... |
| `code-review` | Perform thorough code reviews on files or pull requests, checking for bugs, security ... |
| `testing` | Generate, execute, and analyze tests for codebases, covering unit, integration, and e... |
| `refactoring` | "Improve code quality and maintainability through systematic identification of code s... |
| `version-control` | "Manage Git repositories and collaborative workflows — branching strategies, commit h... |
| `code-documentation` | "Automatically generate clear, comprehensive documentation for codebases — including ... |
| `multi-agent-orchestration` | Design and operate bounded multi-agent workflows with task decomposition, dependency ... |
| `agent-evaluation` | Design reproducible evaluations for AI agents with representative task sets, explicit... |
| `prompt-injection-defense` | Threat-model and harden AI agents, RAG systems, assistants, and tool-using workflows ... |

## (b) superpowers, reached through the junction into `repos/`

| name | description (first line of frontmatter) |
|---|---|
| `brainstorming` | "You MUST use this before any creative work - creating features, building components,... |
| `dispatching-parallel-agents` | Use when facing 2+ independent tasks that can be worked on without shared state or se... |
| `subagent-driven-development` | Use when executing implementation plans with independent tasks in the current session |
| `systematic-debugging` | Use when encountering any bug, test failure, or unexpected behavior, before proposing... |
| `test-driven-development` | Use when implementing any feature or bugfix, before writing implementation code |
| `executing-plans` | Use when executing an implementation plan in the current session as the implementer y... |
| `requesting-code-review` | Use when completing tasks, implementing major features, or before merging to verify w... |
| `finishing-a-development-branch` | Use when implementation is complete, all tests pass, and you need to decide how to in... |

## (c) agents-lib role file per group leader

Role = a file, not a persistent process (see the architecture note in
docs/decisions.md). Memory of a leader is its `board.md`, not its context.

| group | keyword | chosen file | exact? |
|---|---|---|---|
| G-PERF | `performance` | performance-engineer.md | preference rank 1 |
| G-FIX | `debug` | team-debugger.md | preference rank 1 |
| G-COVER | `test` | test-automator.md | preference rank 1 |
| G-HYGIENE | `document` | api-documenter.md | preference rank 1 |
| review | `review` | team-reviewer.md | preference rank 1 |
| security | `security` | security-auditor.md | preference rank 1 |
## counts

- installed this campaign: 9/9
- superpowers reachable: 8/8
- agents-lib files scanned: 774
- total skills visible to the agent: 32 in `C:\Users\Илья\.config\opencode\skills`
