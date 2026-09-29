# consolidation log

One line per tick, strictly:
`TICK <n> | runs: <k> | done <a>/<b> | rej <c> | 429: <d> | gate: <sec> | X-><new>`
TICK 1 | runs: 8 | done 7/8 (verifier not a card) | rej 0 | 429: 0 | gate: 60s | X 8->12

      worker yield: TASK-01 1 | TASK-02 1 | TASK-03 1 | TASK-04 1 | TASK-07 1 | TASK-13 1 | TASK-17 1 | TASK-26 (read-only, 10 seeds)

      verifier: gate GREEN x2, boundary clean, 0 new skip/xfail. Found 1 HIGH regression I introduced (CMC resume) - FIXED with 7-test detector.

      control-scout: 4 of 7 high-benefit RAID seeds SURVIVE, 2 killed, 1 narrowed. 2 seed numbers corrected.

      weights: G-FIX 1, G-COVER 1, G-PERF 1, G-HYGIENE 1, G-RAID 1+4(raid yield) = 6

TICK 2 | runs: 4 | done 4/4 | rej 0 | 429: 0 | gate: 75s | X 12

      TASK-27 tests for run_comparison (16) | TASK-28 determinism fixture (15) | TASK-29 typo paths (6) | TASK-30 baseline parity (23)

      3 of 4 workers corrected the card they were given. Gate hole closed: 5 test files were never in gate.sh.

