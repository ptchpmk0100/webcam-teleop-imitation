# 02: Eval harness and null checks
What to build: evaluate(policy, seeds) over fixed held-out seeds returning success rate and a binomial confidence interval, plus a held-out seed list and a do-nothing and a random-action policy.
Blocked by: 01
Kind: AFK
- [ ] The test command exits 0 only if the do-nothing policy scores 0 of 100 and the random policy at most 2 of 100.
- [ ] A deliberately broken harness (always reports success) makes that test fail (mutation check, kept as a test).
- [ ] Evaluation is deterministic for a fixed seed list.

Status: CLOSED 2026-11-14. Verified: `make test` exit 0 (28 passed); `python -m teleop_il.evaluation` exit 0 (hold 0/100, random 0/100, CI 0.000-0.037); 5 mutations of the harness each turn it red. Note: the null checks prove the harness can fail, not that success is reachable; ticket 06 (expert) supplies the positive control.
