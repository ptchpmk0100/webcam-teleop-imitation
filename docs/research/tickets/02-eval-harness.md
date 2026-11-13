# 02: Eval harness and null checks
What to build: evaluate(policy, seeds) over fixed held-out seeds returning success rate and a binomial confidence interval, plus a held-out seed list and a do-nothing and a random-action policy.
Blocked by: 01
Kind: AFK
- [ ] The test command exits 0 only if the do-nothing policy scores 0 of 100 and the random policy at most 2 of 100.
- [ ] A deliberately broken harness (always reports success) makes that test fail (mutation check, kept as a test).
- [ ] Evaluation is deterministic for a fixed seed list.
