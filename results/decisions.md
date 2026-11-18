# Decisions and approvals

One entry per approval gate (RUNBOOK_OMNI.md, H1–H3) or other team decision. Never edit a past entry; add a new one.

| Date | Gate | Decision | Approved by |
| --- | --- | --- | --- |
| 2026-11-18 | Scope | Prepare only seed-0 GiGPO, threshold, and weighted runs; omit GRPO and extra seeds. Run both variants whenever thresholds can be derived, even if forecast threshold impact is small. GPU resource and spending gates H1/H2 remain pending. | Project owner (chat request) |
| 2026-11-18 | Experiment design (before any run) | Replace threshold rule v1 with v2: `MIN_GROUP_SIZE=2` (was 3), `MIN_RETURN_STD=0.075` fixed from the reward structure (was p25 of spread-group std), `WEIGHT_TAU` unchanged (median, control steps 1–20). The extension records per-group min/max for the band report (diagnostic only). Rationale: `knowledge_base/threshold_rule.md`. No GPU job had run and no data had been seen. H1/H2 still pending. | Aditi Kumar, written request in a Claude Code session |
