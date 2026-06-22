# Resources Catalog

All resources gathered for **"LLMs Know the Answer They Want to Give."** Companion docs:
`literature_review.md` (synthesis), `papers/README.md`, `datasets/README.md`, `code/README.md`,
and `papers/deep_read_notes.json` (full structured notes on the 7 deep-read papers).

## Summary
- **Papers downloaded:** 18 (7 deep-read with full methodology notes, 11 abstract-screened)
- **Datasets downloaded:** 4 (GSM8K, HotpotQA, ARC-Challenge, StrategyQA)
- **Repositories cloned:** 3 (unary-feedback/UFO, LRM-plans-CoT, plan_trace)

---

## Papers (18)

| # | Title | Year | File | arXiv | Cluster |
|---|-------|------|------|-------|---------|
| 1 | Emergent Response Planning in LLMs **[DEEP]** | 2025 | papers/emergent_response_planning.pdf | 2502.06258 | A pre-planning |
| 2 | On Reasoning Strength Planning in LRMs **[DEEP]** | 2025 | papers/reasoning_strength_planning.pdf | 2506.08390 | A pre-planning |
| 3 | Do Language Models Plan Ahead for Future Tokens? **[DEEP]** | 2024 | papers/do_lms_plan_ahead.pdf | 2404.00859 | A pre-planning |
| 4 | Detecting and Characterizing Planning in LMs **[DEEP]** | 2025 | papers/detecting_planning_lms.pdf | 2508.18098 | A pre-planning |
| 5 | Future Lens | 2023 | papers/future_lens.pdf | 2311.04897 | A pre-planning |
| 6 | Internal Planning: Horizon & Branch Awareness | 2025 | papers/internal_planning_horizon_branch.pdf | 2509.25260 | A pre-planning |
| 7 | What's the plan? Metrics for implicit planning | 2026 | papers/whats_the_plan_implicit_planning.pdf | 2601.20164 | A pre-planning |
| 8 | "Try Again" Multi-Turn Reasoning (UFO) **[DEEP]** | 2025 | papers/try_again_multiturn.pdf | 2507.14295 | B reactivity/RL |
| 9 | Training LMs to Self-Correct via RL (SCoRe) **[DEEP]** | 2024 | papers/score_self_correct_rl.pdf | 2409.12917 | B reactivity/RL |
| 10 | Recursive Introspection (RISE) | 2024 | papers/recursive_introspection.pdf | 2407.18219 | B reactivity/RL |
| 11 | Mitigating Lost-in-Multi-turn via Curriculum RL | 2025 | papers/lost_in_multiturn_curriculum_rl.pdf | 2510.18731 | B reactivity/RL |
| 12 | Not All Thoughts Are Generated Equal | 2025 | papers/not_all_thoughts_equal.pdf | 2505.11827 | B reactivity/RL |
| 13 | Are You Sure? FlipFlop Experiment **[DEEP]** | 2024 | papers/flipflop_are_you_sure.pdf | 2311.08596 | C sycophancy |
| 14 | TRUTH DECAY: Multi-Turn Sycophancy | 2025 | papers/truth_decay_sycophancy.pdf | 2503.11656 | C sycophancy |
| 15 | Measuring Sycophancy Multi-turn (SYCON) | 2025 | papers/measuring_sycophancy_multiturn.pdf | 2505.23840 | C sycophancy |
| 16 | Overconfidence/Underconfidence & Change of Mind | 2025 | papers/overconfidence_changing_minds.pdf | 2507.03120 | C sycophancy |
| 17 | CoT-UQ: Response-wise Uncertainty Quantification | 2025 | papers/cot_uq.pdf | 2502.17214 | D uncertainty |
| 18 | Uncertainty Profiles: Source Decomposition | 2025 | papers/uncertainty_profiles_decomposition.pdf | 2505.07309 | D uncertainty |

See `papers/README.md` for one-line relevance notes and `papers/deep_read_notes.json` for the
7 deep-read papers' full methodology/datasets/baselines/metrics/results/reusable-assets.

## Datasets (4)

| Name | Source (HF) | Size | Task | Location |
|------|-------------|------|------|----------|
| GSM8K | openai/gsm8k (main) | 1,319 test | multi-step math | datasets/gsm8k/ |
| HotpotQA | hotpotqa/hotpot_qa (distractor) | 2,000 val sample | 2-hop QA (decomposable) | datasets/hotpotqa/ |
| ARC-Challenge | allenai/ai2_arc | 1,172 test | science MC (commitment) | datasets/arc_challenge/ |
| StrategyQA | ChilleD/StrategyQA | 687 test | implicit multi-hop yes/no | datasets/strategyqa/ |

Data files git-ignored; `samples.json` committed. Download instructions in `datasets/README.md`.
Additional literature benchmarks (MATH, MMLU, GPQA, TruthfulQA, MBPP, CommonsenseQA, …) are
pullable on demand via the same `load_dataset` pattern.

## Code Repositories (3)

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| unary-feedback (UFO) | github.com/lichengliu03/unary-feedback | multi-turn "Try Again" reactivity harness + unique-answer metric | code/unary-feedback/ |
| LRM-plans-CoT | github.com/AlphaLab-USTC/LRM-plans-CoT | `<think>`-activation probe + steering of pre-planned reasoning length | code/LRM-plans-CoT/ |
| plan_trace | github.com/ambitious-mechinterp/plan_trace | SAE-circuit planning detection (FTE+PI), Competing-Plans probe | code/plan_trace/ |

Per-repo details, key files, and install notes in `code/README.md`.

---

## Resource Gathering Notes

### Search strategy
Used the paper-finder service (fast mode) with four angle-specific queries: (1) RL pre-planning /
reactivity, (2) internal planning-ahead interpretability, (3) answer commitment / sycophancy /
failure to revise, (4) uncertainty calibration / decomposition. Pooled ~200 ranked hits;
prioritized relevance-2/3 and high-citation foundational work. arXiv IDs resolved via the arxiv
API; PDFs fetched over HTTP and integrity-checked (`%PDF` magic). The 7 most central papers were
deep-read in full (chunked to 3-page PDFs, read by parallel sub-agents into a structured schema).

### Selection criteria
Direct bearing on one of the three hypothesis claims (C1 pre-planning, C2 RL-reduces-reactivity,
C3 rotate-certainty); preference for papers releasing reusable probes/harnesses/metrics and for
named base-vs-RL model pairs.

### Challenges encountered
- paper-finder needed `httpx` (installed); first run fell back. arXiv relevance search returned
  physics noise for short/common-word titles — resolved with `ti:` field queries and direct
  `id_list` lookups. The installed `arxiv` 4.0.0 lacks `download_pdf`, so PDFs were fetched via
  `requests` on `pdf_url`. Two initial title mismatches (a survey, an unrelated paper) were
  detected by word-overlap thresholding and re-fetched by exact ID.

### Gaps and workarounds
- **No single paper isolates RL-vs-base reactivity via pre-planning probes** — this is the core
  experimental gap (see literature_review.md §7). Workaround: combine reusable probe code
  (LRM-plans-CoT) with reusable reactivity harness (UFO) on matched base/RL model pairs.
- **C3 (rotating certainty) has no direct precedent** — design must be built from the
  Competing-Plans probe (plan_trace) and answer-confidence probe (Emergent Response Planning).
- **SCoRe and FlipFlop release no official code** — their metrics are simple and re-implementable
  from the deep-read notes.

## Recommendations for Experiment Design (condensed)

1. **Primary datasets:** GSM8K + ARC-Challenge (clean answer commitment); HotpotQA + StrategyQA
   (decomposable, for C3). Pull MATH/MMLU on demand for scaling.
2. **Models:** matched base-vs-RL pairs — Qwen2.5-{1.5B,7B} base vs SimpleRL-Zoo (GRPO) and vs
   DeepSeek-R1-Distill-Qwen; Qwen2.5-3B-Instruct vs Qwen2.5-3B-UFO.
3. **Methods to reuse:** `<think>`/prompt-position probes + difference-in-means steering
   (LRM-plans-CoT); FTE+PI / Competing-Plans (plan_trace); "Try Again" multi-turn harness (UFO).
4. **Metrics:** answer-commitment-earliness (probe), unique-answer ratio, Δ(t1,t2)/Δ^{c→i}/Δ^{i→c},
   FlipFlop effect, output entropy/ECE — always reported as base-vs-RL deltas and split by
   over- vs under-reaction.
5. **Key risk to design for:** the deficit direction is contested (RL→under-react vs
   widespread over-reaction). Measure both; do not assume a single direction.
