# Literature Review: LLMs Know the Answer They Want to Give

**Hypothesis.** RL-fine-tuned LLMs tend to *pre-plan the end of their next turn*, which gives
them *less ability to react to new information* than base LLMs. By *rotating what is certain*
about the answer across a series of questions, it may be possible to force the model to
*distribute its uncertainty* across different parts of a larger question.

The hypothesis makes three distinct, separately-testable claims:
- **(C1) Pre-planning exists** — models commit to global properties of their response (its
  answer, and the location of its turn-ending) before/early in generation.
- **(C2) RL increases pre-planning and reduces reactivity** — RL fine-tuning sharpens this
  commitment relative to the base model, degrading the ability to revise given new information.
- **(C3) Rotating certainty redistributes uncertainty** — manipulating which part of a question
  is held fixed forces the model to spread uncertainty across the remaining parts.

The literature strongly supports C1, gives partial and somewhat *mixed* support for C2, and
leaves C3 essentially open — which is where this project's novelty lies.

---

## 1. Research Area Overview

Four threads converge on this hypothesis:

1. **Internal pre-planning / look-ahead (interpretability).** A growing body shows that
   decoder-only LMs, despite next-token training, encode global attributes of their *entire*
   upcoming output in early/prompt hidden states. This is the mechanistic substrate the
   hypothesis depends on.
2. **Reactivity, revision, and RL training effects.** Work on multi-turn RL, self-correction,
   and "lost-in-conversation" shows that RL-style post-training can collapse a model's ability
   to revise — the predicted deficit.
3. **Sycophancy / change-of-mind (behavioral).** A complementary literature measures how models
   react (or over-react) to challenges and new information across turns.
4. **Uncertainty quantification & decomposition.** Methods for measuring and decomposing LLM
   uncertainty — the toolkit for operationalizing "distributing uncertainty."

---

## 2. Key Papers (by claim)

### C1 — Pre-planning exists (well supported)

- **Emergent Response Planning in LLMs** (Dong et al., ICML 2025; arXiv:2502.06258). Simple MLP
  probes on the *prompt* hidden state predict the full response's **length, number of reasoning
  steps, final multiple-choice answer, and eventual correctness/confidence** — all before the
  first token. Prompts are engineered so the attribute is *not* readable from the immediate next
  token, ruling out the first-token shortcut. Probing far exceeds the model's own verbalized
  self-estimate. A three-phase "U-shape" along the generation timeline shows commitment is
  strongest at the start, dips mid-generation, and recovers near the end. **Instruction-tuned >
  base** on *structure* attributes (length, steps); content/behavior planning is present even in
  base models. *This is the single most important supporting paper for C1, and its probing
  protocol is directly reusable.*

- **On Reasoning Strength Planning in LRMs** (NeurIPS 2025; arXiv:2506.08390). The reasoning-token
  budget is linearly decodable from the `<think>` activation (Spearman > 0.8) and is encoded as a
  **single steerable direction** whose L2 norm grows with question difficulty and *causally*
  controls where the model emits `</think>`. This is the most literal evidence for "pre-plan the
  end of the turn" — but all subjects are RL/distilled reasoning models, with **no base contrast**.

- **Do Language Models Plan Ahead for Future Tokens?** (Wu et al., COLM 2024; arXiv:2404.00859).
  Formalizes "pre-caching" vs "breadcrumbs," introduces *myopic training* and the **myopia gap**.
  Natural-language models mostly use breadcrumbs, but **pre-caching grows with scale** — bigger
  models plan further ahead.

- **Detecting and Characterizing Planning in LMs** (arXiv:2508.18098). Causal SAE-circuit
  definition of planning (Future-Token Encoding + Precursor Influence) and a Planning-vs-
  **Improvisation** ("commit early vs decide late") axis. Notably finds **instruction tuning
  *selects among* existing plans rather than creating planning**, and that base models can hold
  **"Competing Plans"** (multiple candidate futures simultaneously) — directly relevant to C3.

- **Future Lens** (Pal et al.; arXiv:2311.04897) is the foundational probing primitive (future
  tokens linearly decodable from one hidden state). **Internal Planning… Horizon and Branch
  Awareness** (arXiv:2509.25260) and **What's the plan?** (arXiv:2601.20164) extend implicit-
  planning probes/steering (e.g., a planned rhyme or answer can be changed by steering the
  preceding token), and scale them across many models.

### C2 — RL increases pre-planning / reduces reactivity (partial, mixed)

- **A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning (UFO)** (arXiv:2507.14295). **The
  strongest direct support.** Single-turn RLVR (PPO, GRPO, DAPO, Dr.GRPO) **collapses revision**:
  in ~70% of failures the RL model repeats the *identical* answer across 5 turns, and the
  **unique-answer ratio drops after RL across every method and scale (0.5B–32B)**. Mechanistic
  account: RL produces a peaked, low-entropy policy, and collision probability ≥ exp(−H) forces
  repetition. Base/pre-RL models, by contrast, do revise. The fix (UFO) reintroduces reactivity
  by injecting unary feedback and rewarding answer diversity.

- **Training LMs to Self-Correct via RL (SCoRe)** (Kumar et al., DeepMind; arXiv:2409.12917).
  Naive RL/SFT causes **"behavior collapse"** — the model locks in its first answer (edit
  distance ≈ 0, "different-answer frequency" falls during RL). Clean metrics Δ(t1,t2), Δ^{i→c},
  Δ^{c→i} separate *reacts-and-fixes* from *commits-and-refuses* from *over-reacts-and-breaks*.
  Importantly, the **base** Gemini here *over-reacts* (Δ = −11%: it flips correct answers), while
  training pushes toward *not changing* — so the base-vs-RL story is **direction-dependent**.

- **Recursive Introspection (RISE)** (arXiv:2407.18219) — even strong LLMs don't sequentially
  self-improve even when told they erred; fine-tuning can install the ability. **Lost-in-
  Conversation / RLAAR** (arXiv:2510.18731) — performance degrades as information is revealed
  progressively across turns (the literal "react to new information" axis), mitigated by
  curriculum RL with abstention rewards. **Not All Thoughts Are Generated Equal** (arXiv:2505.11827)
  — multi-turn RL to compress CoT, quantifying which thoughts matter.

### C2 (behavioral) — sycophancy & change-of-mind (mixed evidence on RL)

- **Are You Sure? The FlipFlop Experiment** (Laban et al.; arXiv:2311.08596). 10 models × 7 tasks
  × 5 challengers: **46% flip rate, −17% accuracy** when challenged. Crucially, sycophancy is
  **universal across models trained with and without RLHF** and is **not monotonic in RLHF**
  (GPT-4/PaLM resist; Claude V1.3/V2 flip most) — this *weakens* a naive "RL causes the deficit"
  reading and suggests pre-training contributes. Here the failure is *over*-reaction, the inverse
  of pre-commitment — the two failure modes coexist and depend on model/task.

- **TRUTH DECAY** (arXiv:2503.11656) and **SYCON Bench / Measuring Sycophancy Multi-turn**
  (arXiv:2505.23840; metrics Turn-of-Flip, Number-of-Flip) quantify sycophancy over extended
  dialogues. **Overconfidence/Underconfidence & Change of Mind** (arXiv:2507.03120) elicits
  confidence *without memory* of the prior judgment, revealing a **choice-supportive bias** and
  resistance to changing one's mind — almost a literal restatement of "knows the answer it wants
  to give."

### C3 — Rotating certainty / distributing uncertainty (largely open)

- **CoT-UQ** (arXiv:2502.17214): response-wise (not prompt-wise) uncertainty via reasoning steps;
  flags overconfidence when reasoning. **Uncertainty Profiles** (arXiv:2505.07309): decomposes
  LLM uncertainty into four sources — the conceptual basis for "uncertainty across parts." No
  paper found directly tests *rotating which part of a question is fixed* to redistribute
  uncertainty; the closest mechanistic hook is the **"Competing Plans"** probe from Detecting-
  Planning and the answer-confidence probe from Emergent Response Planning.

---

## 3. Common Methodologies

- **Probing pre-commitment:** train small (MLP/linear/Lasso) probes on hidden states — at the
  prompt position or at a control token (`<think>`) — to predict full-response attributes (answer,
  length, correctness). (Emergent Response Planning; Reasoning Strength Planning; Future Lens.)
- **Causal steering / ablation:** add or subtract a difference-in-means direction (`h' = h + λr`)
  or ablate SAE latents, then regenerate, to test whether a plan is *causal*. (Reasoning Strength
  Planning; plan_trace's FTE+PI; What's the plan?)
- **Multi-turn reactivity harnesses:** convert single-turn data into multi-turn episodes with
  feedback ("Try Again", "Are you sure?", challengers) and measure revision. (UFO; SCoRe;
  FlipFlop; SYCON; RLAAR.)
- **Behavior-collapse / repetition metrics:** unique-answer ratio, edit-distance ratio,
  different-answer frequency, output entropy.

## 4. Standard Baselines

- **Base vs instruction-tuned vs RL-tuned** of the *same* model family (the central contrast our
  hypothesis needs). Matched pairs exist: **Qwen2.5-{0.5B,1.5B,7B,14B,32B}** vs SimpleRL-Zoo
  (GRPO) / Oat-Zero (Dr.GRPO) / DAPO; **DeepSeek-R1-Distill-Qwen/Llama** vs their Qwen2.5/Llama-3
  bases; Qwen2.5-3B-Instruct vs the released Qwen2.5-3B-UFO.
- Method baselines for reactivity: Self-Refine, STaR, Pair-SFT, RISE, SCoRe, UFO.
- Probing baselines: random/majority; the model's own verbalized self-estimate; Future-Lens
  affine probe.

## 5. Evaluation Metrics

- **Reactivity / revision:** Δ(t1,t2), Δ^{i→c}, Δ^{c→i} (SCoRe); FlipFlop effect Δ_FF and flip
  rates (Any/Correct/Wrong→Flip); Turn-of-Flip, Number-of-Flip (SYCON); unique/effective-answer
  ratio, AvgTurns, Succ@k vs Pass@k (UFO).
- **Pre-commitment / planning:** probe accuracy (Spearman/Kendall/Pearson for continuous,
  balanced F1 for categorical); *how early* the answer becomes decodable along the generation
  timeline; myopia gap; FTE+PI satisfaction; number of Competing Plans.
- **Uncertainty:** output entropy/collision probability; calibration (ECE); CoT-UQ response-wise
  scores; source-decomposed uncertainty profiles.

## 6. Datasets in the Literature

- **Math/reasoning:** GSM8K, MATH/MATH500, AIME2024, OlympiadBench, TheoremQA, GPQA.
- **Multiple-choice / commonsense:** MMLU(/-STEM/-Pro), ARC-Challenge, CommonsenseQA, SocialIQA,
  MedMCQA, SciQ, TruthfulQA, PIQA, LAMBADA.
- **Multi-hop / decomposable QA:** HotpotQA, ConcurrentQA, StrategyQA (ideal for C3).
- **Fact-checking / stance:** CREAK, FEVER, SummEdits, LegalBench-CCQA.
- **Code:** MBPP/MBPP-R, HumanEval.
- **Open-ended:** UltraChat, AlpacaEval (for length planning), TinyStories/ROCStories.

*Downloaded locally* (samples committed): GSM8K (test), HotpotQA (distractor val sample),
ARC-Challenge (test), StrategyQA (test). See `datasets/README.md`.

## 7. Gaps and Opportunities

1. **No clean base-vs-RL test of reactivity *via pre-planning probes*.** Pre-planning work
   (Emergent Response Planning, Reasoning Strength Planning) does not isolate RL; RL-reactivity
   work (UFO, SCoRe) is behavioral and does not probe internal pre-commitment. **Combining the
   two** — probe how early the answer is committed in matched base vs RL models, and correlate
   that with behavioral non-reactivity — is the central open experiment (C1+C2).
2. **The deficit direction is contested.** UFO/SCoRe show RL → *under*-reaction (commit/repeat);
   FlipFlop/sycophancy work shows widespread *over*-reaction, not RLHF-monotonic. A careful study
   must measure *both* directions and the conditions (model, task, challenge type) that select
   between them — this is a real risk to a naive C2 and should be designed for, not assumed.
3. **C3 is essentially untested.** No work rotates what is held certain across a question series to
   redistribute uncertainty. The "Competing Plans" probe and the answer-confidence probe provide
   the mechanism to *measure* whether uncertainty actually spreads.
4. **Verbalization gap.** Models encode far more about their plan than they can state — so C3
   should be evaluated with *probes*, not just self-reported confidence.

## 8. Recommendations for Our Experiment

- **Models (matched base/RL pairs):** start with Qwen2.5-{1.5B,7B} base vs SimpleRL-Zoo (GRPO)
  and vs DeepSeek-R1-Distill-Qwen; add Qwen2.5-3B-Instruct vs Qwen2.5-3B-UFO for the reactivity
  axis. These give same-architecture base/RL contrasts on commodity GPUs.
- **Probes (C1/C2):** reuse `LRM-plans-CoT`'s `<think>`-activation probe + difference-in-means
  steering, and Emergent-Response-Planning's prompt-position MLP probes, but **extend the target
  from length to the answer**, and **compare base vs RL** + measure *how early* the answer is
  decodable. Use `plan_trace` (FTE+PI) where SAEs are available (Gemma-2-2B base/`-it`).
- **Behavioral reactivity (C2):** use the UFO "Try Again" harness and SCoRe's Δ-metrics, plus the
  FlipFlop challenger protocol, on the same models. Report unique-answer ratio, Δ^{c→i}/Δ^{i→c},
  and flip rates so over- vs under-reaction are both visible.
- **Rotating certainty (C3):** on decomposable multi-hop items (HotpotQA, StrategyQA) and
  multi-choice (ARC), construct question variants that fix different sub-answers as "given," then
  measure (a) probe-estimated answer commitment and (b) entropy / Competing-Plans count on the
  remaining parts. Test whether holding part A certain *raises* measured uncertainty on part B.
- **Metrics:** primary = answer-commitment-earliness (probe) and unique-answer-ratio /
  Δ-metrics (behavior); secondary = output entropy, ECE, FlipFlop effect. Always report the
  base-vs-RL delta, and the over- vs under-reaction breakdown.
- **Decoding:** greedy/temperature-0 for deterministic commitment comparisons; a temperature
  sweep (per FlipFlop) to bound sensitivity.

**Bottom line.** C1 is on firm ground and immediately reusable infrastructure exists. The
project's contribution is to (i) supply the missing base-vs-RL pre-planning-probe contrast tying
C1 to C2, while honestly characterizing the over- vs under-reaction tension the literature
exposes, and (ii) be the first to test C3 — rotating certainty to redistribute uncertainty —
using probes rather than self-report.
