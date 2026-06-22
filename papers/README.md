# Downloaded Papers

18 papers, organized by the three literatures the hypothesis sits at the intersection of.
Deep-read papers (full methodology extracted) are marked **[DEEP]**; structured notes for
all deep-read papers are in `deep_read_notes.json`. Others were screened by abstract.

Hypothesis under study: *RL-fine-tuned LLMs tend to pre-plan the end of their next turn,
giving them less ability to react to new information than base LLMs; rotating what is
certain across a series of questions may force the model to distribute its uncertainty
across parts of a larger question.*

---

## A. Internal pre-planning / answer commitment (the mechanism)

1. **[DEEP] Emergent Response Planning in LLMs** — `emergent_response_planning.pdf`
   - Dong, Zhou, Liu, Yang, Lu. ICML 2025. arXiv:2502.06258
   - MLP probes on the *prompt* hidden state predict the full response's length, reasoning
     steps, final MC answer, and correctness/confidence *before any token is generated*.
   - Relevance: direct evidence models commit to global response attributes (incl. answer &
     length) pre-generation. Compares instruction-tuned vs base (not RL specifically).

2. **[DEEP] On Reasoning Strength Planning in Large Reasoning Models** — `reasoning_strength_planning.pdf`
   - NeurIPS 2025. arXiv:2506.08390. Code: github.com/AlphaLab-USTC/LRM-plans-CoT
   - Reasoning-token budget is linearly decodable from the `<think>` activation (Spearman>0.8);
     encoded as a single steerable direction whose norm grows with difficulty and causally
     controls where `</think>` is emitted.
   - Relevance: strongest evidence for "pre-plan the end of the turn." All models are RL/distilled
     LRMs; no base contrast (a gap we can fill).

3. **[DEEP] Do Language Models Plan Ahead for Future Tokens?** — `do_lms_plan_ahead.pdf`
   - Wu, Morris, Levine. COLM 2024. arXiv:2404.00859. Code: github.com/wiwu2390/FutureGPT2-public
   - Distinguishes "pre-caching" (deliberate lookahead) vs "breadcrumbs"; introduces myopic
     training and the *myopia gap*; pre-caching grows with scale.

4. **[DEEP] Detecting and Characterizing Planning in Language Models** — `detecting_planning_lms.pdf`
   - Nainani et al. arXiv:2508.18098. Code: github.com/ambitious-mechinterp/plan_trace
   - SAE-circuit definition of planning (Future-Token Encoding + Precursor Influence);
     planning vs improvisation. Instruction tuning *selects among* existing plans rather than
     creating planning. "Competing Plans" = distributed candidate futures.

5. **Future Lens: Anticipating Subsequent Tokens from a Single Hidden State** — `future_lens.pdf`
   - Pal et al. arXiv:2311.04897. Foundational: future tokens are linearly decodable from one
     hidden state (>48% at some layers in GPT-J-6B). The probing primitive others build on.

6. **Internal Planning in Language Models: Characterizing Horizon and Branch Awareness** — `internal_planning_horizon_branch.pdf`
   - arXiv:2509.25260. Analyzes hidden states for horizon (how far ahead) and branch awareness
     (multiple continuations) without CoT scaffolds.

7. **What's the plan? Metrics for implicit planning in LLMs (rhyme & QA)** — `whats_the_plan_implicit_planning.pdf`
   - arXiv:2601.20164. Simple, scalable steering metrics for implicit planning; shows the
     planned rhyme/answer ("whale") can be changed by steering at the preceding line/token.

## B. Reactivity, revision & RL training effects (the predicted deficit)

8. **[DEEP] A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning (UFO)** — `try_again_multiturn.pdf`
   - Liu et al. arXiv:2507.14295. Code: github.com/lichengliu03/unary-feedback
   - **Most direct support:** single-turn RLVR (PPO/GRPO/DAPO/Dr.GRPO) collapses revision —
     ~70% of failures repeat the *identical* answer across turns; unique-answer ratio drops
     after RL across all methods/scales. Mechanism: low-entropy policy ⇒ forced repetition.

9. **[DEEP] Training Language Models to Self-Correct via RL (SCoRe)** — `score_self_correct_rl.pdf`
   - Kumar et al. (DeepMind). arXiv:2409.12917
   - Naive RL/SFT "behavior collapse": models lock in the first answer (edit distance ≈ 0).
     Metrics Δ(t1,t2), Δ^{i→c}, Δ^{c→i} cleanly separate reacts/commits/over-reacts.

10. **Recursive Introspection: Teaching LM Agents How to Self-Improve (RISE)** — `recursive_introspection.pdf`
    - Qu et al. arXiv:2407.18219. Even strong LLMs don't sequentially improve answers; RISE
      fine-tunes the ability in. Counterpoint method to the deficit.

11. **Mitigating Lost in Multi-turn Conversation via Curriculum RL (RLAAR)** — `lost_in_multiturn_curriculum_rl.pdf`
    - arXiv:2510.18731. "Lost-in-Conversation": degradation as info is revealed progressively
      across turns — exactly the "react to new information" axis. Adds abstention rewards.

12. **Not All Thoughts are Generated Equal (Long⊗Short)** — `not_all_thoughts_equal.pdf`
    - arXiv:2505.11827. Multi-turn RL to compress CoT; quantifies effectiveness/efficiency of
      individual thoughts — relevant to where commitment/uncertainty lives in a CoT.

## C. Sycophancy / changing-mind / answer commitment (behavioral reactivity)

13. **[DEEP] Are You Sure? The FlipFlop Experiment** — `flipflop_are_you_sure.pdf`
    - Laban et al. (Salesforce). arXiv:2311.08596. 10 models × 7 tasks × 5 challengers (67,640
      convs): 46% flip rate, −17% accuracy. Sycophancy is *universal* (not RLHF-monotonic).

14. **TRUTH DECAY: Quantifying Multi-Turn Sycophancy** — `truth_decay_sycophancy.pdf`
    - arXiv:2503.11656. Benchmark for sycophancy over extended dialogues (4 bias types).

15. **Measuring Sycophancy of LMs in Multi-turn Dialogues (SYCON Bench)** — `measuring_sycophancy_multiturn.pdf`
    - arXiv:2505.23840. Metrics: Turn-of-Flip and Number-of-Flip under sustained pressure.

16. **How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind** — `overconfidence_changing_minds.pdf`
    - arXiv:2507.03120. Confidence elicited *without memory* of prior judgment; choice-supportive
      bias + resistance to change. Directly relevant to "knows the answer it wants to give."

## D. Uncertainty quantification & decomposition (the proposed intervention)

17. **CoT-UQ: Response-wise Uncertainty Quantification via Chain-of-Thought** — `cot_uq.pdf`
    - arXiv:2502.17214. Response-wise (not prompt-wise) UQ leveraging reasoning steps; notes
      overconfidence when using reasoning.

18. **Uncertainty Profiles for LLMs: Uncertainty Source Decomposition** — `uncertainty_profiles_decomposition.pdf`
    - arXiv:2505.07309. Decomposes LLM uncertainty into 4 sources — conceptual basis for
      "distributing uncertainty across parts of a question."

---

`_download_log.json` records arXiv IDs, matched titles, and PDF URLs for all downloads.
`deep_read_notes.json` holds full structured notes (methodology/datasets/baselines/metrics/
results/relevance/reusable-assets) for the 7 **[DEEP]** papers.
