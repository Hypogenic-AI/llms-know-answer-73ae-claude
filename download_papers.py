"""Resolve selected papers to arXiv and download PDFs into papers/."""
import re, time, json
import arxiv, requests

SELECTED = [
    ("emergent_response_planning", "Emergent Response Planning in LLMs"),
    ("reasoning_strength_planning", "On Reasoning Strength Planning in Large Reasoning Models"),
    ("do_lms_plan_ahead", "Do language models plan ahead for future tokens"),
    ("future_lens", "Future Lens Anticipating Subsequent Tokens from a Single Hidden State"),
    ("detecting_planning_lms", "Detecting and Characterizing Planning in Language Models"),
    ("internal_planning_horizon_branch", "Internal Planning in Language Models Characterizing Horizon and Branch Awareness"),
    ("whats_the_plan_implicit_planning", "What's the plan Metrics for implicit planning in language models rhyme generation"),
    ("try_again_multiturn", "A Simple Try Again Can Elicit Multi-Turn LLM Reasoning"),
    ("score_self_correct_rl", "Training Language Models to Self-Correct via Reinforcement Learning"),
    ("recursive_introspection", "Recursive Introspection Teaching Language Model Agents How to Self-Improve"),
    ("flipflop_are_you_sure", "Are You Sure Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment"),
    ("truth_decay_sycophancy", "Truth Decay Quantifying Multi-Turn Sycophancy in Language Models"),
    ("overconfidence_changing_minds", "Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Changing Minds in LLMs"),
    ("measuring_sycophancy_multiturn", "Measuring Sycophancy of Language Models in Multi-turn Dialogues"),
    ("cot_uq", "CoT-UQ Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought"),
    ("lost_in_multiturn_curriculum_rl", "Mitigating Lost in Multi-turn Conversation via Curriculum Reinforcement Learning"),
    ("not_all_thoughts_equal", "Not All Thoughts are Generated Equal Efficient LLM Reasoning Multi-Turn Reinforcement Learning"),
    ("uncertainty_profiles_decomposition", "Uncertainty Profiles for LLMs Uncertainty Source Decomposition"),
]

client = arxiv.Client(page_size=8, delay_seconds=3, num_retries=3)
HDRS = {"User-Agent": "Mozilla/5.0 (research paper fetcher)"}

def words(s):
    return set(re.findall(r'[a-z0-9]+', s.lower()))

log = []
for stem, title in SELECTED:
    try:
        q = re.sub(r'[^a-zA-Z0-9 ]', ' ', title)
        search = arxiv.Search(query=q, max_results=8,
                              sort_by=arxiv.SortCriterion.Relevance)
        tw = words(title)
        best = None
        for r in client.results(search):
            ov = len(tw & words(r.title)) / max(1, len(tw))
            if best is None or ov > best[0]:
                best = (ov, r)
        if best and best[0] >= 0.55:
            r = best[1]
            aid = r.get_short_id()
            url = r.pdf_url
            resp = requests.get(url, headers=HDRS, timeout=60)
            resp.raise_for_status()
            with open(f"papers/{stem}.pdf", "wb") as f:
                f.write(resp.content)
            kb = len(resp.content)//1024
            log.append({"stem": stem, "title": title, "matched": r.title,
                        "arxiv_id": aid, "overlap": round(best[0],2),
                        "pdf_url": url, "abs_url": r.entry_id, "kb": kb, "status": "ok"})
            print(f"OK   {stem:36s} {aid:13s} {kb:5d}KB ov={best[0]:.2f} {r.title[:50]}")
        else:
            ov = best[0] if best else 0
            log.append({"stem": stem, "title": title, "status": "no_match",
                        "best_overlap": round(ov,2),
                        "best_title": best[1].title if best else None})
            print(f"MISS {stem:36s} ov={ov:.2f} : {best[1].title[:55] if best else 'none'}")
    except Exception as e:
        log.append({"stem": stem, "title": title, "status": "error", "error": str(e)})
        print(f"ERR  {stem:36s} {e}")
    time.sleep(1)

json.dump(log, open("papers/_download_log.json", "w"), indent=2)
ok = sum(1 for r in log if r['status']=='ok')
print(f"\n=== downloaded {ok}/{len(SELECTED)} ===")
