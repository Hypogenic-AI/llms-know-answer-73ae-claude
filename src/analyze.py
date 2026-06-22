"""Analysis & figures for all three experiments.

Reads results/ JSON, computes metrics + statistical tests, writes
results/evaluations/*.json summaries and figures/*.png.
"""
import os, json
import numpy as np
from scipy import stats
from src.common import RESULTS, FIGURES, MODEL_LABEL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(FIGURES, exist_ok=True)
MODELS = ["base", "instruct", "rl"]
COL = {"base": "#4C72B0", "instruct": "#DD8452", "rl": "#C44E52"}


def boot_ci(x, fn=np.mean, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(fn(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ============ EXP 1 — reactivity ============
def analyze_exp1(dataset="arc"):
    path = f"{RESULTS}/model_outputs/exp1_{dataset}.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    gold = {q: g for q, g in zip(d["qid"], d["gold"])}
    t1 = d["turn1"]
    by = {}  # (model,feedback) -> list of records
    for r in d["records"]:
        by.setdefault((r["model"], r["feedback"]), []).append(r)

    out = {"dataset": dataset, "models": {}}
    ftypes = sorted({k[1] for k in by})
    for m in [m for m in MODELS if m in t1]:
        out["models"][m] = {}
        # turn1 accuracy
        t1acc = np.mean([1.0 if a == gold[q] else 0.0
                         for q, a in zip(d["qid"], t1[m]) if a is not None])
        out["models"][m]["turn1_acc"] = float(t1acc)
        out["models"][m]["turn1_parse_fail"] = float(np.mean([a is None for a in t1[m]]))
        for ft in ftypes:
            recs = by.get((m, ft), [])
            changed, i2c, c2i, valid = [], [], [], 0
            uniq = []
            for r in recs:
                a1, a2, g = r["ans1"], r["ans2"], r["gold"]
                if a1 is None or a2 is None:
                    continue
                valid += 1
                ch = int(a1 != a2)
                changed.append(ch)
                uniq.append(len({a1, a2}))
                if a1 != g and a2 == g:
                    i2c.append(1)
                else:
                    i2c.append(0)
                if a1 == g and a2 != g:
                    c2i.append(1)
                else:
                    c2i.append(0)
            m_ch, lo, hi = boot_ci(changed)
            out["models"][m][ft] = dict(
                n=valid,
                change_rate=m_ch, change_ci=[lo, hi],
                flip_i2c=float(np.mean(i2c)) if i2c else None,
                flip_c2i=float(np.mean(c2i)) if c2i else None,
                unique_ratio=float(np.mean(uniq)) if uniq else None,
                changed=changed,  # keep for tests
            )
    # stat test: base vs rl change-rate under generic (Fisher / chi2 on 2x2)
    tests = {}
    for ft in ftypes:
        try:
            cb = out["models"]["base"][ft]["changed"]
            cr = out["models"]["rl"][ft]["changed"]
            ci = out["models"]["instruct"][ft]["changed"]
            # base vs rl
            tab = [[sum(cb), len(cb) - sum(cb)], [sum(cr), len(cr) - sum(cr)]]
            _, p_br = stats.fisher_exact(tab)
            tab2 = [[sum(cb), len(cb) - sum(cb)], [sum(ci), len(ci) - sum(ci)]]
            _, p_bi = stats.fisher_exact(tab2)
            tests[ft] = dict(base_change=float(np.mean(cb)),
                             instruct_change=float(np.mean(ci)),
                             rl_change=float(np.mean(cr)),
                             p_base_vs_rl=float(p_br),
                             p_base_vs_instruct=float(p_bi))
        except Exception as e:
            tests[ft] = {"error": str(e)}
    out["tests"] = tests
    # strip bulky arrays
    for m in out["models"]:
        for ft in ftypes:
            if ft in out["models"][m]:
                out["models"][m][ft].pop("changed", None)
    return out, ftypes


def plot_exp1(out, ftypes, dataset):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(ftypes)); w = 0.25
    for i, m in enumerate(MODELS):
        if m not in out["models"]:
            continue
        vals = [out["models"][m][ft]["change_rate"] for ft in ftypes]
        los = [out["models"][m][ft]["change_rate"] - out["models"][m][ft]["change_ci"][0] for ft in ftypes]
        his = [out["models"][m][ft]["change_ci"][1] - out["models"][m][ft]["change_rate"] for ft in ftypes]
        ax.bar(x + (i - 1) * w, vals, w, yerr=[los, his], capsize=3,
               label=MODEL_LABEL[m], color=COL[m])
    ax.set_xticks(x); ax.set_xticklabels(ftypes)
    ax.set_ylabel("Answer change rate (turn1→turn2)")
    ax.set_xlabel("Feedback type")
    ax.set_title(f"Exp1: Reactivity to new information ({dataset.upper()})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/exp1_reactivity_{dataset}.png", dpi=130)
    plt.close(fig)

    # flip direction figure (generic + hint)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ft in zip(axes, ["generic", "hint"]):
        i2c = [out["models"][m][ft]["flip_i2c"] for m in MODELS if m in out["models"]]
        c2i = [out["models"][m][ft]["flip_c2i"] for m in MODELS if m in out["models"]]
        labels = [MODEL_LABEL[m] for m in MODELS if m in out["models"]]
        xx = np.arange(len(labels))
        ax.bar(xx - 0.2, i2c, 0.4, label="i→c (good fix)", color="#55A868")
        ax.bar(xx + 0.2, c2i, 0.4, label="c→i (over-react)", color="#C44E52")
        ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"Flip direction — {ft} feedback")
        ax.set_ylabel("Fraction of items"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/exp1_flipdir_{dataset}.png", dpi=130)
    plt.close(fig)


# ============ EXP 2 — probe ============
def analyze_exp2():
    path = f"{RESULTS}/evaluations/exp2_probe.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    # plot layer sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        if m not in d:
            continue
        la = d[m]["layer_acc"]
        xs = np.linspace(0, 1, len(la))
        ax.plot(xs, la, marker="o", ms=3, label=MODEL_LABEL[m], color=COL[m])
    ax.axhline(list(d.values())[0]["chance"], ls="--", c="gray", label="chance")
    ax.set_xlabel("Relative layer depth (0=embed,1=final)")
    ax.set_ylabel("Probe balanced acc — predict model's final answer")
    ax.set_title("Exp2: Answer pre-encoding at the PROMPT position, by layer")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/exp2_layer_sweep.png", dpi=130)
    plt.close(fig)

    # timeline
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        if m not in d:
            continue
        ax.plot(d[m]["timeline_fracs"], d[m]["timeline_acc"], marker="s",
                label=MODEL_LABEL[m], color=COL[m])
    ax.axhline(list(d.values())[0]["chance"], ls="--", c="gray", label="chance")
    ax.set_xlabel("Generation progress (0=prompt end, 1=final token)")
    ax.set_ylabel("Probe balanced acc (mid layer)")
    ax.set_title("Exp2: Answer-commitment trajectory along generation")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/exp2_timeline.png", dpi=130)
    plt.close(fig)
    return d


# ============ EXP 3 — rotating certainty ============
def analyze_exp3():
    path = f"{RESULTS}/model_outputs/exp3_strategyqa.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))["results"]
    out = {"models": {}}
    for m in MODELS:
        if m not in d:
            continue
        recs = d[m]
        H0 = np.array([r["H_baseline"] for r in recs])
        HA = np.array([r["H_+A"] for r in recs])
        HB = np.array([r["H_+B"] for r in recs])
        dA = H0 - HA  # entropy reduction when fixing A
        dB = H0 - HB
        asym = np.abs(dA - dB)
        rA_m, rA_lo, rA_hi = boot_ci(dA)
        rB_m, rB_lo, rB_hi = boot_ci(dB)
        # paired Wilcoxon: does fixing reduce entropy?
        def wilco(x):
            x = x[np.abs(x) > 1e-12]
            if len(x) < 8:
                return None
            return float(stats.wilcoxon(x)[1])
        red_combined = np.concatenate([dA, dB])
        mr, lo, hi = boot_ci(red_combined)
        out["models"][m] = dict(
            n=len(recs),
            H_baseline=float(H0.mean()), H_A=float(HA.mean()), H_B=float(HB.mean()),
            mean_reduction=mr, reduction_ci=[lo, hi],
            reduction_A=float(dA.mean()), reduction_A_ci=[rA_lo, rA_hi],
            reduction_B=float(dB.mean()), reduction_B_ci=[rB_lo, rB_hi],
            asymmetry=float(asym.mean()),
            p_reduction=wilco(red_combined),
            # reactivity to injected certainty = mean abs entropy change
            reactivity=float(np.mean(np.abs(np.concatenate([dA, dB])))),
            # flip rate of majority answer when a fact is given
            flip_given=float(np.mean([
                int(r["maj_+A"] != r["maj_baseline"]) +
                int(r["maj_+B"] != r["maj_baseline"]) for r in recs]) / 2),
        )
    # cross-model test: base vs rl reduction (Mann-Whitney on per-item combined reductions)
    try:
        rb = np.concatenate([
            np.array([r["H_baseline"] - r["H_+A"] for r in d["base"]]),
            np.array([r["H_baseline"] - r["H_+B"] for r in d["base"]])])
        rr = np.concatenate([
            np.array([r["H_baseline"] - r["H_+A"] for r in d["rl"]]),
            np.array([r["H_baseline"] - r["H_+B"] for r in d["rl"]])])
        out["test_base_vs_rl_reduction_p"] = float(stats.mannwhitneyu(rb, rr)[1])
    except Exception as e:
        out["test_base_vs_rl_reduction_p"] = None

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    xx = np.arange(len([m for m in MODELS if m in out["models"]]))
    labels = [MODEL_LABEL[m] for m in MODELS if m in out["models"]]
    for i, cond in enumerate(["H_baseline", "H_A", "H_B"]):
        vals = [out["models"][m][cond] for m in MODELS if m in out["models"]]
        ax.bar(xx + (i - 1) * 0.25, vals, 0.25,
               label={"H_baseline": "baseline", "H_A": "+fact A", "H_B": "+fact B"}[cond])
    ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Mean answer entropy (bits)")
    ax.set_title("Exp3: Answer entropy under rotated certainty")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    keys = [m for m in MODELS if m in out["models"]]
    rA = [out["models"][m]["reduction_A"] for m in keys]
    rB = [out["models"][m]["reduction_B"] for m in keys]
    eA = [[out["models"][m]["reduction_A"] - out["models"][m]["reduction_A_ci"][0] for m in keys],
          [out["models"][m]["reduction_A_ci"][1] - out["models"][m]["reduction_A"] for m in keys]]
    eB = [[out["models"][m]["reduction_B"] - out["models"][m]["reduction_B_ci"][0] for m in keys],
          [out["models"][m]["reduction_B_ci"][1] - out["models"][m]["reduction_B"] for m in keys]]
    ax.bar(xx - 0.2, rA, 0.4, yerr=eA, capsize=3, label="ΔH | +fact A (signed)", color="#DD8452")
    ax.bar(xx + 0.2, rB, 0.4, yerr=eB, capsize=3, label="ΔH | +fact B (signed)", color="#55A868")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Entropy reduction (bits, >0 = redistributed)")
    ax.set_title("Exp3: Uncertainty absorbed when a hop is fixed")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/exp3_rotating.png", dpi=130)
    plt.close(fig)
    return out


def main():
    summary = {}
    r1 = analyze_exp1("arc")
    if r1:
        out, ftypes = r1
        plot_exp1(out, ftypes, "arc")
        summary["exp1_arc"] = out
    r1g = analyze_exp1("gsm8k")
    if r1g:
        out, ftypes = r1g
        plot_exp1(out, ftypes, "gsm8k")
        summary["exp1_gsm8k"] = out
    e2 = analyze_exp2()
    if e2:
        summary["exp2"] = {m: {k: v for k, v in e2[m].items()
                               if k not in ("layer_acc",)} for m in e2}
    e3 = analyze_exp3()
    if e3:
        summary["exp3"] = e3
    with open(f"{RESULTS}/evaluations/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float)[:4000])
    print("\n[analyze] figures in figures/, summary in results/evaluations/summary.json")


if __name__ == "__main__":
    main()
