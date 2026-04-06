"""Recreate ablation_tradeoff.png with Original annotation showing τ_edit."""
import json, os, glob, re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

base = "results_v2/ablation"
records = []

for d in sorted(os.listdir(base)):
    dpath = os.path.join(base, d)
    if not os.path.isdir(dpath):
        continue
    for sf in glob.glob(os.path.join(dpath, "*_summary.json")):
        if sf.endswith("/summary.json"):
            continue
        with open(sf) as f:
            s = json.load(f)
        mode = s.get("mode", "")
        strategy = s.get("strategy", "")
        acc = s.get("accuracy", 0) * 100

        if mode == "original":
            token_mods = s.get("avg_t2t_edits", 0)
            tau = None
            c_max = None
            rho = None
            editing_threshold = s.get("editing_threshold", None)
        else:
            token_mods = s.get("avg_remask_total", 0)
            tau = s.get("remask_threshold", None)
            m = re.search(r"_c(\d+)_", s.get("tag", ""))
            c_max = int(m.group(1)) if m else None
            rho = s.get("max_remask_ratio", None)
            editing_threshold = s.get("editing_threshold", None)

        records.append({
            "mode": mode,
            "strategy": strategy,
            "tau": tau,
            "c_max": c_max,
            "rho": rho,
            "acc": acc,
            "token_mods": token_mods,
            "avg_fwd": s.get("avg_forward_passes", 0),
            "avg_output_tokens": s.get("avg_output_tokens", 0),
            "editing_threshold": editing_threshold,
        })

tau_ranges = {
    "low_prob": (0.1, 0.9),
    "t2t_remask": (0.5, 0.9),
    "logit_diff": (0.1, 0.5),
}

def tau_alpha(strategy, tau):
    if tau is None:
        return 1.0
    lo, hi = tau_ranges.get(strategy, (0, 1))
    if hi == lo:
        return 1.0
    return 0.35 + 0.65 * (tau - lo) / (hi - lo)

fig, ax = plt.subplots(figsize=(14, 7))

for r in records:
    if r["mode"] != "remask" or r["strategy"] != "low_prob":
        continue
    alpha = tau_alpha("low_prob", r["tau"])
    color = (0.12, 0.38, 0.75, alpha)
    ax.scatter(r["token_mods"], r["acc"], c=[color], marker='o', s=60,
               edgecolors='none', zorder=3)

for r in records:
    if r["mode"] != "remask" or r["strategy"] != "t2t_remask":
        continue
    alpha = tau_alpha("t2t_remask", r["tau"])
    color = (0.93, 0.60, 0.15, alpha)
    ax.scatter(r["token_mods"], r["acc"], c=[color], marker='^', s=60,
               edgecolors='none', zorder=3)

for r in records:
    if r["mode"] != "remask" or r["strategy"] != "logit_diff":
        continue
    alpha = tau_alpha("logit_diff", r["tau"])
    color = (0.15, 0.55, 0.20, alpha)
    ax.scatter(r["token_mods"], r["acc"], c=[color], marker='D', s=55,
               edgecolors='none', zorder=3)

orig = [r for r in records if r["mode"] == "original"][0]
ax.scatter(orig["token_mods"], orig["acc"], c='black', marker='*', s=180, zorder=5)

legend_elements = [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markersize=14, label='Original (T2T edits)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.12, 0.38, 0.75),
               markersize=9, label='LowProb (remasks)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=(0.93, 0.60, 0.15),
               markersize=9, label='T2T-Remask (remasks)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=(0.15, 0.55, 0.20),
               markersize=8, label='LogitDiff (remasks)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.12, 0.38, 0.75, 0.35),
               markersize=9, label='τ small (lighter)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.12, 0.38, 0.75, 1.0),
               markersize=9, label='τ large (darker)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5,
          framealpha=0.9, edgecolor='gray')

strategy_label = {"low_prob": "LowProb", "t2t_remask": "T2T-Remask", "logit_diff": "LogitDiff"}
remasks = [r for r in records if r["mode"] == "remask"]

best = max(remasks, key=lambda r: (r["acc"], -r["token_mods"]))
ax.annotate(
    f'Best: {strategy_label[best["strategy"]]} τ={best["tau"]} C={best["c_max"]} ρ={best["rho"]}\n'
    f'{best["acc"]:.1f}%, {best["token_mods"]:.1f} remasks, {best["avg_fwd"]:.0f} fwd',
    xy=(best["token_mods"], best["acc"]),
    xytext=(620, 93.5), fontsize=8,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', edgecolor='red', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
    color='red', fontweight='bold'
)

eff = max((r for r in remasks if r["token_mods"] < 100),
          key=lambda r: (r["acc"], -r["token_mods"]))
ax.annotate(
    f'Most efficient: {strategy_label[eff["strategy"]]} τ={eff["tau"]} C={eff["c_max"]} ρ={eff["rho"]}\n'
    f'{eff["acc"]:.1f}%, {eff["token_mods"]:.1f} remasks, {eff["avg_fwd"]:.0f} fwd',
    xy=(eff["token_mods"], eff["acc"]),
    xytext=(280, 93.2), fontsize=8,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0F0FF', edgecolor='blue', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
    color='blue', fontweight='bold'
)

orig_acc = orig["acc"]
cheap = min((r for r in remasks if r["acc"] > orig_acc), key=lambda r: r["token_mods"])
ax.annotate(
    f'Cheapest: {strategy_label[cheap["strategy"]]} τ={cheap["tau"]} C={cheap["c_max"]} ρ={cheap["rho"]}\n'
    f'{cheap["acc"]:.1f}%, {cheap["token_mods"]:.1f} remasks, {cheap["avg_fwd"]:.0f} fwd',
    xy=(cheap["token_mods"], cheap["acc"]),
    xytext=(280, 84.8), fontsize=8,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE8D0', edgecolor='orangered', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='orangered', lw=1.2),
    color='orangered', fontweight='bold'
)

ax.annotate(
    f'Original (T2T): τ_edit={orig["editing_threshold"]}\n'
    f'{orig["acc"]:.1f}%, {orig["token_mods"]:.1f} edits, {orig["avg_fwd"]:.0f} fwd',
    xy=(orig["token_mods"], orig["acc"]),
    xytext=(60, 80.5), fontsize=8,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='gray', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
    color='black', fontweight='bold'
)

all_out_tok = [r["avg_output_tokens"] for r in records]
mean_out_tok = np.mean(all_out_tok)
ax.axvline(x=mean_out_tok, color='magenta', linestyle=':', linewidth=1, alpha=0.6)
ax.text(mean_out_tok + 5, 80.5, f'avg output tokens = {mean_out_tok:.0f}',
        color='magenta', fontsize=8, alpha=0.8)

ax.set_xlabel('Average Token Modifications per Sample (T2T edits / T2M remasks)',
              fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Token Modification Cost vs. Accuracy (CMATH 100 samples)',
             fontsize=13, fontweight='bold')

ax.set_ylim(79.5, 96)
ax.set_xlim(-20, 1500)
ax.yaxis.set_major_locator(plt.MultipleLocator(2))
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
for ext in ("pdf", "png"):
    out = f"paper/figures/ablation_tradeoff.{ext}"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved to {out}")
plt.close()
