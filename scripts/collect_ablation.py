"""Collect ablation results and print summary tables."""
import json, os, glob

base = "results_v2/ablation"
summaries = {}
for f in sorted(glob.glob(os.path.join(base, "*/summary.json"))):
    with open(f) as fh:
        s = json.load(fh)
    summaries[s["tag"]] = s

# Rebuild config list with tags
configs = []
cid = 0
configs.append((f"{cid:03d}_original", "original", "", "", 3, 0.25)); cid += 1
for strategy, taus in [("low_prob", [0.1,0.3,0.5,0.7,0.9]),
                        ("t2t_remask", [0.5,0.7,0.9]),
                        ("logit_diff", [0.1,0.2,0.3,0.5])]:
    for tau in taus:
        for c_max in [1, 3, 5]:
            for rho in [0.25, 0.50, 1.0]:
                tag = f"{cid:03d}_{strategy}_t{tau}_c{c_max}_r{rho}"
                configs.append((tag, "remask", strategy, str(tau), c_max, rho)); cid += 1

done = sum(1 for tag, *_ in configs if tag in summaries)
print(f"Completed: {done}/{len(configs)}\n")

print(f"{'Tag':>45s} {'Acc%':>6s} {'Correct':>7s}")
print("-" * 62)
for tag, mode, strategy, tau, c, rho in configs:
    if tag in summaries:
        s = summaries[tag]
        print(f"{tag:>45s} {s['accuracy']*100:6.1f} {s['correct']:>4d}/{s['total']}")
    else:
        print(f"{tag:>45s} {'PEND':>6s}")

# Best per strategy
print("\n=== Best per strategy ===")
for strat in ["low_prob", "t2t_remask", "logit_diff"]:
    best_tag, best_acc = None, 0
    for tag, mode, strategy, tau, c, rho in configs:
        if strategy == strat and tag in summaries:
            acc = summaries[tag]["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_tag = tag
    if best_tag:
        print(f"  {strat:>12s}: {best_tag} -> {best_acc*100:.1f}%")

with open(os.path.join(base, "all_results.json"), "w") as f:
    json.dump([summaries[tag] for tag, *_ in configs if tag in summaries], f, indent=2)
print(f"\nSaved to {base}/all_results.json")
