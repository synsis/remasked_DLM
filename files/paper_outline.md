# Paper Outline & Writing Notes

> Paper: "Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models"
> Target: NeurIPS 2026 (9 pages + unlimited appendix/refs)
> Location: `paper/main.tex`

---

## Structure Overview

| Section | Pages | Status |
|---------|-------|--------|
| Abstract | 0.3 | Written |
| 1. Introduction | 1.5 | Written |
| 2. Background | 1.0 | Written |
| 3. Method | 1.5 | Written (with Algorithm 1) |
| 4. Theoretical Analysis | 2.0 | Written (4 subsections, 1 proposition, 1 remark) |
| 5. Related Work | 1.0 | Written |
| 6. Experiments | 2.0 | Written (tables empty, settings done) |
| 7. Conclusion | 0.5 | Written |
| **Total** | **~9.0** | |
| Appendix A: Benchmarks | -- | Written |
| Appendix B: Hyperparams | -- | Written |
| Appendix C: Case Studies | -- | Placeholder |
| Checklist | -- | Template included |

---

## Key Narrative / Story Arc

1. **Problem**: LLaDA2.1's T2T editing has fundamental issues (coupling, context pollution, distribution shift)
2. **Solution**: T2M remasking — reset to mask instead of replacing
3. **Theory**: Why mask > wrong token as context (4 axes: decoupling, purification, noise mismatch, delayed commitment)
4. **Practice**: Three strategies, training-free, drop-in replacement
5. **Differentiation**: We improve T2T editing specifically; ReMDM changes the whole sampling process; CORE needs extra forward passes; RemeDi/ProSeCo need training

---

## Reviewer Concerns to Preempt

### Q: How does this differ from ReMDM?
- ReMDM: modifies reverse posterior, random σ_t remask probability for ALL tokens at EVERY step
- Ours: targeted detection at the T2T editing step only, deterministic based on error signals
- Analogy: random re-exam vs targeted debugging

### Q: CORE criticizes low-prob heuristics as "myopic"
- Our LogitDiff strategy captures temporal dynamics (not just static probability)
- CORE requires additional perturbation forward passes (2-3x cost); we are zero-overhead
- We can acknowledge this in future work and suggest combining approaches

### Q: Why not run ReMDM/CORE as baselines?
- Different architectural level: ReMDM modifies sampling process; CORE works on base MDLM
- Neither targets LLaDA2.1's T2T editing mechanism specifically
- Fair comparison would require adapting them to LLaDA2.1's block-level generation, which is non-trivial
- We focus on controlled comparison: same model, same loop, only T2T step changed

### Q: Results don't match LLaDA2.1 paper numbers
- "LLaDA2.1 does not open-source its evaluation code" — stated in Section 6.1
- All comparisons use identical evaluation conditions
- Absolute numbers may differ but relative comparisons are valid

---

## TODO Before Submission

### Experiments to Run
- [ ] Full run_all_evals.sh with original + 3 strategies
- [ ] Fill all empty table cells
- [ ] Compute remask statistics (remasks/block, remasks/generation, time overhead)
- [ ] Safety mechanism ablation (C_max, ρ_max variations)
- [ ] Collect 2-3 case studies for Appendix C

### Paper Polish
- [ ] Fill in XX count in abstract ("XX benchmarks")
- [ ] Verify page count with compiled PDF (must be ≤ 9)
- [ ] Complete NeurIPS checklist (checklist.tex)
- [ ] Add figure: T2T vs T2M diagram (visual comparison)
- [ ] Add figure: accuracy vs threshold sensitivity plot
- [ ] Proofread all sections

### References
- [ ] Verify all bib entries have correct years, venues, author lists
- [ ] Replace placeholder author names (e.g., "RemeDi Authors") with real names
- [ ] Add any missing references from reviewer perspective

---

## Compilation

```bash
cd paper/
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Requires: `neurips_2026.sty` (copied), `checklist.tex` (copied), `references.bib` (written).
