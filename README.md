# LLaDA2.1 Targeted Remasking — Inference Codebase

This repository accompanies the paper **Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models**.

The PDF is at the repository root:

`Targeted_Remasking__Replacing_Token_Editing_with_Token_to_Mask_Refinement_in_Discrete_Diffusion_Language_Models.pdf`

A Chinese summary and outline are in `paper/paper_cn.md`.

## What this project does

For **LLaDA2.1** generation, we replace the official **Token-to-Token (T2T) editing** step (directly swapping a committed token for another) with the paper’s **Token-to-Mask (T2M) remasking**: when a committed token looks unreliable under the model, we do not substitute another token immediately—we reset that position to `<|mask|>` and let later diffusion iterations repredict it. This mitigates context pollution and train–inference noise mismatch. The code only overrides the T2T-equivalent step inside `generate()`.

**Intuition:** if the self-probability `P(current token | context)` falls below a threshold, remask that position and let the diffusion process fill it again.

## Project structure

```
remask/                     core package
  modeling_llada2_moe.py    fork from upstream; T2T isolated in _token_edit_step()
  modeling_remask.py        subclass: overrides _token_edit_step() with T2M remasking
  configuration_llada2_moe.py
  loader.py                 load_remask_model / load_original_model
  utils.py                  prompt formatting, answer extraction, QA metrics
eval/                       benchmark scripts (33 benchmarks aligned with LLaDA2.1 paper, etc.)
  gsm_plus.py               GSM-Plus  (2400 test)
  cmath.py                  CMATH     (1098 test)
  aime2025.py               AIME 2025 (30)
  olympiadbench.py          OlympiadBench (674 test)
  omni_math.py              Omni-MATH (4428 test)
  mmlu_pro.py               MMLU-Pro  (12032 test)
  gpqa.py                   GPQA-Diamond (198, gated)
  ceval.py                  C-Eval    (val)
  triviaqa.py               TriviaQA  (17944 val)
  phybench.py               PHYBench
  bbh.py                    BIG-Bench Hard (250 per task)
  bbh_extra_hard.py         BIG-Bench Extra Hard (4520)
  bbh_zh.py                 BBH Chinese
  hellaswag.py              HellaSwag (10042 val)
  piqa.py                   PIQA      (1838 test)
  prontoqa.py               PrOntoQA  (500 val)
  ocnli.py                  OCNLI     (2950 val)
  kor_bench.py              KOR-Bench
  drop.py                   DROP      (9535 val)
  squad2.py                 SQuAD 2.0 (11873 val)
  musr.py                   MuSR      (756 total)
  zebralogic.py             ZebraLogic
  cruxeval.py               CRUXEval-O
  humaneval.py              HumanEval+ (164)
  mbpp.py                   MBPP+     (378)
  ifeval.py                 IFEval    (541, offline scoring)
  bfcl.py                   BFCL v3   (offline scoring)
data/                       cached datasets
results/                    full evaluation outputs
results_tune/               small-sample tuning outputs
run_all_evals.sh            full benchmark runner
run_tune.sh                 small-sample hyperparameter search
scripts/                    cluster submit, ablations, single-run helpers
```

## Quick start

```bash
# single benchmark (original vs remask)
python -m eval.gsm_plus --mode original --max_samples 20
python -m eval.gsm_plus --mode remask --strategy low_prob --remask_threshold 0.1 --max_samples 20

# small-sample tuning across benchmarks
bash run_tune.sh --max_samples 20

# full evaluation
bash run_all_evals.sh
```

## Implementation

`remask/modeling_llada2_moe.py` keeps upstream behavior but factors the T2T block into `_token_edit_step()`. `remask/modeling_remask.py` subclasses it and **only overrides** that method with T2M logic; scheduling and the rest of generation match the official stack.

## Key parameters (aligned with official LLaDA2.1)

### Inference parameters (from official README)

| Parameter           | Q Mode | S Mode | Description                          |
|---------------------|--------|--------|--------------------------------------|
| `threshold`         | 0.7    | 0.5    | M2T confidence threshold             |
| `editing_threshold` | 0.5    | 0.0    | T2T editing confidence (original path) |
| `max_post_steps`    | 16     | 16     | Extra iterations after masks clear   |
| `block_length`      | 32     | 32     | Tokens per block                     |
| `steps`             | 32     | 32     | Denoising steps per block            |
| `temperature`       | 0.0    | 0.0    | Sampling temperature                 |

### Remask-specific parameters

| Parameter            | Description                               | Default      |
|----------------------|-------------------------------------------|--------------|
| `remask_strategy`    | `low_prob` / `t2t_remask` / `logit_diff`  | `low_prob`   |
| `remask_threshold`   | Strategy-specific threshold               | per strategy |
| `max_remask_per_pos` | Max remasks per position (avoid deadlock) | 3            |

Strategy defaults (see code for exact values):

- `low_prob`: remask if `P(old token) <` threshold (default threshold 0.1)
- `t2t_remask`: on a T2T-like trigger, remask instead of replace (e.g. default 0.9)
- `logit_diff`: remask if `|P_new − P_prev|` exceeds threshold (default 0.3)

## Benchmark coverage (33 in paper)

| Category | Benchmarks | Notes |
|----------|------------|-------|
| Knowledge | MMLU-Pro, GPQA-Diamond, C-Eval, PHYBench, TriviaQA | GPQA needs HF token |
| Reasoning | BBH, BBH Extra Hard, bbh-zh, MuSR, ZebraLogic, PrOntoQA, PIQA, OCNLI, HellaSwag, KOR-Bench, DROP, SQuAD 2.0 | |
| Coding | HumanEval+, MBPP+, CRUXEval-O, MultiPL-E*, etc. | * requires extra tooling |
| Math | GSM-Plus, CMATH, AIME 2025, OlympiadBench, Omni-MATH | |
| Agent | IFEval, BFCL v3*, etc. | * often offline scoring |

## Citation

If you use this code or the method, please cite *Targeted Remasking* (use the BibTeX from the published version when available).
