# LLaDA2.1 Remask Inference

Replace T2T editing (token→token) with T2M remasking (token→mask) in LLaDA2.1's `generate()`.

If a committed token's self-probability `P(current_token | context)` falls below a threshold,
it is set back to `<|mask|>` and re-predicted by the diffusion process from scratch.

## Project structure

```
remask/                     core package
  modeling_llada2_moe.py    copied from original, T2T extracted into _token_edit_step()
  modeling_remask.py        subclass: overrides _token_edit_step() with T2M remasking
  configuration_llada2_moe.py
  loader.py                 load_remask_model / load_original_model
  utils.py                  prompt formatting, answer extraction, QA metrics
eval/                       benchmark scripts (all 33 from LLaDA2.1 paper)
  gsm_plus.py               GSM-Plus  (2400 test)
  cmath.py                  CMATH     (1098 test)
  aime2025.py               AIME 2025 (30 problems)
  olympiadbench.py          OlympiadBench (674 test)
  omni_math.py              Omni-MATH (4428 test)
  mmlu_pro.py               MMLU-Pro  (12032 test)
  gpqa.py                   GPQA-Diamond (198, gated)
  ceval.py                  C-Eval    (val split)
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
  humaneval.py              HumanEval+ (164 problems)
  mbpp.py                   MBPP+     (378 problems)
  ifeval.py                 IFEval    (541, offline scoring)
  bfcl.py                   BFCL v3   (offline scoring)
data/                       cached datasets
results/                    full evaluation outputs
results_tune/               small-sample parameter tuning outputs
run_all_evals.sh            full benchmark runner
run_tune.sh                 small-sample parameter tuning
```

## Quick start

```bash
# single benchmark
python -m eval.gsm_plus --mode original --max_samples 20
python -m eval.gsm_plus --mode remask --strategy low_prob --remask_threshold 0.1 --max_samples 20

# parameter tuning (small samples across benchmarks)
bash run_tune.sh --max_samples 20

# full evaluation
bash run_all_evals.sh
```

## How it works

`remask/modeling_llada2_moe.py` copies the original model, extracting the T2T editing
block into `_token_edit_step()`. `remask/modeling_remask.py` subclasses it and only
overrides that one method with T2M remasking logic.

## Key parameters (aligned with official LLaDA2.1)

### Inference parameters (from official README)

| param                | Q Mode  | S Mode  | description                    |
|----------------------|---------|---------|--------------------------------|
| `threshold`          | 0.7     | 0.5     | M2T confidence threshold       |
| `editing_threshold`  | 0.5     | 0.0     | T2T editing confidence         |
| `max_post_steps`     | 16      | 16      | extra iterations after masks resolved |
| `block_length`       | 32      | 32      | tokens per block               |
| `steps`              | 32      | 32      | denoising steps per block      |
| `temperature`        | 0.0     | 0.0     | sampling temperature           |

### Remask-specific parameters

| param                | description                              | default       |
|----------------------|------------------------------------------|---------------|
| `remask_strategy`    | `low_prob` / `t2t_remask` / `logit_diff` | `low_prob`    |
| `remask_threshold`   | strategy-specific threshold              | per-strategy  |
| `max_remask_per_pos` | max remask count per position (防死锁)    | 3             |

Strategy defaults:
- `low_prob`: threshold=0.1 (remask if P(old_token) < 0.1)
- `t2t_remask`: threshold=0.9 (remask if P(new_token) > 0.9 and differs)
- `logit_diff`: threshold=0.3 (remask if |P_new - P_prev| > 0.3)

## Benchmark coverage (论文 33 个)

| Category | Benchmarks | Status |
|----------|-----------|--------|
| Knowledge | MMLU-Pro, GPQA-Diamond, C-Eval, PHYBench, TriviaQA | ✅ (GPQA需HF token) |
| Reasoning | BBH, BBH Extra Hard, bbh-zh, MuSR, ZebraLogic, PrOntoQA, PIQA, OCNLI, HellaSwag, KOR-Bench, DROP, SQuAD 2.0 | ✅ |
| Coding | HumanEval+, MBPP+, CRUXEval-O, MultiPL-E*, BigCodeBench*, LiveCodeBench*, Spider*, BIRD-SQL* | ✅ (带*需外部工具) |
| Math | GSM-Plus, CMATH, AIME 2025, OlympiadBench, Omni-MATH | ✅ |
| Agent | IFEval, BFCL v3*, Nexus FC* | ✅ (带*需离线评分) |
