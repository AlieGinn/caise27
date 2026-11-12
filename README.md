# Artifact

Offline materials for the routing study. The live application, the vector store, and raw logs are not included. Masked part numbers (for example `021XBXXXXXXF04`) stay in the queries.

`id`, `gold`, and `final` in the JSONL files are the published values. `code/score_answers.py` rescores a JSONL offline. In this copy the refusal sentence and the scorer pattern both read `system guidelines`, so a rescore matches the published checklist:

```text
python code/score_answers.py --input results/benchmark_2026-09-08_production_qwen3_14b.jsonl
```

`code/run_rq2b_followthrough.py` is the §5.2b protocol. `--run-14b` calls the live system and does not run in this repository.

## Data and code

| Path | Paper |
|---|---|
| `data/benchmark_routing_dataset.csv` | §4.1 official eval (`Eval_split=eval`, *n*=220). Three few-shot pool ids remain in the CSV and are not scored (§4.2). |
| `data/answer_keys_118.json` | §4.1 written answer key for the checklist. |
| `code/score_answers.py` | Offline checklist. Same JSONL and the same key give the same score. |
| `code/run_rq2b_followthrough.py` | §5.2b oracle follow-through. Does not run here. |

## Published runs

Wave-1 JSONL is the published A–D run (2026-09-08, system C on 2026-09-09), *n*=127 including the three pool ids. Wave-2 JSONL adds the 96 locked items (2026-09-21). Confirmatory counts use the merged scored intersection (§4.1, §5).

| System | Wave-1 | Wave-2 |
|---|---|---|
| A monolithic | `results/benchmark_2026-09-08_fat_agent_qwen3_14b.jsonl` | `results/benchmark_2026-09-21_fat_agent_qwen3_14b_wave2.jsonl` |
| B decoder-only | `results/benchmark_2026-09-08_decoder_only_qwen3_14b.jsonl` | `results/benchmark_2026-09-21_decoder_only_qwen3_14b_wave2.jsonl` |
| C prior, no card | `results/benchmark_2026-09-09_decoder_led_qwen3_14b.jsonl` | `results/benchmark_2026-09-21_decoder_led_qwen3_14b_wave2.jsonl` |
| D production | `results/benchmark_2026-09-08_production_qwen3_14b.jsonl` | `results/benchmark_2026-09-21_production_qwen3_14b_wave2.jsonl` |

| Table | Paper |
|---|---|
| `results/paper_mcnemar_primary.csv` | §5.1–§5.4 confirmatory paired tests |
| `results/paper_mcnemar_C_vs_D.csv` | §5.3 card (C vs D) |
| `results/paper_mcnemar_C_vs_B.csv` | §5.4 prior (B vs C) |
| `results/paper_mcnemar_n124.csv` | Wave-1-only *n*=124 row (§4.1, §5) |
| `results/paper_mcnemar_n127.csv` | Historical *n*=127 row (§4.1) |
| `results/paper_mcnemar_n120.csv` | Near-paraphrase footnote (§5.5) |
| `results/paper_mcnemar_kappa45.csv` | §6.2 gold-noise check on the 45-id slice |
| `results/paper_rq2b_followthrough.csv` | §5.2b owned-card recovery (4/5) |
| `results/paper_owned_cards.csv` | Owned-card list |
| `results/paper_failure_mode.csv` | Failure modes |
| `results/paper_leakage.csv` | §4.2 and §5.5 exact overlap 0; four near paraphrases |
| `results/paper_eval_coverage.csv` | Scored intersection *n*=220 |
| `results/paper_gold_mix.csv` | §4.1 gold mix |
| `results/paper_confusion_*.csv` | Confusion matrices for A–D |
| `results/paper_layered_errors.csv`, `results/paper_stratified_mcnemar.csv`, `results/paper_table3_ablation.csv`, `results/paper_paired_mcnemar.csv` | Layered errors, stratified contrasts, ablation table |
| `results/paper_latency_summary.csv` | End-to-end seconds footnote. Not the §5.7 prefix result. |

## Replication, annotation, timing, encoder

| Path | Paper |
|---|---|
| `replication/benchmark_production_2026-10-15.jsonl` | §5.8 production repeat (started 2026-10-15). §5.1–§5.4 cells stay the published files above. |
| `annotation/annotation_protocol.md` | Second-annotator protocol (§4.1, §6.2). |
| `annotation/annotation_kappa.csv` | 45-id sheet, κ=0.944. Not overwritten. |
| `annotation/annotation_sample_blind.csv` | 45-id blind sheet. |
| `annotation/annotation_sample_blind_wave2.csv` | 54-id expanded sample from the *n*=220 mix. Unlabeled. |
| `timing/kaggle_router_prefix.md` | §5.7 prefix protocol. Paper T4 percentiles: prior 2.4 / 2.8 ms; decoder-only 1.5B 976.9 / 1081.3; prefix 984.5 / 1092.4 (*n*=127). |
| `encoder/bert-tiny/` | §3.1 released two-head prior (`prajjwal1/bert-tiny`: `config.json`, `model.safetensors`, tokenizer). Not a 5-way route classifier. |
