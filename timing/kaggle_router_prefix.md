# Kaggle router prefix (milliseconds, no 14B)

Public re-run of the **router prefix only**. Flask, SimPy, clarification cards, and the 14B expert are **absent**. This is not an end-to-end latency study and not a speedup versus fat.

Notebook: [`kaggle_router_prefix.ipynb`](kaggle_router_prefix.ipynb). Local twin: `python tools/kaggle_router_prefix.py`.

## What is measured

| Quantity | Unit | Notes |
|---|---|---|
| bert-tiny encode | ms | `prajjwal1/bert-tiny` (Turc et al.; not TinyBERT/Jiao) from checkpoint `encoder/bert-tiny` |
| 1.5B JSON decode | ms | HuggingFace `Qwen/Qwen2.5-1.5B-Instruct`, greedy (T=0), short JSON |
| Prefix total | ms | bert-tiny + 1.5B **with** prior (system D/C prefix) |
| Decoder-only 1.5B | ms | Same 1.5B, **no** bert-tiny prior (system B prefix) |

Comparison in the paper: **decoder-only 1.5B vs bert-tiny+1.5B** (bert-tiny’s own add-on). Never vs 14B. Never laptop JSONL seconds.

HuggingFace 1.5B and production Ollama `qwen2.5:1.5b` are the **same family, not bit-equal**. Report prefix *order of magnitude* (p50/p95 ms). Do not claim numerical identity with the laptop decoder.

## Kaggle setup

Kaggle mounts (username **yok**):

- Working: `/kaggle/working/kaggle_router_prefix.csv`
- Dataset: `/kaggle/input/test-for-time/benchmark_routing_dataset.csv`
- Model: `/kaggle/input/tinybert8/pytorch/default/1/` (`config.json` + `model.safetensors`; klasör iç içe olabilir)

**Kullanma:** `/kaggle/input/datasets/[redacted-path]/...` ve notebook hücresinde `Path(__file__)`.

1. Add Data: CSV dataset + bert-tiny model (klasörün tamamı: config, safetensors, tokenizer).
2. GPU + Internet. Qwen2.5-1.5B-Instruct HuggingFace’den iner (Qwen3 yok).
3. `tools/kaggle_router_prefix.py` içeriğini bir hücreye yapıştır, sonra `kaggle_run()`.
4. Seed 42; 3 discarded warmup forwards; percentiles on *n*=127 (same 127 ids, including S01–S03); CUDA ise `synchronize`.

## Protocol

- `max_length` 128 for bert-tiny; sigmoid two-head scores (not 5-way softmax).
- Decoder: chat template, `do_sample=False`, `max_new_tokens=48`, JSON object only.
- Warmup queries are the first three CSV rows, run on both arms as discarded forwards; percentiles use the same *n*=127 ids (including those three), timed after warmup.
- Output CSV: `ablation_baselines/results/kaggle_router_prefix.csv` (download from Kaggle and commit). Columns include `tinybert_ms`, `decoder_only_ms`, `decoder_with_prior_ms`, `prefix_total_ms`, `environment`, `gpu_name`.

## Fill-in after the notebook run

| Field | Kaggle HF (paper table) | Local Ollama (engineering footnote) |
|---|---|---|
| GPU | Tesla T4 | torch=cpu; Ollama uses the machine GPU |
| Seed | 42 | 42 |
| Warmup | 3 | 3 |
| *n* timed | 127 | 127 |
| bert-tiny p50 / p95 ms | 2.4 / 2.8 | 5.0 / 10.9 |
| Decoder-only 1.5B p50 / p95 ms | 977 / 1081 | 374.1 / 496.9 |
| bert-tiny+1.5B prefix p50 / p95 ms | 984 / 1092 | 526.8 / 627.1 |
| Decoder backend | HuggingFace `Qwen/Qwen2.5-1.5B-Instruct` | `ollama:qwen2.5:1.5b` |
| environment column | `kaggle_hf` | `local_ollama` |

Paper table = Kaggle HF (`n`=127, seed 42, `decoder_backend=Qwen/Qwen2.5-1.5B-Instruct`). Local Ollama is a **same-family, not bit-equal** footnote; do not treat laptop 14B seconds as this measurement. bert-tiny is a few milliseconds; 1.5B JSON is ~1 s; the 14B expert remains the shared cost. bert-tiny’s own add-on is prefix − decoder-only (p50 ~13 ms on T4) — a small constraint, not a fat-agent speedup. A negative mean add-on is GPU noise / decode-length variation; it is not evidence that bert-tiny speeds up decode. Empty `decoder_*_route` cells (13 and 20 rows) are not routing quality and are unused in the paper; only the timing columns are.

## Not-claims

- Not faster than the fat 14B agent.
- Not 544× / energy.
- Not a 14B replicate.
- bert-tiny is not claimed to beat the 1.5B as a generator; we only time the add-on.
