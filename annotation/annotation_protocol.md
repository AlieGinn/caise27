# Second-annotator protocol (do not rewrite official gold)

## Purpose

Gold `Expected_route` on the official eval (`Eval_split=eval`, *n*=220) is a **single annotator** plus a written answer key. This protocol adds a **second independent label** on a stratified slice so we can report Cohen’s κ. It does **not** change the gold CSV, the 2026-09-08 JSONL files, or RQ1–RQ2 numerators.

The original 45-id wave-1 sheets (`annotation_sample_blind.csv`, κ=0.944) are **not overwritten**. A new 54-id sample from the expanded mix is written to `annotation_sample_blind_wave2.csv`.

If κ is low, the limitation section is strengthened. Labels are not bulk-replaced.

## Codebook (same as production routes)

The annotator assigns exactly one of:

| Label | Use when |
|---|---|
| `static` | Documentation, formula, screen how-to, documented general effect. No live instance readout, no new scenario. |
| `analyst` | Live twin state (dashboard, named part, warehouse, last-run KPI) or arithmetic on given figures. No lever change. |
| `analyst_hybrid` | Live instance **and** documentation/formulas/targets must both be used. |
| `operator` | New simulated world: run / replay / restart / advance, or a controllable lever (shock, seed, horizon, lot size, …). |
| `operator_hybrid` | Operator action **plus** an explicit request to interpret new results against documentation. |
| `ood` | Out of scope, capability tease without execution, or genuinely unresolvable without asking the user which flow. |

Priority (same order as `DECODER_SYSTEM_PROMPT`): (1) execution or lever now → operator family; (2) else live instance / given-figure arithmetic → analyst family; (3) else documented definition → static. Hybrid only when documentation grounding is explicitly required.

Do **not** use model traces, TinyBERT scores, or existing gold while labeling.

## Sampling (reproducible)

```text
python tools/paper_sensitivity.py --write-annotation-sample
```

- Target **54** ids (range 45–60) from `Eval_split=eval` only.
- Seed **42**.
- Stratify on `(Expected_route, Tag)` cells in the official eval. Every non-empty cell contributes at least one id. Remaining slots are filled in proportion to cell size without replacement.
- The **blind** sheet has no gold column. The key sheet is for scoring after labels exist.
- Do not replace `annotation_sample_blind.csv` (legacy 45-id, κ=0.944).

Outputs:

- `docs/paper/annotation_sample_blind_wave2.csv` — give this to annotator 2
- `docs/paper/annotation_sample_key_wave2.csv` — gold + sample metadata; not shown during labeling

## Independent labeling

1. Annotator 2 sees only the blind CSV (`id`, `query`, `tag`, `category`, empty `annotator2_route`, `notes`).
2. Fill `annotator2_route` with one codebook label. Optional `notes` for borderline items.
3. No discussion with annotator 1 until the sheet is complete.
4. Do not look up production/decoder/fat JSONL.

Annotator 1 is the existing gold. There is no re-label of annotator 1 on this slice.

## Cohen’s κ (nominal, 6 classes)

Let *N* be the slice size (45 for the dual-coded wave-1 sheet; 54 for the expanded unlabeled sample). Let *n_{ij}* be the count of items labeled *i* by annotator 1 (gold) and *j* by annotator 2.

Observed agreement:

\[
p_o = \frac{1}{N}\sum_i n_{ii}
\]

Expected agreement (marginal independence):

\[
p_e = \sum_i p_{i\cdot}\,p_{\cdot i},\quad
p_{i\cdot}=\frac{1}{N}\sum_j n_{ij},\quad
p_{\cdot i}=\frac{1}{N}\sum_j n_{ji}
\]

\[
\kappa = \frac{p_o - p_e}{1 - p_e}
\]

After annotator 2 finishes:

```text
python tools/paper_sensitivity.py --score-kappa docs/paper/annotation_sample_blind.csv
```

The script writes `docs/paper/annotation_kappa.csv` and refuses to modify the gold CSV. The 45-id wave-1 sheet remains the dual-coded κ=0.944 source.

Gold-only 45-id silent-safe (already computable): 38/45 vs 33/45, b=5, c=0, p=0.062 — direction matches the headline; the slice does not reverse p<0.001.

### Template (fill after labeling)

| Quantity | Value |
|---|---|
| *N* | 45 |
| *p_o* | 0.9556 (43/45) |
| *p_e* | 0.2010 |
| Cohen κ | 0.9444 |
| Disagreements | 2: O11 (gold `ood` ≠ a2 `operator`); A24 (gold `static` ≠ a2 `ood`) |
| Adjudicated (subset only) | O11 → `ood`; A24 → `static` (appendix-only) |
| Official gold CSV changed by dual-coding? | **No** |
| 45-id silent-safe McNemar (gold) | 38/45 vs 33/45; b=5, c=0; p=0.062 (direction holds; does not reverse headline) |
| 45-id silent-safe McNemar (annotator 2) | 38/45 vs 33/45; b=5, c=0; p=0.062 (does not reverse headline) |

Interpretation (Landis & Koch, descriptive only): κ<0.40 poor; 0.40–0.60 moderate; 0.60–0.80 substantial; >0.80 almost perfect. Low κ → stronger limitation, not a license to edit gold.

## Adjudication (subset only)

1. List ids with `gold ≠ annotator2_route`.
2. A third reader (or annotator 1+2 together) writes a one-line rationale.
3. Record the **adjudicated** label in `annotation_kappa.csv` (`adjudicated_route`). This column is **appendix-only**.
4. Do **not** copy adjudicated labels back into `benchmark_routing_dataset.csv`.

## What this protocol is not

- Not a replacement of RQ2 (checklist).
- Not a pairwise preference study (the llama3.1 *n*=26 sheet stays exploratory).
- Not a TinyBERT 5-way labeling task. Gold is the 5 routes + `ood`.
