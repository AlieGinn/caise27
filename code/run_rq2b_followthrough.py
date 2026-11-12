"""RQ2b: owned-card follow-through with oracle gold family (no gold-OOD force).

Primary RQ2 remains first-publish checklist. This script does not McNemar.
Expected recovery: 4/5 of the wave-1 owned cards (A29, AH07, T01, P02);
S05 may remain a claim/route miss.

  python code/run_rq2b_followthrough.py
  python code/run_rq2b_followthrough.py --run-14b

--run-14b calls the live system and does not run in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = _PROJECT_ROOT / "results"
KEYS = _PROJECT_ROOT / "data" / "answer_keys_118.json"
OWNED_CSV = RESULTS / "paper_owned_cards.csv"
WAVE1_PROD = RESULTS / "benchmark_2026-09-08_production_qwen3_14b.jsonl"
WAVE1_PROD_CONTENT = RESULTS / "benchmark_2026-09-08_production_qwen3_14b_content_summary.csv"
DEFAULT_OUT = RESULTS / "benchmark_2026-09-21_rq2b_owned_followthrough.jsonl"

GOLD_TO_CARD = {
    "static": "Technical context",
    "analyst": "Analysis",
    "analyst_hybrid": "Analysis + Technical context",
    "operator": "Operator",
    "operator_hybrid": "Operator + Technical context",
}


def _load_owned(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _index_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {(r.get("id") or "").strip(): r for r in csv.DictReader(fh)}


def _index_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("id") or "")
            if qid:
                out[qid] = rec
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_table(
    *,
    owned: list[dict[str, str]],
    follow_content: dict[str, dict[str, str]],
    first_publish: dict[str, dict[str, str]],
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in owned:
        qid = (rec.get("id") or "").strip()
        gold = (rec.get("gold") or "").strip()
        if gold == "ood":
            continue
        first = first_publish.get(qid) or {}
        follow = follow_content.get(qid) or {}
        b_check = _as_bool(rec.get("b_checklist"))
        rows.append({
            "id": qid,
            "gold": gold,
            "tag": rec.get("tag") or "",
            "oracle_card": GOLD_TO_CARD.get(gold, ""),
            "d_first_final": rec.get("d_final") or "ood",
            "d_first_checklist": first.get("binary_pass") or "0",
            "b_checklist": rec.get("b_checklist") or "",
            "c_checklist": rec.get("c_checklist") or "",
            "follow_final": follow.get("final") or "",
            "follow_route_match": follow.get("route_match") or "",
            "follow_checklist": follow.get("binary_pass") or "",
            "follow_failure_mode": follow.get("failure_mode") or "",
            "recovered": int(_as_bool(follow.get("binary_pass"))),
            "expected_from_b": int(b_check),
            "query": rec.get("query") or "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    recovered = sum(int(r["recovered"]) for r in rows)
    expected = sum(int(r["expected_from_b"]) for r in rows)
    print(
        f"RQ2b owned in-scope n={len(rows)}; checklist recovered "
        f"{recovered}/{len(rows)}; B already passed {expected}/{len(rows)}",
        file=sys.stderr,
    )
    print(f"wrote {path}", file=sys.stderr)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RQ2b owned-card oracle follow-through")
    parser.add_argument("--run-14b", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--owned", type=Path, default=OWNED_CSV)
    parser.add_argument("--session-id", default="paper-rq2b-followthrough")
    args = parser.parse_args(argv)

    owned = [
        r for r in _load_owned(args.owned)
        if (r.get("gold") or "").strip() != "ood"
    ]
    if not owned:
        print(f"owned-card CSV yok veya boş: {args.owned}", file=sys.stderr)
        return 1
    ids = [r["id"] for r in owned]
    print(
        f"RQ2b ids={','.join(ids)} (gold-OOD follow-through yok)",
        file=sys.stderr,
    )

    if args.run_14b:
        from ablation_baselines.run_ablation_benchmark import run_benchmark

        rc = run_benchmark(
            dataset_path=_PROJECT_ROOT / "data" / "benchmark_routing_dataset.csv",
            output_path=args.output,
            architectures=["production"],
            resume=True,
            session_id=args.session_id,
            limit=None,
            eval_only=False,
            keep_ids=set(ids),
            route_override_gold=True,
        )
        if rc != 0:
            return rc
        if args.output.exists():
            from score_answers import main as score_main
            score_main(["--input", str(args.output), "--keys", str(KEYS)])
    else:
        print(
            "--run-14b yok: mevcut follow-through JSONL skorlanır (yoksa tablo boş follow kolonları)",
            file=sys.stderr,
        )
        if args.output.exists():
            from score_answers import main as score_main
            score_main(["--input", str(args.output), "--keys", str(KEYS)])

    follow_csv = args.output.with_name(args.output.stem + "_content_summary.csv")
    follow_content = _index_csv(follow_csv)
    first_publish = _index_csv(WAVE1_PROD_CONTENT)
    table_path = RESULTS / "paper_rq2b_followthrough.csv"
    write_table(
        owned=owned,
        follow_content=follow_content,
        first_publish=first_publish,
        path=table_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
