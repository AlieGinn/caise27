"""Çevrimdışı cevap checklist skoru — yargıç model yok.

Aynı JSONL + aynı answer_keys_118.json → aynı skor.
Chatbot dökümü yeniden koşulmaz.

  python code/score_answers.py --input results/benchmark_2026-09-08_production_qwen3_14b.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KEYS = _PROJECT_ROOT / "data" / "answer_keys_118.json"
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "benchmark_routing_dataset.csv"
_DEFAULT_COMPARE = (
    _PROJECT_ROOT / "results" / "production_p7_tb8_answers_118_summary.csv"
)
_DEFAULT_COMPARE_CONTENT = (
    _PROJECT_ROOT / "results" / "production_p7_tb8_answers_118_content_summary.csv"
)
_DEFAULT_INPUT = (
    _PROJECT_ROOT / "results" / "benchmark_2026-09-08_production_qwen3_14b.jsonl"
)

OOD_CARD_RE = re.compile(
    r"hangi akışta ele alınacağını aşağıdan seçin|yetki yok|kapsam dışı",
    re.IGNORECASE,
)
RAG_REFUSE_RE = re.compile(
    r"this information is not available in the system guidelines",
    re.IGNORECASE,
)
LEVER_REFUSE_RE = re.compile(
    r"lever is not supported|not supported|desteklenmiyor|"
    r"unsupported (?:lever|action|feature)|"
    r"cannot (?:export|toggle|reset|trigger|recalibrate)",
    re.IGNORECASE,
)
OPERATOR_GOLD = frozenset({"operator", "operator_hybrid"})
READ_FAMILIES = frozenset({"analyst", "analyst_hybrid", "static"})
LIVE_KPI_HALLUCINATION_RE = re.compile(
    r"(?:dashboard|on[- ]screen|current(?:ly)?|live|shown)\b.{0,80}"
    r"(?:service[_\s-]?level|stockout|backlog|inventory|holding[_\s-]?cost|"
    r"p90|fulfillment|fill[_\s-]?rate).{0,40}"
    r"(?:\d+\.\d+|\d+\s*%|\d+\s*days?)",
    re.IGNORECASE | re.DOTALL,
)
MISSING_KPI_RE = re.compile(
    r"no live|not (?:available|shown|present)|empty kpis?|without (?:live |dashboard )?"
    r"kpi|kpi(?:s)? (?:are |is )?not|no (?:dashboard|on-screen) (?:data|kpi)",
    re.IGNORECASE,
)

AUDIT_IDS = (
    "S02", "S01", "S06", "S10",
    "O01", "O12",
    "X01", "X07",
    "A13", "AH07", "OH01", "A17",
)

FAMILY_ORDER = (
    "static",
    "analyst",
    "operator",
    "analyst_hybrid",
    "operator_hybrid",
    "ood",
)
CORE_SCORE_TAGS = frozenset({"core", "clarified"})
TAG_ORDER = (
    "core",
    "clarified",
    "trap",
    "probe_conflict",
    "probe_ood",
    "probe_hybrid",
)
CLARIFIED_FOCUS_IDS = (
    "S10", "S19", "S05", "S04", "S22", "S12",
    "A16", "A17", "A27", "A26",
    "A06", "A09", "A13", "A23", "A29", "O16",
    "AH03", "AH04", "AH05", "AH06", "AH07", "AH08",
    "OH04",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _load_tags(dataset: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    if not dataset.exists():
        return tags
    with dataset.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            qid = (row.get("Id") or "").strip()
            if not qid:
                continue
            tag = (row.get("Tag") or "core").strip().lower() or "core"
            tags[qid] = tag
    return tags


def _as_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _haystack(rec: dict[str, Any]) -> str:
    parts = [
        str(rec.get("answer") or rec.get("llm_response") or ""),
        json.dumps(rec.get("pending_action"), ensure_ascii=False, default=str),
        str(rec.get("operator_status") or ""),
    ]
    return "\n".join(parts)


def _search(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
    except re.error:
        return pattern.lower() in (text or "").lower()


def _claim_hit(pattern: str, text: str) -> bool:
    return _search(pattern, text)


def _is_ood_card(text: str, rec: dict[str, Any]) -> bool:
    if rec.get("ood_card"):
        return True
    return bool(OOD_CARD_RE.search(text or ""))


def _is_rag_refuse(text: str) -> bool:
    stripped = (text or "").strip()
    if RAG_REFUSE_RE.fullmatch(stripped):
        return True
    return bool(RAG_REFUSE_RE.search(stripped)) and len(stripped) < 120


def _is_content_refuse(text: str) -> bool:
    if _is_rag_refuse(text):
        return True
    return bool(LEVER_REFUSE_RE.search(text or ""))


def _pending_kind(rec: dict[str, Any]) -> Optional[str]:
    pending = rec.get("pending_action")
    if isinstance(pending, dict) and pending.get("kind"):
        return str(pending.get("kind"))
    hay = _haystack(rec)
    m = re.search(
        r"Proposed action:\s*(run_fresh|replay|advance)",
        hay,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def _kind_ok(expected: Any, actual: Optional[str]) -> bool:
    if not expected:
        return True
    if actual is None:
        return False
    actual_n = str(actual).strip().lower()
    if isinstance(expected, (list, tuple)):
        return actual_n in {str(x).strip().lower() for x in expected}
    return actual_n == str(expected).strip().lower()


def _family(gold: str) -> str:
    g = (gold or "").strip()
    if g in FAMILY_ORDER:
        return g
    return "ood" if g == "ood" else g or "unknown"


def _evaluate_claims(
    spec: dict[str, Any],
    text: str,
) -> tuple[int, int, list[dict[str, Any]], bool]:
    details: list[dict[str, Any]] = []
    hit = 0
    total = 0
    forbid_hit = False

    for pat in spec.get("must") or []:
        total += 1
        ok = _claim_hit(str(pat), text)
        details.append({"kind": "must", "pattern": pat, "hit": ok})
        if ok:
            hit += 1

    for group in spec.get("must_any") or []:
        total += 1
        patterns = [str(p) for p in group]
        ok = any(_claim_hit(p, text) for p in patterns)
        details.append({"kind": "must_any", "pattern": patterns, "hit": ok})
        if ok:
            hit += 1

    for pat in spec.get("forbid") or []:
        bad = _claim_hit(str(pat), text)
        details.append({"kind": "forbid", "pattern": pat, "hit": not bad})
        if bad:
            forbid_hit = True

    for pat in spec.get("pending_contains") or []:
        total += 1
        ok = _claim_hit(str(pat), text)
        details.append({"kind": "pending_contains", "pattern": pat, "hit": ok})
        if ok:
            hit += 1

    return hit, total, details, forbid_hit


def _behavior(
    spec: dict[str, Any],
    rec: dict[str, Any],
    text: str,
) -> tuple[bool, str, bool]:
    """Dönüş: (pass, fail_tag, kpi_grounding_unscored)."""
    behavior = spec.get("behavior") or "static_doc"
    gold = str(rec.get("gold") or rec.get("expected_route") or "")
    final = str(rec.get("final") or rec.get("predicted_route") or "")
    ood_card = _is_ood_card(text, rec)
    refuse = _is_content_refuse(str(rec.get("answer") or rec.get("llm_response") or ""))
    pending = rec.get("pending_action")
    proposed = bool(rec.get("operator_proposed") or rec.get("operator_status") == "proposed" or pending)

    if behavior == "ood_refuse":
        leak = False
        for pat in spec.get("forbid") or []:
            if _claim_hit(str(pat), text) and not (ood_card or refuse):
                leak = True
        if leak:
            return False, "leaked_ood", False
        if ood_card or refuse:
            return True, "", False
        return False, "leaked_ood", False

    if behavior in {"operator_propose"}:
        if ood_card:
            return False, "behavior_fail", False
        if not proposed:
            return False, "bad_pending", False
        if rec.get("operator_status") not in (None, "proposed"):
            if rec.get("operator_status") in {"executed", "confirmed", "running"}:
                return False, "bad_pending", False
        if not _kind_ok(spec.get("pending_kind"), _pending_kind(rec)):
            return False, "bad_pending", False
        return True, "", False

    if behavior == "analyst_empty_kpi":
        unscored = bool(rec.get("empty_kpis", True))
        if ood_card:
            return False, "behavior_fail", unscored
        if proposed:
            return False, "bad_pending", unscored
        answer = str(rec.get("answer") or "")
        if unscored and LIVE_KPI_HALLUCINATION_RE.search(answer) and not MISSING_KPI_RE.search(answer):
            return False, "hallucinated_kpi", unscored
        return True, "", unscored

    # static_doc
    if ood_card:
        return False, "behavior_fail", False
    if proposed:
        return False, "bad_pending", False
    if refuse and gold == "static":
        return False, "claim_miss", False
    del final
    return True, "", False


def score_record(rec: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    qid = str(rec.get("id") or "")
    gold = str(rec.get("gold") or rec.get("expected_route") or "")
    final = str(rec.get("final") or rec.get("predicted_route") or "")
    route_match = gold == final
    behavior = spec.get("behavior") or "static_doc"
    text = _haystack(rec)
    answer = str(rec.get("answer") or rec.get("llm_response") or "")

    refuse_or_card = _is_ood_card(text, rec) or _is_content_refuse(answer)
    route_ok = route_match or (behavior == "ood_refuse" and refuse_or_card)

    behavior_pass, behavior_fail, kpi_unscored = _behavior(spec, rec, text)
    claims_hit_raw, claims_n, claim_details, forbid_hit = _evaluate_claims(spec, text)
    if behavior == "ood_refuse":
        claims_n = 0
        claims_hit_raw = 0
        forbid_hit = False

    claims_complete = claims_hit_raw == claims_n and not forbid_hit
    ungated_pass = behavior_pass and claims_complete

    if not route_ok:
        claims_hit = 0
        binary_pass = False
        behavior_pass_gated = False
        failure_mode = "wrong_route"
    elif ungated_pass:
        claims_hit = claims_hit_raw
        binary_pass = True
        behavior_pass_gated = True
        failure_mode = "pass"
    else:
        claims_hit = claims_hit_raw
        binary_pass = False
        behavior_pass_gated = behavior_pass
        if behavior_fail:
            failure_mode = behavior_fail
        elif forbid_hit or claims_hit_raw < claims_n:
            failure_mode = "claim_miss"
        else:
            failure_mode = "behavior_fail"

    return {
        "id": qid,
        "tag": str(rec.get("tag") or "core").strip().lower() or "core",
        "gold": gold,
        "final": final,
        "family": _family(gold),
        "behavior": behavior,
        "route_match": route_match,
        "route_ok": route_ok,
        "behavior_pass": behavior_pass_gated,
        "behavior_pass_ungated": behavior_pass,
        "claims_hit": claims_hit,
        "claims_hit_ungated": claims_hit_raw,
        "claims_n": claims_n,
        "forbid_hit": forbid_hit,
        "binary_pass": binary_pass,
        "binary_pass_ungated": ungated_pass,
        "failure_mode": failure_mode,
        "paper_failure_mode": (
            "error" if final == "error"
            else "kart" if final == "ood" and gold != "ood"
            else "wrong_route" if not route_ok
            else "pass" if binary_pass
            else "claim_miss"
        ),
        "kpi_grounding_unscored": kpi_unscored,
        "ood_card": bool(rec.get("ood_card")),
        "operator_proposed": bool(rec.get("operator_proposed")),
        "pending_kind": _pending_kind(rec),
        "answer_length": rec.get("answer_length") or len(answer),
        "claim_details": claim_details,
        "source": spec.get("source") or "",
    }


def _frac(num: int, den: int) -> str:
    if den == 0:
        return "0/0"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def _print_report(rows: list[dict[str, Any]], jsonl_path: Path) -> None:
    n = len(rows)
    tagged_core = [r for r in rows if (r.get("tag") or "core") in CORE_SCORE_TAGS]
    traps = [r for r in rows if (r.get("tag") or "") == "trap"]
    probes = [r for r in rows if str(r.get("tag") or "").startswith("probe")]
    core = [r for r in rows if r["family"] != "ood"]
    xrows = [r for r in rows if str(r["id"]).startswith("X")]
    binary = sum(1 for r in rows if r["binary_pass"])
    route = sum(1 for r in rows if r["route_match"])
    claim_hit = sum(int(r["claims_hit_ungated"]) for r in rows)
    claim_n = sum(int(r["claims_n"]) for r in rows)
    gated_hit = sum(int(r["claims_hit"]) for r in rows)
    on_route = [r for r in rows if r["route_match"]]
    on_route_pass = sum(1 for r in on_route if r["binary_pass_ungated"])

    print(f"JSONL: {jsonl_path}", file=sys.stderr)
    print(
        f"N={n}  binary_pass {_frac(binary, n)}  "
        f"route_match {_frac(route, n)}  "
        f"claim micro-recall {_frac(claim_hit, claim_n)}  "
        f"gated-claim {_frac(gated_hit, claim_n)}",
        file=sys.stderr,
    )
    print(
        f"özet Tag=core|clarified binary "
        f"{_frac(sum(1 for r in tagged_core if r['binary_pass']), len(tagged_core))}  "
        f"rota {_frac(sum(1 for r in tagged_core if r['route_match']), len(tagged_core))}",
        file=sys.stderr,
    )
    print(
        f"trap binary {_frac(sum(1 for r in traps if r['binary_pass']), len(traps))}  "
        f"probe binary {_frac(sum(1 for r in probes if r['binary_pass']), len(probes))}",
        file=sys.stderr,
    )
    core_x_excl = [r for r in rows if not str(r["id"]).startswith("X")]
    print(
        f"çekirdek binary {sum(1 for r in core if r['binary_pass'])}/{len(core)}  "
        f"X binary {sum(1 for r in xrows if r['binary_pass'])}/{len(xrows)}",
        file=sys.stderr,
    )
    print(
        f"çekirdek rota (X hariç) {_frac(sum(1 for r in core_x_excl if r['route_match']), len(core_x_excl))}  "
        f"X rota {_frac(sum(1 for r in xrows if r['route_match']), len(xrows))}",
        file=sys.stderr,
    )
    print(
        "doğru rota koşuluyla içerik: "
        + _frac(on_route_pass, len(on_route)),
        file=sys.stderr,
    )
    print("aile kırılımı:", file=sys.stderr)
    by_fam: dict[str, list] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam in FAMILY_ORDER:
        group = by_fam.get(fam) or []
        if not group:
            continue
        bp = sum(1 for r in group if r["binary_pass"])
        rm = sum(1 for r in group if r["route_match"])
        print(
            f"  {fam:<18} n={len(group):3d}  binary {_frac(bp, len(group))}  "
            f"rota {_frac(rm, len(group))}",
            file=sys.stderr,
        )

    print("tag kırılımı:", file=sys.stderr)
    by_tag: dict[str, list] = defaultdict(list)
    for r in rows:
        by_tag[str(r.get("tag") or "core")].append(r)
    for tag in TAG_ORDER:
        group = by_tag.get(tag) or []
        if not group:
            continue
        bp = sum(1 for r in group if r["binary_pass"])
        rm = sum(1 for r in group if r["route_match"])
        print(
            f"  {tag:<18} n={len(group):3d}  binary {_frac(bp, len(group))}  "
            f"rota {_frac(rm, len(group))}",
            file=sys.stderr,
        )

    modes = Counter(r["failure_mode"] for r in rows)
    print("failure_mode:", file=sys.stderr)
    for mode, cnt in modes.most_common():
        print(f"  {mode}: {cnt}", file=sys.stderr)
    unscored = sum(1 for r in rows if r.get("kpi_grounding_unscored"))
    print(
        f"kpi_grounding_unscored={unscored} (empty_kpis dökümü; canlı KPI doğruluğu yok)",
        file=sys.stderr,
    )
    print(
        "Not: birincil metrik binary_pass (davranış + must + forbid). "
        "Rota skoru referanstır, yerine geçmez.",
        file=sys.stderr,
    )
    _print_operator_gaps(rows)


def _print_operator_gaps(rows: list[dict[str, Any]]) -> None:
    """Gold operator (gerçek kaldıraç) ama analist/static, veya teklifsiz operator."""
    supported = [
        r for r in rows
        if r.get("gold") in OPERATOR_GOLD and r.get("behavior") == "operator_propose"
    ]
    deprived = [r for r in supported if r.get("final") in READ_FAMILIES]
    no_proposal = [
        r for r in supported
        if not r.get("operator_proposed") and not r.get("pending_kind")
    ]
    misroute_ok = [
        r for r in rows
        if not r.get("route_match") and r.get("binary_pass")
    ]
    print(
        "1) gold operator (desteklenen kaldıraç) ama final analist/static "
        f"— tool mahrumiyeti ({len(deprived)}):",
        file=sys.stderr,
    )
    if not deprived:
        print("  yok", file=sys.stderr)
    else:
        for rec in deprived:
            print(
                f"  {rec['id']} gold={rec['gold']} final={rec['final']} "
                f"mode={rec['failure_mode']}",
                file=sys.stderr,
            )
    print(
        f"2) teklifsiz operator (propose_* yok) ({len(no_proposal)}):",
        file=sys.stderr,
    )
    if not no_proposal:
        print("  yok", file=sys.stderr)
    else:
        for rec in no_proposal:
            print(
                f"  {rec['id']} gold={rec['gold']} final={rec['final']} "
                f"pending={rec.get('pending_kind')} mode={rec['failure_mode']}",
                file=sys.stderr,
            )
    print(
        "yönlendirici şaşırdı, cevap idare "
        f"(yanlış rota ama binary_pass / refuse) ({len(misroute_ok)}):",
        file=sys.stderr,
    )
    if not misroute_ok:
        print("  yok", file=sys.stderr)
    else:
        for rec in misroute_ok:
            print(
                f"  {rec['id']} gold={rec['gold']} final={rec['final']} "
                f"mode={rec['failure_mode']}",
                file=sys.stderr,
            )


def _write_operator_gaps(rows: list[dict[str, Any]], csv_path: Path) -> None:
    supported = [
        r for r in rows
        if r.get("gold") in OPERATOR_GOLD and r.get("behavior") == "operator_propose"
    ]
    deprived_ids = {
        r["id"] for r in supported if r.get("final") in READ_FAMILIES
    }
    no_proposal_ids = {
        r["id"] for r in supported
        if not r.get("operator_proposed") and not r.get("pending_kind")
    }
    misroute_ok_ids = {
        r["id"] for r in rows
        if not r.get("route_match") and r.get("binary_pass")
    }
    fieldnames = ["id", "gap", "gold", "final", "pending_kind", "failure_mode"]
    out_rows: list[dict[str, Any]] = []
    for rec in supported:
        qid = rec["id"]
        if qid in deprived_ids:
            out_rows.append({
                "id": qid,
                "gap": "tool_deprived",
                "gold": rec.get("gold"),
                "final": rec.get("final"),
                "pending_kind": rec.get("pending_kind"),
                "failure_mode": rec.get("failure_mode"),
            })
        if qid in no_proposal_ids:
            out_rows.append({
                "id": qid,
                "gap": "no_proposal",
                "gold": rec.get("gold"),
                "final": rec.get("final"),
                "pending_kind": rec.get("pending_kind"),
                "failure_mode": rec.get("failure_mode"),
            })
    for rec in rows:
        if rec["id"] in misroute_ok_ids:
            out_rows.append({
                "id": rec["id"],
                "gap": "misroute_ok_refuse",
                "gold": rec.get("gold"),
                "final": rec.get("final"),
                "pending_kind": rec.get("pending_kind"),
                "failure_mode": rec.get("failure_mode"),
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Operator gap CSV: {csv_path}", file=sys.stderr)


def _print_audit(rows: list[dict[str, Any]], ids: tuple[str, ...]) -> None:
    by_id = {r["id"]: r for r in rows}
    print("denetim (12 id):", file=sys.stderr)
    for qid in ids:
        rec = by_id.get(qid)
        if not rec:
            print(f"  {qid} EKSİK", file=sys.stderr)
            continue
        print(
            f"  {qid} gold={rec['gold']} final={rec['final']} "
            f"route_ok={rec['route_ok']} behavior_pass={rec['behavior_pass']} "
            f"claims={rec['claims_hit']}/{rec['claims_n']} "
            f"binary={rec['binary_pass']} mode={rec['failure_mode']}",
            file=sys.stderr,
        )
        for d in rec.get("claim_details") or []:
            if d["kind"] == "forbid":
                mark = "clean" if d.get("hit") else "LEAK"
            else:
                mark = "hit" if d.get("hit") else "MISS"
            print(f"    [{d['kind']}] {mark} {d['pattern']}", file=sys.stderr)


def _write_outputs(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "tag", "gold", "final", "family", "behavior",
        "route_match", "route_ok", "behavior_pass",
        "claims_hit", "claims_n", "forbid_hit",
        "binary_pass", "binary_pass_ungated", "failure_mode", "paper_failure_mode",
        "kpi_grounding_unscored",
        "pending_kind", "answer_length",
    ]
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            slim = {k: rec.get(k) for k in fieldnames}
            slim["claim_details"] = rec.get("claim_details")
            slim["source"] = rec.get("source")
            fh.write(json.dumps(slim, ensure_ascii=False, default=str) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in rows:
            writer.writerow({k: rec.get(k) for k in fieldnames})
    print(f"Özet CSV: {csv_path}", file=sys.stderr)
    print(f"Skor JSONL: {jsonl_path}", file=sys.stderr)
    _write_operator_gaps(rows, jsonl_path.with_name(jsonl_path.stem + "_operator_gaps.csv"))


def _load_summary_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {str(row.get("id") or ""): row for row in csv.DictReader(fh) if row.get("id")}


def _diagnose_note(row: dict[str, Any], old_route: Optional[dict[str, str]]) -> str:
    qid = str(row.get("id") or "")
    tag = str(row.get("tag") or "core")
    gold = str(row.get("gold") or "")
    final = str(row.get("final") or "")
    new_rm = bool(row.get("route_match"))
    old_final = (old_route or {}).get("final") or ""
    old_rm = _as_bool((old_route or {}).get("route_match"))
    tb = (old_route or {}).get("tb_bucket") or ""
    decoder = (old_route or {}).get("decoder") or ""
    agree = _as_bool((old_route or {}).get("families_agree"))

    if tag == "clarified" and old_rm is False and new_rm:
        return "clarified: wrong_route kalktı (cümle netleşince hizalandı)"
    if tag == "clarified" and gold == "static" and final == "ood":
        return "veto: clarified static hâlâ OOD (gold muallak değil; conflict kuralı)"
    if tag == "clarified" and old_rm is False and not new_rm:
        return "clarified: rota hâlâ yanlış"
    if tag == "probe_conflict" and final == "ood":
        return "veto: conflict OOD (decoder doğru / TB agented|ood) — kural, gold hatası değil"
    if tag == "probe_ood" and not new_rm:
        return "probe: ood sızıntısı (family_agree ile desteklenmeyen/trivia yola gömüldü)"
    if tag == "probe_hybrid" and not new_rm:
        return "probe: hibrit bit düşmesi"
    if tag == "trap" and new_rm:
        return "trap: diğer anlam ayrıldı"
    if tag == "trap" and not new_rm:
        return "trap: diğer anlam ayrılmadı"
    if qid in CLARIFIED_FOCUS_IDS and old_rm is True and new_rm:
        return "clarified: rota aynı kaldı (zaten doğruydu)"
    del tb, decoder, agree, old_final
    return ""


def _print_compare(
    rows: list[dict[str, Any]],
    old_route_csv: Path,
    old_content_csv: Path,
    out_csv: Path,
) -> None:
    old_route = _load_summary_csv(old_route_csv)
    old_content = _load_summary_csv(old_content_csv)
    fieldnames = [
        "id", "tag", "gold",
        "old_final", "new_final",
        "old_route_match", "new_route_match",
        "old_binary_pass", "new_binary_pass",
        "old_tb_bucket", "old_decoder", "old_families_agree",
        "new_failure_mode", "note",
    ]
    diagnose_rows: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("id") or "")
        prev_r = old_route.get(qid) or {}
        prev_c = old_content.get(qid) or {}
        rec = {
            "id": qid,
            "tag": row.get("tag") or "core",
            "gold": row.get("gold") or "",
            "old_final": prev_r.get("final") or "",
            "new_final": row.get("final") or "",
            "old_route_match": prev_r.get("route_match") or "",
            "new_route_match": row.get("route_match"),
            "old_binary_pass": prev_c.get("binary_pass") or "",
            "new_binary_pass": row.get("binary_pass"),
            "old_tb_bucket": prev_r.get("tb_bucket") or "",
            "old_decoder": prev_r.get("decoder") or "",
            "old_families_agree": prev_r.get("families_agree") or "",
            "new_failure_mode": row.get("failure_mode") or "",
            "note": _diagnose_note(row, prev_r or None),
        }
        diagnose_rows.append(rec)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnose_rows)
    print(f"Teşhis CSV: {out_csv}", file=sys.stderr)

    by_id = {r["id"]: r for r in rows}
    print("clarified core (eski 118 vs yeni):", file=sys.stderr)
    for qid in CLARIFIED_FOCUS_IDS:
        rec = by_id.get(qid)
        prev = old_route.get(qid) or {}
        if not rec:
            print(f"  {qid} EKSİK", file=sys.stderr)
            continue
        old_rm = prev.get("route_match")
        print(
            f"  {qid} gold={rec['gold']} old={prev.get('final')}/{old_rm} "
            f"new={rec['final']}/{rec['route_match']} "
            f"mode={rec['failure_mode']} "
            f"{_diagnose_note(rec, prev or None)}",
            file=sys.stderr,
        )

    print("trap / probe:", file=sys.stderr)
    for row in rows:
        tag = str(row.get("tag") or "")
        if tag != "trap" and not tag.startswith("probe"):
            continue
        print(
            f"  {row['id']} tag={tag} gold={row['gold']} final={row['final']} "
            f"route={row['route_match']} binary={row['binary_pass']} "
            f"mode={row['failure_mode']} {_diagnose_note(row, old_route.get(row['id']))}",
            file=sys.stderr,
        )

    still_ood = [
        r for r in rows
        if (r.get("tag") or "") == "clarified"
        and r.get("gold") == "static"
        and r.get("final") == "ood"
    ]
    print(
        f"clarified static hâlâ OOD (veto, gold muallak değil): "
        + (", ".join(r["id"] for r in still_ood) or "yok"),
        file=sys.stderr,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cevap checklist skoru (deterministik)")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--keys", type=Path, default=_DEFAULT_KEYS)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        nargs="?",
        const=_DEFAULT_COMPARE,
        help="Eski rota özet CSV (varsayılan production_p7_tb8_answers_118_summary.csv)",
    )
    parser.add_argument(
        "--compare-content",
        type=Path,
        default=_DEFAULT_COMPARE_CONTENT,
        help="Eski içerik özet CSV (binary_pass)",
    )
    parser.add_argument(
        "--audit-ids",
        default=",".join(AUDIT_IDS),
        help="Virgülle id listesi (varsayılan 12 id denetim)",
    )
    args = parser.parse_args(argv)

    keys_doc = _load_json(args.keys)
    specs: dict[str, Any] = keys_doc.get("ids") or keys_doc
    recs = _load_jsonl(args.input)
    tags = _load_tags(args.dataset)
    for rec in recs:
        qid = str(rec.get("id") or "")
        rec["tag"] = rec.get("tag") or tags.get(qid) or "core"
    missing = [r["id"] for r in recs if r.get("id") not in specs]
    if missing:
        print(f"Uyarı: anahtarda yok: {missing}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for rec in recs:
        qid = str(rec.get("id") or "")
        spec = specs.get(qid) or {"behavior": "static_doc", "must": []}
        rows.append(score_record(rec, spec))

    extra_keys = sorted(set(specs) - {r["id"] for r in recs})
    if extra_keys:
        if len(recs) >= max(1, len(specs) - 8):
            print(f"Uyarı: JSONL'de yok: {extra_keys}", file=sys.stderr)
        else:
            print(
                f"Uyarı: JSONL subset {len(recs)} vs keys {len(specs)} "
                f"(ör. wave2/RQ2b); {len(extra_keys)} anahtar bu dosyada yok",
                file=sys.stderr,
            )

    out_jsonl = args.output or args.input.with_name(args.input.stem + "_content.jsonl")
    out_csv = out_jsonl.with_name(out_jsonl.stem + "_summary.csv")

    _write_outputs(rows, out_jsonl, out_csv)
    _print_report(rows, args.input)
    audit = tuple(x.strip() for x in str(args.audit_ids).split(",") if x.strip())
    _print_audit(rows, audit)
    if args.compare:
        diagnose_csv = args.input.with_name(args.input.stem + "_diagnose.csv")
        _print_compare(rows, args.compare, args.compare_content, diagnose_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
