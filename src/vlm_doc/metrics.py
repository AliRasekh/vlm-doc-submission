"""DocVQA scoring with versioned ANLS definitions.

Primary (Phase 2B+): ``anls_normalized_strict_v2``
  - lowercase + strip + whitespace collapse
  - NL = edit_distance / max(len(normalized_a), len(normalized_b))
  - score = (1 - NL) if NL < 0.5 else 0.0   (strict cutoff on distance)
  - max over valid nonempty reference strings
  - failures / missing requested predictions count as 0 in requested-set means

This is an explicitly defined project metric. It is **not** claimed to be
byte-identical to Pix2Struct's ``anls_metric`` (different normalization
pipelines). The strict ``NL < 0.5`` cutoff follows the public Pix2Struct
snippet semantics for the distance threshold
(https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
at the revision recorded in docs/sources.md).

Legacy inclusive cutoff (Phase 2A primary): ``anls_normalized_inclusive_v1``
  - same normalization and lengths
  - score = (1 - NL) if (1 - NL) >= 0.5 else 0.0  (i.e. keep when NL <= 0.5)

VLMEvalKit compatibility: ``anls_vlmevalkit`` (original-string upper length
denominator quirk; inclusive similarity gate).
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

ANLS_THRESHOLD = 0.5
SCORER_VERSION_PRIMARY = "anls_normalized_strict_v2"
SCORER_VERSION_LEGACY_INCLUSIVE = "anls_normalized_inclusive_v1"
SCORER_VERSION_VLMEVALKIT = "anls_vlmevalkit"


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min(distances[i1], distances[i1 + 1], distances_[-1])
                )
        distances = distances_
    return distances[-1]


def _normalized_similarity(gt: str, pred: str) -> tuple[float, float]:
    """Return (similarity, NL) on normalized strings."""
    g = normalize_text(gt)
    p = normalize_text(pred)
    if not g and not p:
        return 1.0, 0.0
    if not g or not p:
        return 0.0, 1.0
    dist = levenshtein_distance(g, p)
    nl = dist / max(len(g), len(p))
    return 1.0 - nl, nl


def _valid_references(references: Sequence[str]) -> list[str]:
    out: list[str] = []
    for r in references:
        if r is None:
            continue
        s = str(r)
        if normalize_text(s) == "":
            continue
        out.append(s)
    return out


def anls_normalized_strict_v2(
    prediction: str, references: Sequence[str], *, threshold: float = ANLS_THRESHOLD
) -> float:
    """Primary Phase 2B ANLS: zero unless NL < threshold (strict)."""
    refs = _valid_references(references)
    if not refs:
        return 0.0
    scores = []
    for ref in refs:
        sim, nl = _normalized_similarity(ref, prediction)
        scores.append(sim if nl < threshold else 0.0)
    return max(scores)


def anls_normalized_inclusive_v1(
    prediction: str, references: Sequence[str], *, threshold: float = ANLS_THRESHOLD
) -> float:
    """Legacy Phase 2A primary: keep when similarity >= threshold (NL <= 0.5)."""
    refs = _valid_references(references)
    if not refs:
        return 0.0
    scores = []
    for ref in refs:
        sim, _nl = _normalized_similarity(ref, prediction)
        scores.append(sim if sim >= threshold else 0.0)
    return max(scores)


# Back-compat alias used by older imports/tests until updated.
def anls_single(prediction: str, references: Sequence[str], *, threshold: float = ANLS_THRESHOLD) -> float:
    """Deprecated alias → legacy inclusive v1 (Phase 2A behavior)."""
    return anls_normalized_inclusive_v1(prediction, references, threshold=threshold)


def anls_vlmevalkit_single(
    prediction: str,
    references: Sequence[str],
    *,
    threshold: float = ANLS_THRESHOLD,
) -> float:
    """VLMEvalKit-compatible ANLS (denominator uses original uppercased lengths)."""
    refs = [str(r) for r in references if r is not None]
    refs = [r for r in refs if normalize_text(r) != ""]
    if not refs:
        return 0.0
    pred = str(prediction)
    dists = []
    for gt in refs:
        gt_n = " ".join(gt.strip().lower().split())
        pred_n = " ".join(pred.strip().lower().split())
        dist = levenshtein_distance(gt_n, pred_n)
        length = max(len(gt.upper()), len(pred.upper()))
        nl = 0.0 if length == 0 else float(dist) / float(length)
        dists.append(nl)
    best_sim = 1.0 - min(dists)
    return 0.0 if best_sim < threshold else best_sim


def exact_match_single(prediction: str, references: Sequence[str]) -> float:
    pred_n = normalize_text(prediction)
    refs = [normalize_text(r) for r in _valid_references(references)]
    if not refs:
        return 0.0
    return 1.0 if any(pred_n == r for r in refs) else 0.0


def score_example(
    prediction: str | None,
    references: Sequence[str],
    *,
    status: str = "ok",
    primary: str = SCORER_VERSION_PRIMARY,
) -> dict[str, Any]:
    if status != "ok" or prediction is None:
        return {
            "primary": primary,
            "anls": 0.0,
            "anls_normalized_strict_v2": 0.0,
            "anls_normalized_inclusive_v1": 0.0,
            "anls_vlmevalkit": 0.0,
            "exact_match": 0.0,
            "scored": False,
            "failure": True,
            "invalid_references": False,
        }
    refs = list(references) if references is not None else []
    valid = _valid_references(refs)
    invalid_refs = len(valid) == 0
    strict = anls_normalized_strict_v2(prediction, refs)
    inclusive = anls_normalized_inclusive_v1(prediction, refs)
    vlk = anls_vlmevalkit_single(prediction, refs)
    em = exact_match_single(prediction, refs)
    primary_score = {
        SCORER_VERSION_PRIMARY: strict,
        SCORER_VERSION_LEGACY_INCLUSIVE: inclusive,
        SCORER_VERSION_VLMEVALKIT: vlk,
    }.get(primary, strict)
    return {
        "primary": primary,
        "anls": primary_score,  # primary field
        "anls_normalized_strict_v2": strict,
        "anls_normalized_inclusive_v1": inclusive,
        "anls_vlmevalkit": vlk,
        "exact_match": em,
        "scored": True,
        "failure": False,
        "invalid_references": invalid_refs,
    }


def scorer_source_hash() -> str:
    """Hash of this module's source for provenance."""
    with open(__file__, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()


def validate_answers_sidecar(
    answers: Sequence[dict],
    *,
    required_question_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate answer sidecar before scoring.

    Flags duplicate IDs, missing coverage, and invalid/empty references.
    Does not reward empty-reference matches.
    """
    req = [int(x) for x in required_question_ids]
    req_set = set(req)
    seen: dict[int, int] = {}
    by_qid: dict[int, list[str]] = {}
    invalid_empty_refs: list[int] = []
    for i, item in enumerate(answers):
        qid = int(item["question_id"])
        seen[qid] = seen.get(qid, 0) + 1
        refs = item.get("answers")
        if refs is None:
            refs = []
        refs_list = list(refs)
        valid = _valid_references(refs_list)
        by_qid[qid] = valid
        if len(valid) == 0:
            invalid_empty_refs.append(qid)
    duplicates = sorted([q for q, n in seen.items() if n > 1])
    missing = sorted(req_set - set(by_qid))
    extras = sorted(set(by_qid) - req_set)
    ok = not duplicates and not missing and not invalid_empty_refs
    return {
        "ok": ok,
        "n_required": len(req),
        "n_sidecar_unique": len(by_qid),
        "duplicate_question_ids": duplicates,
        "missing_question_ids": missing,
        "extra_question_ids": extras,
        "invalid_empty_reference_question_ids": sorted(set(invalid_empty_refs)),
        "answers_by_qid": {str(k): v for k, v in by_qid.items()},
    }


def aggregate_scores(
    per_example: Sequence[dict],
    *,
    n_requested: int,
    required_question_ids: Sequence[int] | None = None,
    primary_key: str = "anls_normalized_strict_v2",
) -> dict[str, Any]:
    """Aggregate over the full requested set.

    Rejects duplicate question_ids or out-of-manifest records when
    ``required_question_ids`` is provided (raises ValueError).
    """
    if n_requested <= 0:
        raise ValueError("n_requested must be positive")

    if required_question_ids is not None:
        req = [int(x) for x in required_question_ids]
        if len(req) != n_requested:
            raise ValueError(
                f"n_requested={n_requested} != len(required_question_ids)={len(req)}"
            )
        if len(req) != len(set(req)):
            raise ValueError(
                "required_question_ids contains duplicate question_id values"
            )
        req_set = set(req)
        seen: set[int] = set()
        for row in per_example:
            qid = int(row["question_id"])
            if qid not in req_set:
                raise ValueError(f"out-of-manifest question_id={qid}")
            if qid in seen:
                raise ValueError(f"duplicate question_id in aggregate records: {qid}")
            seen.add(qid)

    anls_sum = 0.0
    em_sum = 0.0
    inclusive_sum = 0.0
    vlk_sum = 0.0
    n_ok = 0
    n_err = 0
    # Reject duplicates even when required_question_ids is omitted.
    seen_any: set[int] = set()
    for row in per_example:
        qid = int(row["question_id"])
        if qid in seen_any:
            raise ValueError(f"duplicate question_id in aggregate records: {qid}")
        seen_any.add(qid)

        status = row.get("status", "ok")
        if status == "ok":
            n_ok += 1
        else:
            n_err += 1
        # Failed / non-ok records contribute zero regardless of cached metrics.
        if status != "ok":
            continue
        metrics = row.get("metrics")
        if not metrics:
            metrics = score_example(
                row.get("prediction"),
                row.get("reference_answers") or [],
                status=status,
            )
        # Never trust a stale primary alone: prefer explicit versioned fields.
        anls_sum += float(metrics.get(primary_key, metrics.get("anls", 0.0)))
        em_sum += float(metrics.get("exact_match", 0.0))
        inclusive_sum += float(
            metrics.get("anls_normalized_inclusive_v1", metrics.get("anls", 0.0))
        )
        vlk_sum += float(metrics.get("anls_vlmevalkit", 0.0))

    n_missing = max(0, n_requested - len(per_example))
    return {
        "n_requested": n_requested,
        "n_records": len(per_example),
        "n_completed_ok": n_ok,
        "n_failed": n_err,
        "n_missing": n_missing,
        "failure_rate_requested": (n_err + n_missing) / n_requested,
        "primary_metric": primary_key,
        "mean_anls_requested": anls_sum / n_requested,
        "mean_exact_match_requested": em_sum / n_requested,
        "mean_anls_normalized_inclusive_v1_requested": inclusive_sum / n_requested,
        "mean_anls_vlmevalkit_requested": vlk_sum / n_requested,
        "mean_anls_completed_ok_only": (anls_sum / n_ok) if n_ok else None,
        "mean_exact_match_completed_ok_only": (em_sum / n_ok) if n_ok else None,
        "anls_definition": (
            "anls_normalized_strict_v2: lowercase+strip+collapse_ws; "
            "NL=edit/max(len_norm); score=1-NL if NL<0.5 else 0; max over nonempty refs; "
            "failures/missing count as 0 in requested mean"
        ),
        "scorer_source_sha256": scorer_source_hash(),
    }
