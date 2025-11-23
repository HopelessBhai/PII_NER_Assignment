from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from labels import ID2LABEL, label_is_pii

Span = Tuple[int, int, str]


def bio_to_spans(
    offsets: Sequence[Tuple[int, int]],
    label_ids: Sequence[int],
    attention_mask: Sequence[int] | None = None,
) -> List[Span]:
    spans: List[Span] = []
    current_label = None
    current_start = None
    current_end = None

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if attention_mask is not None and idx < len(attention_mask) and attention_mask[idx] == 0:
            continue
        if start == 0 and end == 0:
            continue

        if lid == -100:
            label = "O"
        else:
            label = ID2LABEL.get(int(lid), "O")

        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def _compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def span_metrics(
    gold: Dict[str, Iterable[Span]],
    pred: Dict[str, Iterable[Span]],
) -> Dict[str, Dict]:
    label_set = set()
    for spans in gold.values():
        for _, _, lab in spans:
            label_set.add(lab)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for uid in gold.keys():
        gold_spans = set(gold.get(uid, []))
        pred_spans = set(pred.get(uid, []))

        for span in pred_spans:
            if span in gold_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in gold_spans:
            if span not in pred_spans:
                fn[span[2]] += 1

    per_label = {}
    macro_f1 = 0.0
    for lab in sorted(label_set):
        prec, rec, f1 = _compute_prf(tp[lab], fp[lab], fn[lab])
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1}
        macro_f1 += f1
    macro_f1 /= max(1, len(label_set))

    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0

    for uid in gold.keys():
        g_spans = gold.get(uid, [])
        p_spans = pred.get(uid, [])

        g_pii = set((s, e, "PII") for s, e, lab in g_spans if label_is_pii(lab))
        g_non = set((s, e, "NON") for s, e, lab in g_spans if not label_is_pii(lab))
        p_pii = set((s, e, "PII") for s, e, lab in p_spans if label_is_pii(lab))
        p_non = set((s, e, "NON") for s, e, lab in p_spans if not label_is_pii(lab))

        for span in p_pii:
            if span in g_pii:
                pii_tp += 1
            else:
                pii_fp += 1
        for span in g_pii:
            if span not in p_pii:
                pii_fn += 1

        for span in p_non:
            if span in g_non:
                non_tp += 1
            else:
                non_fp += 1
        for span in g_non:
            if span not in p_non:
                non_fn += 1

    pii_scores = _compute_prf(pii_tp, pii_fp, pii_fn)
    non_scores = _compute_prf(non_tp, non_fp, non_fn)

    return {
        "per_label": per_label,
        "macro_f1": macro_f1,
        "pii": {"precision": pii_scores[0], "recall": pii_scores[1], "f1": pii_scores[2]},
        "non_pii": {"precision": non_scores[0], "recall": non_scores[1], "f1": non_scores[2]},
    }

