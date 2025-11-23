import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from labels import label_is_pii
from span_utils import bio_to_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--doc_stride", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer_source = args.model_dir if args.model_name is None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            samples.append({"id": obj["id"], "text": obj["text"]})

    results = defaultdict(list)
    batch_size = max(1, args.batch_size)
    use_overflow = args.doc_stride > 0

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]
        texts = [sample["text"].lower() if args.lowercase else sample["text"] for sample in batch]

        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=use_overflow,
            padding=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)
        offsets = enc["offset_mapping"].tolist()
        attn_mask_cpu = attention_mask.cpu().tolist()

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

        overflow_mapping = enc.get("overflow_to_sample_mapping", None)
        if overflow_mapping is not None:
            if hasattr(overflow_mapping, "tolist"):
                sample_mapping = overflow_mapping.tolist()
            else:
                sample_mapping = list(overflow_mapping)
        else:
            sample_mapping = list(range(len(batch)))

        for idx, sample_idx in enumerate(sample_mapping):
            uid = batch[sample_idx]["id"]
            spans = bio_to_spans(offsets[idx], pred_ids[idx], attn_mask_cpu[idx])
            for start, end, label in spans:
                if end <= start:
                    continue
                results[uid].append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "label": label,
                        "pii": bool(label_is_pii(label)),
                    }
                )

    deduped = {}
    for sample in samples:
        uid = sample["id"]
        spans = results.get(uid, [])
        uniq = {(span["start"], span["end"], span["label"], span["pii"]) for span in spans}
        deduped[uid] = [
            {"start": s, "end": e, "label": lab, "pii": pii} for (s, e, lab, pii) in sorted(uniq)
        ]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(deduped)} utterances to {args.output}")


if __name__ == "__main__":
    main()
