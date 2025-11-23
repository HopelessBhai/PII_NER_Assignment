from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset


LABEL_MAP = {
    "credit_card": "CREDIT_CARD",
    "person": "PERSON_NAME",
    "name": "PERSON_NAME",
    "phone_number": "PHONE",
    "email_address": "EMAIL",
    "location": "LOCATION",
    "city": "CITY",
    "date_time": "DATE",
    "datetime": "DATE",
}

AI4PRIVACY_LABEL_MAP = {
    # credit card / financial
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD",
    "CREDITCARDISSUER": "CREDIT_CARD",
    "MASKEDNUMBER": "CREDIT_CARD",
    "ACCOUNTNUMBER": "CREDIT_CARD",
    "IBAN": "CREDIT_CARD",
    "BIC": "CREDIT_CARD",
    "PIN": "CREDIT_CARD",
    # phones
    "PHONENUMBER": "PHONE",
    "PHONE": "PHONE",
    "PHONEIMEI": "PHONE",
    # email
    "EMAIL": "EMAIL",
    # people
    "FIRSTNAME": "PERSON_NAME",
    "LASTNAME": "PERSON_NAME",
    "MIDDLENAME": "PERSON_NAME",
    "FULLNAME": "PERSON_NAME",
    "PERSON": "PERSON_NAME",
    # dates
    "DATE": "DATE",
    "DOB": "DATE",
    "BIRTHDATE": "DATE",
    # city
    "CITY": "CITY",
    # broader locations
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "STREET": "LOCATION",
    "BUILDINGNUMBER": "LOCATION",
    "SECONDARYADDRESS": "LOCATION",
    "NEARBYGPSCOORDINATE": "LOCATION",
    "ADDRESS": "LOCATION",
}


def normalize_label(label: str) -> str | None:
    label = label.lower()
    return LABEL_MAP.get(label)


def convert_record(record: dict, language: str | None = None) -> dict | None:
    if "source_text" in record and "privacy_mask" in record:
        return convert_ai4privacy_record(record, language)

    text = record.get("text") or record.get("transcript")
    entities = record.get("entities") or record.get("spans")
    if not text or not entities:
        return None

    new_entities = []
    for ent in entities:
        start = ent.get("start") or ent.get("begin")
        end = ent.get("end") or ent.get("finish")
        label = ent.get("label") or ent.get("entity_label")
        norm_label = normalize_label(label or "")
        if norm_label is None:
            continue
        new_entities.append({"start": int(start), "end": int(end), "label": norm_label})

    if not new_entities:
        return None

    return {"id": record.get("id") or record.get("guid"), "text": text, "entities": new_entities}


def convert_ai4privacy_record(record: dict, language: str | None) -> dict | None:
    lang = record.get("language")
    if language and lang and lang.lower() != language.lower():
        return None

    text = record.get("source_text")
    spans = record.get("privacy_mask", [])
    if not text or not spans:
        return None

    entities = []
    for ent in spans:
        label = AI4PRIVACY_LABEL_MAP.get((ent.get("label") or "").upper())
        if not label:
            continue
        try:
            start = int(ent.get("start"))
            end = int(ent.get("end"))
        except (TypeError, ValueError):
            continue
        if start >= end:
            continue
        entities.append({"start": start, "end": end, "label": label})

    if not entities:
        return None

    return {
        "id": str(record.get("id") or record.get("guid")),
        "text": text,
        "entities": entities,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ai4privacy/pii-masking-200k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="data/extended_train.jsonl")
    ap.add_argument("--input_jsonl", default=None, help="Existing JSONL file to convert")
    ap.add_argument("--language", default=None, help="Filter ai4privacy rows by language code (e.g. en)")
    args = ap.parse_args()

    if args.input_jsonl:
        def iterator():
            with open(args.input_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
        records = iterator()
    else:
        records = load_dataset(args.dataset, split=args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            converted = convert_record(record, language=args.language)
            if converted is None:
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"Wrote {written} examples to {output_path}")


if __name__ == "__main__":
    main()

