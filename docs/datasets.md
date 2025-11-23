## Dataset Strategy

### Provided Samples
- `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl` contain 2/2/2 utterances respectively.
- Useful for smoke tests only; macro-F1 quickly saturates and offers no generalization.

### External Corpora (Hugging Face)
- `ai4privacy/pii-masking-200k`: synthetic STT-style utterances with token spans for common PII classes. Map labels via `scripts/build_dataset.py`.
- `microsoft/PII-Masking`: mix of support transcripts and synthetic personas with phone, email, dates, credit cards; good for PERSON_NAME variety.
- `pii_dataset_cleaned`: manually annotated chat transcripts; includes addresses and organization names for negative sampling.

### Bootstrapping Ideas
- Generate synthetic card/phone/email mentions by sampling formats and inserting into real transcripts (e.g., Common Voice text).
- Apply weak regex labeling for obvious entity types, then manually vet a subset to reduce noise.
- Perform back-translation or phonetic perturbations (“double eight”, “oh four”) to mimic STT quirks.

### Integration Steps
1. Use `scripts/build_dataset.py` to pull a subset (`--limit`) of an external dataset and convert to JSONL with repo label schema.
2. Concatenate with the provided sample files to keep evaluation comparable; maintain a held-out dev split that mirrors STT noise.
3. Track provenance (dataset name + license) alongside the merged file for reproducibility.

