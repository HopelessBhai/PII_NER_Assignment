# PII NER Assignment

Token-level NER pipeline that trains lightweight BERT-style models (default: `prajjwal1/bert-mini`) to tag PII in noisy STT transcripts, with tooling to ingest larger datasets (e.g., `ai4privacy/pii-masking-200k`) and run training/inference on Modal A100 GPUs.

---

## 1. Environment

```bash
uv pip install -r requirements.txt          # or pip install -r requirements.txt
```

Python 3.12+. The repo includes `uv.lock` for reproducibility.

---

## 2. Data Preparation

### Convert the Ai4Privacy English split

```bash
uv run python scripts/build_dataset.py \
  --input_jsonl data/english_pii_43k.jsonl \
  --output data/english_pii_converted.jsonl \
  --language en
```

### Train/dev/test split

```bash
uv run python scripts/split_dataset.py \
  --input data/english_pii_converted.jsonl \
  --train_out data/pii_en_train.jsonl \
  --dev_out data/pii_en_dev.jsonl \
  --test_out data/pii_en_test.jsonl \
  --seed 123 \
  --train_ratio 0.7 --dev_ratio 0.15 --test_ratio 0.15
```

`scripts/inspect_labels.py` can be used to audit label distributions.

---

## 3. Local Training / Evaluation

```bash
uv run python src/train.py \
  --model_name prajjwal1/bert-mini \
  --train data/pii_en_train.jsonl \
  --dev data/pii_en_dev.jsonl \
  --out_dir out/pii_en \
  --epochs 3 \
  --batch_size 16 \
  --max_length 256
```

```bash
uv run python src/predict.py \
  --model_dir out/pii_en/best \
  --input data/pii_en_dev.jsonl \
  --output out/pii_en/dev_pred.json \
  --max_length 256 \
  --batch_size 16
```

```bash
uv run python src/eval_span_f1.py \
  --gold data/pii_en_dev.jsonl \
  --pred out/pii_en/dev_pred.json
```

```bash
uv run python src/measure_latency.py \
  --model_dir out/pii_en/best \
  --input data/pii_en_dev.jsonl \
  --runs 50 \
  --device cpu
```

---

## 4. Modal GPU Workflow

All Modal functions mount the repo and a persistent volume `pii-ner-models` at `/vol`. After training, checkpoints are copied into `/vol/<subdir>` so you can download them later.

### Train on A100
```powershell
uv run modal run modal_app.py::train `
  --model-name prajjwal1/bert-mini `
  --train-path data/pii_en_train.jsonl `
  --dev-path data/pii_en_dev.jsonl `
  --out-dir out/pii_en `
  --epochs 3 `
  --batch-size 32 `
  --max-length 256 `
  --volume-subdir pii_en_run1
```

### Predict (results saved to Modal volume)
```powershell
uv run modal run modal_app.py::predict `
  --model-dir /vol/pii_en_run1/best `
  --input data/pii_en_test.jsonl `
  --output /vol/pii_en_run1/test_pred.json `
  --max-length 256 `
  --batch-size 16
```

### Evaluate on Modal
```powershell
uv run modal run modal_app.py::evaluate `
  --gold-path data/pii_en_dev.jsonl `
  --pred-path /vol/pii_en_run1/dev_pred.json
```

### Latency on Modal GPU
```powershell
uv run modal run modal_app.py::latency `
  --model_dir /vol/pii_en_run1/best `
  --input data/pii_en_dev.jsonl `
  --runs 50 `
  --device cuda
```

### Retrieve artifacts

```powershell
# Copy a model/prediction file from the volume to your local machine
modal volume get pii-ner-models /vol/pii_en_run1/best ./out/pii_en_run1/best

# Or stream a zipped directory directly
uv run modal run modal_app.py::download --out-dir /vol/pii_en_run1 --archive-name pii_en_run1.zip > pii_en_run1.zip
```

---

## 5. Repo Structure

```
├── data/                       # starter + converted JSONL files
├── docs/datasets.md            # dataset sourcing notes
├── scripts/
│   ├── build_dataset.py        # convert HF records -> assignment schema
│   ├── split_dataset.py        # train/dev/test splitter
│   └── inspect_labels.py       # label histogram helper
├── src/
│   ├── dataset.py              # PIIDataset with doc stride + tensor batching
│   ├── train.py                # HF trainer w/ eval + checkpointing
│   ├── predict.py              # batched inference with span decoding
│   ├── eval_span_f1.py         # span + PII metrics
│   ├── measure_latency.py
│   └── span_utils.py           # shared BIO utilities
├── modal_app.py                # Modal entrypoints (train/predict/eval/latency/download)
└── out/                        # local outputs (ignored by git)
```

---

## 6. Optional CPU Quantization

Dynamic quantization can significantly reduce CPU latency without retraining:

```bash
uv run python scripts/quantize_model.py \
  --model_dir out/pii_en/best \
  --output_dir out/pii_en_quantized \
  --dtype qint8
```

Then point inference/latency scripts at `out/pii_en_quantized` instead of the original checkpoint.

---

## 7. Notes

- `src/train.py` supports gradient accumulation, class weighting, mixed precision (`--fp16`), layer freezing, and dev evaluation either per epoch or every `--eval_every` steps.
- Predictions are deduplicated spans with auto `pii` flags derived from labels defined in `src/labels.py`.
- Use `modal volume ls pii-ner-models` to inspect stored checkpoints and predictions.
- Keep p95 latency around 20 ms (batch size 1, CPU) as requested in the assignment. Use smaller models / shorter max length if needed. 

Happy redacting!
