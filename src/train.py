from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from functools import partial
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model
from span_utils import bio_to_spans, span_metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="prajjwal1/bert-mini")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--doc_stride", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=0, help="Evaluate every N steps (0 => per epoch)")
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--freeze_layers", type=int, default=0, help="Freeze lowest N encoder layers")
    ap.add_argument("--class_weight_power", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_label_weights(dataset: PIIDataset, power: float) -> torch.Tensor:
    counts = torch.zeros(len(LABELS), dtype=torch.float)
    for item in dataset.items:
        for label_id in item["labels"]:
            counts[label_id] += 1
    counts = torch.clamp(counts, min=1.0)
    weights = torch.pow(1.0 / counts, power)
    weights = weights / weights.mean()
    return weights


def maybe_freeze_layers(model, num_layers: int):
    if num_layers <= 0:
        return
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        print("Warning: unable to locate base model for freezing.")
        return
    encoder = getattr(base_model, "encoder", None) or getattr(base_model, "transformer", None)
    if encoder is None or not hasattr(encoder, "layer"):
        print("Warning: encoder layers not found; skipping freezing.")
        return
    layers = encoder.layer
    freeze_until = min(len(layers), num_layers)
    for layer in layers[:freeze_until]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Froze {freeze_until} encoder layers.")


def evaluate(
    model,
    dataloader: DataLoader,
    gold_spans: Dict[str, List[Tuple[int, int, str]]],
    device: str,
):
    model.eval()
    predictions = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            attn = batch["attention_mask"].cpu().tolist()

            for uid, preds, offsets, mask in zip(
                batch["example_ids"], pred_ids, batch["offset_mapping"], attn
            ):
                spans = bio_to_spans(offsets, preds, mask)
                if spans:
                    predictions[uid].extend(spans)

    # deduplicate spans per utterance
    pred_sets = {uid: list({span for span in spans}) for uid, spans in predictions.items()}
    metrics = span_metrics(gold_spans, pred_sets)
    model.train()
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(
        args.train,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        lowercase=args.lowercase,
        is_train=True,
    )
    dev_ds = PIIDataset(
        args.dev,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        doc_stride=0,
        lowercase=args.lowercase,
        is_train=False,
    )

    collate_fn = partial(
        collate_batch,
        pad_token_id=tokenizer.pad_token_id,
        label_pad_id=-100,
        return_tensors=True,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = create_model(args.model_name, dropout=args.dropout)
    maybe_freeze_layers(model, args.freeze_layers)
    model.to(args.device)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"])
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_updates_per_epoch = math.ceil(len(train_dl) / max(1, args.gradient_accumulation_steps))
    total_steps = max(1, total_updates_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    label_weights = None
    if args.class_weight_power > 0:
        label_weights = compute_label_weights(train_ds, args.class_weight_power).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=label_weights, ignore_index=-100)

    scaler = GradScaler(enabled=args.fp16 and args.device.startswith("cuda"))

    best_macro = -1.0
    best_metrics = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad()
        progress = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            with autocast(enabled=scaler.is_enabled()):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.max_grad_norm:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.eval_every and global_step % args.eval_every == 0:
                    metrics = evaluate(model, dev_dl, dev_ds.gold_spans, args.device)
                    macro = metrics["macro_f1"]
                    print(f"[step {global_step}] Macro-F1={macro:.3f} PII-F1={metrics['pii']['f1']:.3f}")
                    if macro > best_macro:
                        best_macro = macro
                        best_metrics = metrics
                        model.save_pretrained(os.path.join(args.out_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(args.out_dir, "best"))
                        print(f"Saved new best model to {os.path.join(args.out_dir, 'best')}")

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        if not args.eval_every:
            metrics = evaluate(model, dev_dl, dev_ds.gold_spans, args.device)
            macro = metrics["macro_f1"]
            print(f"[epoch {epoch}] Macro-F1={macro:.3f} PII-F1={metrics['pii']['f1']:.3f}")
            if macro > best_macro:
                best_macro = macro
                best_metrics = metrics
                model.save_pretrained(os.path.join(args.out_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.out_dir, "best"))
                print(f"Saved new best model to {os.path.join(args.out_dir, 'best')}")

    # Always save final checkpoint
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved final model + tokenizer to {args.out_dir}")
    if best_metrics:
        print(f"Best macro-F1: {best_macro:.3f} | PII F1: {best_metrics['pii']['f1']:.3f}")


if __name__ == "__main__":
    main()
