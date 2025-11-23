from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(description="Dynamic quantization for Torch token-classification models.")
    ap.add_argument("--model_dir", required=True, help="Directory containing the trained model.")
    ap.add_argument("--output_dir", required=True, help="Where to store the quantized model.")
    ap.add_argument("--dtype", default="qint8", choices=["qint8", "float16"], help="Quantization dtype.")
    return ap.parse_args()


def main():
    args = parse_args()
    dtype = torch.qint8 if args.dtype == "qint8" else torch.float16

    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=dtype,
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    quantized.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.save_pretrained(output_path)

    print(f"Quantized model saved to {output_path} ({args.dtype}).")


if __name__ == "__main__":
    main()

