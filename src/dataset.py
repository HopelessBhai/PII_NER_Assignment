import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _select_label_for_token(start: int, end: int, char_tags: List[str]) -> str:
    if start == end or start >= len(char_tags):
        return "O"
    end = min(end, len(char_tags))
    span_tags = char_tags[start:end]
    for tag in span_tags:
        if tag != "O":
            return tag
    return "O"


class PIIDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        label_list: List[str],
        max_length: int = 256,
        doc_stride: int = 0,
        lowercase: bool = False,
        is_train: bool = True,
    ):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length
        self.doc_stride = max(0, doc_stride)
        self.lowercase = lowercase
        self.is_train = is_train
        self.gold_spans = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                original_text = obj["text"]
                text = original_text.lower() if self.lowercase else original_text
                entities = obj.get("entities", [])
                self.gold_spans[obj["id"]] = [
                    (e["start"], e["end"], e["label"]) for e in entities
                ]

                char_tags = ["O"] * len(text)
                for e in entities:
                    s, e_idx, lab = e["start"], e["end"], e["label"]
                    if s < 0 or e_idx > len(text) or s >= e_idx:
                        continue
                    char_tags[s] = f"B-{lab}"
                    for i in range(s + 1, e_idx):
                        char_tags[i] = f"I-{lab}"

                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                    stride=self.doc_stride,
                    return_overflowing_tokens=self.doc_stride > 0,
                )
                input_sequences = enc["input_ids"]
                attention_sequences = enc["attention_mask"]
                offset_sequences = enc["offset_mapping"]

                def ensure_list_of_lists(seq):
                    if not seq:
                        return []
                    first = seq[0]
                    if isinstance(first, list):
                        return seq
                    return [seq]

                input_sequences = ensure_list_of_lists(input_sequences)
                attention_sequences = ensure_list_of_lists(attention_sequences)
                offset_sequences = ensure_list_of_lists(offset_sequences)

                for idx in range(len(input_sequences)):
                    offsets = offset_sequences[idx]
                    bio_tags = []
                    for (start, end) in offsets:
                        bio_tags.append(_select_label_for_token(start, end, char_tags))

                    if len(bio_tags) != len(input_sequences[idx]):
                        bio_tags = ["O"] * len(input_sequences[idx])

                    label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]
                    chunk_id = f"{obj['id']}::chunk{idx}" if idx > 0 else obj["id"]

                    self.items.append(
                        {
                            "id": chunk_id,
                            "example_id": obj["id"],
                            "text": original_text,
                            "input_ids": input_sequences[idx],
                            "attention_mask": attention_sequences[idx],
                            "labels": label_ids,
                            "offset_mapping": offsets,
                            "chunk_index": idx,
                        }
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_batch(
    batch,
    pad_token_id: Optional[int],
    label_pad_id: int = -100,
    return_tensors: bool = True,
):
    input_ids_list = [x["input_ids"] for x in batch]
    attention_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]
    offsets_list = [x["offset_mapping"] for x in batch]

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = pad_token_id if pad_token_id is not None else 0

    def pad(seq, pad_value):
        return seq + [pad_value] * (max_len - len(seq))

    def pad_offsets(offsets: List[Tuple[int, int]]):
        return offsets + [(0, 0)] * (max_len - len(offsets))

    input_ids = [pad(ids, pad_id) for ids in input_ids_list]
    attention_mask = [pad(am, 0) for am in attention_list]
    labels = [pad(lab, label_pad_id) for lab in labels_list]
    offsets = [pad_offsets(ofs) for ofs in offsets_list]

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long) if return_tensors else input_ids,
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        if return_tensors
        else attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long) if return_tensors else labels,
        "ids": [x["id"] for x in batch],
        "example_ids": [x.get("example_id", x["id"]) for x in batch],
        "chunk_indices": [x.get("chunk_index", 0) for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": offsets,
    }
    return out
