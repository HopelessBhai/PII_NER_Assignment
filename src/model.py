from __future__ import annotations

from transformers import AutoConfig, AutoModelForTokenClassification

from labels import ID2LABEL, LABEL2ID


def create_model(model_name: str, dropout: float | None = None):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if dropout is not None:
        for attr in ["hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "dropout"]:
            if hasattr(config, attr):
                setattr(config, attr, dropout)

    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    return model
