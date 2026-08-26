import argparse
import os
import sys
from typing import Any, Dict, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForTokenClassification
from runner.graphdoc.graphdoc_collator import GraphDocCollator


class GraphDocTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        for key, value in inputs.items():
            if hasattr(value, "to") and hasattr(value, "device"):
                inputs[key] = value.to(self.args.device)
        return inputs


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predicted_labels = np.argmax(predictions, axis=-1)
    active = labels != -100
    predicted_labels = predicted_labels[active]
    labels = labels[active]
    accuracy = (predicted_labels == labels).mean()

    class_f1 = []
    for label_id in range(7):
        true_positive = np.sum((predicted_labels == label_id) & (labels == label_id))
        false_positive = np.sum((predicted_labels == label_id) & (labels != label_id))
        false_negative = np.sum((predicted_labels != label_id) & (labels == label_id))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    return {
        "accuracy": float(accuracy),
        "f1_micro": float(accuracy),
        "f1_macro": float(np.mean(class_f1)),
        "f1_O": float(class_f1[0]),
        "f1_HEADER": float((class_f1[1] + class_f1[2]) / 2),
        "f1_QUESTION": float((class_f1[3] + class_f1[4]) / 2),
        "f1_ANSWER": float((class_f1[5] + class_f1[6]) / 2),
    }


def set_trainable_parameters(model, train_backbone):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = train_backbone or any(
            component in name
            for component in (
                "visual_patch_proj",
                "visual_patch_placeholder",
                "classifier",
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/funsd_graphdoc")
    parser.add_argument("--model-dir", default="pretrained_model/graphdoc")
    parser.add_argument("--output-dir", default="outputs/graphdoc-patch-funsd")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Disable the 196 visual patch tokens for a legacy baseline.",
    )
    parser.add_argument("--skip-sanity-check", action="store_true")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)
    config = GraphDocConfig.from_pretrained(args.model_dir)
    config.use_visual_patch_tokens = not args.legacy
    config.visual_patch_grid_size = 14
    config.num_labels = 7

    model = GraphDocForTokenClassification.from_pretrained(
        args.model_dir,
        config=config,
    )
    set_trainable_parameters(model, args.train_backbone)

    collator = GraphDocCollator()
    if not args.skip_sanity_check:
        batch = collator([dataset["train"][0], dataset["train"][1]])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        batch = GraphDocTrainer(
            model=model,
            args=TrainingArguments(output_dir=os.path.join(args.output_dir, "sanity")),
        )._prepare_inputs(batch)
        model.eval()
        with torch.no_grad():
            sanity_output = model(**batch)
        print("sanity loss:", float(sanity_output.loss.item()))
        print("sanity logits:", tuple(sanity_output.logits.shape))
        model.train()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    trainer = GraphDocTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("saved model:", os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
