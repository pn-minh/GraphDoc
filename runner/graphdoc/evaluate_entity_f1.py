import argparse
import os
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForTokenClassification
from runner.graphdoc.graphdoc_collator import GraphDocCollator


LABEL_NAMES = {
    0: "O",
    1: "HEADER",
    2: "HEADER",
    3: "QUESTION",
    4: "QUESTION",
    5: "ANSWER",
    6: "ANSWER",
}


def extract_entities(labels):
    entities = set()
    active_type = None
    active_start = None

    def close_entity(end):
        nonlocal active_type, active_start
        if active_type is not None:
            entities.add((active_type, active_start, end))
        active_type = None
        active_start = None

    for index, label_id in enumerate(labels):
        if label_id == -100:
            continue
        if label_id == 0:
            close_entity(index)
            continue

        entity_type = LABEL_NAMES.get(int(label_id))
        is_begin = int(label_id) in (1, 3, 5)
        if entity_type is None:
            close_entity(index)
        elif is_begin or active_type != entity_type:
            close_entity(index)
            active_type = entity_type
            active_start = index

    close_entity(len(labels))
    return entities


def safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/funsd_graphdoc")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)["test"]
    collator = GraphDocCollator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    config = GraphDocConfig.from_pretrained(args.checkpoint)
    model = GraphDocForTokenClassification.from_pretrained(
        args.checkpoint,
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    true_positive = Counter()
    predicted = Counter()
    gold = Counter()
    total_true_positive = 0
    total_predicted = 0
    total_gold = 0

    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if hasattr(value, "to") and hasattr(value, "device"):
                    batch[key] = value.to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()

            for predicted_labels, gold_labels in zip(predictions, labels):
                predicted_entities = extract_entities(predicted_labels)
                gold_entities = extract_entities(gold_labels)
                matches = predicted_entities & gold_entities

                total_true_positive += len(matches)
                total_predicted += len(predicted_entities)
                total_gold += len(gold_entities)

                for entity_type, _, _ in matches:
                    true_positive[entity_type] += 1
                for entity_type, _, _ in predicted_entities:
                    predicted[entity_type] += 1
                for entity_type, _, _ in gold_entities:
                    gold[entity_type] += 1

    precision = safe_divide(total_true_positive, total_predicted)
    recall = safe_divide(total_true_positive, total_gold)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    print("checkpoint:", args.checkpoint)
    print("entity_precision:", precision)
    print("entity_recall:", recall)
    print("entity_f1:", f1)
    print("matched_entities:", total_true_positive)
    print("predicted_entities:", total_predicted)
    print("gold_entities:", total_gold)

    for entity_type in ("HEADER", "QUESTION", "ANSWER"):
        type_precision = safe_divide(true_positive[entity_type], predicted[entity_type])
        type_recall = safe_divide(true_positive[entity_type], gold[entity_type])
        type_f1 = safe_divide(
            2 * type_precision * type_recall,
            type_precision + type_recall,
        )
        print(entity_type + "_precision:", type_precision)
        print(entity_type + "_recall:", type_recall)
        print(entity_type + "_f1:", type_f1)


if __name__ == "__main__":
    main()
