import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoModel, AutoTokenizer

from layoutlmft.data.datasets.funsd import Funsd


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-script",
        default="layoutlmft/data/datasets/funsd.py",
    )
    parser.add_argument(
        "--sentence-model",
        default="pretrained_model/sentence-bert",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/funsd_graphdoc",
    )
    parser.add_argument("--max-sentence-length", type=int, default=32)
    args = parser.parse_args()

    if args.dataset_script == "layoutlmft/data/datasets/funsd.py":
        builder = Funsd()
        builder.download_and_prepare()
        dataset = builder.as_dataset()
    else:
        dataset = load_dataset(
            args.dataset_script,
            name="funsd",
            trust_remote_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.sentence_model)
    sentence_model = AutoModel.from_pretrained(args.sentence_model).to(device).eval()

    def convert_example(example):
        encoded = tokenizer(
            example["tokens"],
            padding=True,
            truncation=True,
            max_length=args.max_sentence_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            sentence_output = sentence_model(**encoded)
            embeddings = mean_pooling(
                sentence_output,
                encoded["attention_mask"],
            ).cpu()

        region_count = len(example["tokens"])
        embeddings = torch.cat(
            [torch.zeros(1, embeddings.shape[-1]), embeddings],
            dim=0,
        )
        boxes = torch.tensor(
            [[0, 0, 1000, 1000]] + example["bboxes"],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [-100] + example["ner_tags"],
            dtype=torch.long,
        )
        region_mask = torch.ones(region_count + 1, dtype=torch.long)

        return {
            "inputs_embeds": embeddings.tolist(),
            "input_sentences_masks": region_mask.tolist(),
            "attention_mask": region_mask.tolist(),
            "bbox": boxes.tolist(),
            "labels": labels.tolist(),
            "image": example["image"],
        }

    processed = {}
    for split in ("train", "test"):
        processed[split] = dataset[split].map(
            convert_example,
            remove_columns=dataset[split].column_names,
            desc="Creating GraphDoc inputs for " + split,
        )

    first = processed["train"][0]
    assert len(first["inputs_embeds"]) == len(first["bbox"])
    assert len(first["bbox"]) == len(first["labels"])
    assert len(first["labels"]) == len(first["attention_mask"])
    assert len(first["inputs_embeds"][0]) == 768
    assert len(first["image"]) == 3
    assert len(first["image"][0]) == 224
    assert len(first["image"][0][0]) == 224
    assert all(
        0 <= coordinate <= 1000
        for box in first["bbox"]
        for coordinate in box
    )
    assert first["labels"][0] == -100

    processed = DatasetDict(processed)
    processed.save_to_disk(args.output_dir)
    print(processed)
    print("saved to:", args.output_dir)
    print("sample regions:", len(first["labels"]))


if __name__ == "__main__":
    main()