import torch
from detectron2.structures import ImageList


class GraphDocCollator:
    def __call__(self, features):
        batch_size = len(features)
        max_regions = max(len(feature["labels"]) for feature in features)
        hidden_size = len(features[0]["inputs_embeds"][0])

        inputs_embeds = torch.zeros(
            batch_size,
            max_regions,
            hidden_size,
            dtype=torch.float32,
        )
        input_sentences_masks = torch.zeros(
            batch_size,
            max_regions,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            batch_size,
            max_regions,
            dtype=torch.long,
        )
        bbox = torch.zeros(
            batch_size,
            max_regions,
            4,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_regions),
            -100,
            dtype=torch.long,
        )

        images = []
        for index, feature in enumerate(features):
            region_count = len(feature["labels"])

            inputs_embeds[index, :region_count] = torch.tensor(
                feature["inputs_embeds"],
                dtype=torch.float32,
            )
            input_sentences_masks[index, :region_count] = torch.tensor(
                feature["input_sentences_masks"],
                dtype=torch.long,
            )
            attention_mask[index, :region_count] = torch.tensor(
                feature["attention_mask"],
                dtype=torch.long,
            )
            bbox[index, :region_count] = torch.tensor(
                feature["bbox"],
                dtype=torch.long,
            )
            labels[index, :region_count] = torch.tensor(
                feature["labels"],
                dtype=torch.long,
            )
            images.append(torch.tensor(feature["image"], dtype=torch.float32))

        return {
            "inputs_embeds": inputs_embeds,
            "input_sentences_masks": input_sentences_masks,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "labels": labels,
            "image": ImageList.from_tensors(images, 32),
        }
