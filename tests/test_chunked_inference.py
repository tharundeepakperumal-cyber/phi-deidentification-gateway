from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.model_inference import extract_entities


class FakeTokenizer:
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([
                [101, 11, 12, 102],
                [101, 13, 14, 102],
            ]),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "offset_mapping": torch.tensor([
                [[0, 0], [0, 5], [6, 11], [0, 0]],
                [[0, 0], [12, 15], [16, 19], [0, 0]],
            ]),
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            id2label={
                0: "O",
                1: "B-NAME",
                2: "I-NAME",
            }
        )
        self.batch_sizes = []
        self.predictions = [
            [0, 1, 2, 0],
            [0, 0, 1, 0],
        ]

    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        self.batch_sizes.append(batch_size)

        logits = torch.zeros((1, 4, 3))

        for position, prediction in enumerate(
            self.predictions[len(self.batch_sizes) - 1]
        ):
            logits[0, position, prediction] = 10.0

        return SimpleNamespace(logits=logits)


def test_chunked_inference_processes_one_chunk_at_a_time():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    with patch(
        "src.model_inference.load_model",
        return_value=(tokenizer, model),
    ):
        entities = extract_entities("Alice Smith met Bob")

    assert model.batch_sizes == [1, 1]

    assert [
        (entity["text"], entity["start"], entity["end"])
        for entity in entities
    ] == [
        ("Alice Smith", 0, 11),
        ("Bob", 16, 19),
    ]
