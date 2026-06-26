from __future__ import annotations

import unittest

import torch

from egowm_incremental.vendor.predict_phrases import (
    HARDCODED_PHRASES,
    SYSTEM_PROMPT,
    _build_in_hand_variants,
    _parse_phrases_from_text,
    predict_phrases,
)


class _FakeInputs(dict):
    def __init__(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]])
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _FakeProcessor:
    def __init__(self, decoded_text: str) -> None:
        self.decoded_text = decoded_text
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return _FakeInputs()

    def batch_decode(self, _generated_ids_trimmed, **_kwargs):
        return [self.decoded_text]


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        return input_ids


class VendorPredictPhrasesTest(unittest.TestCase):
    def test_system_prompt_matches_legacy_ego_interaction_instructions(self) -> None:
        self.assertIn("ego-view video of human hands interacting with objects", SYSTEM_PROMPT)
        self.assertIn("Return ONLY a JSON list of strings", SYSTEM_PROMPT)
        self.assertIn("brown glasses", SYSTEM_PROMPT)

    def test_build_in_hand_variants_adds_object_in_hand_forms(self) -> None:
        phrases = _build_in_hand_variants(["brown glasses", "person", "brown glasses in hand"])
        self.assertEqual(
            phrases,
            ["brown glasses", "brown glasses in hand", "person", "brown glasses in hand"],
        )

    def test_predict_phrases_adds_in_hand_variants_and_hardcoded_human_terms(self) -> None:
        model = _FakeModel()
        processor = _FakeProcessor('["brown glasses", "beige case"]')
        phrases = predict_phrases("dummy.mp4", model=model, processor=processor)
        self.assertEqual(
            phrases,
            [
                "brown glasses",
                "brown glasses in hand",
                "beige case",
                "beige case in hand",
                *HARDCODED_PHRASES,
            ],
        )
        self.assertEqual(
            processor.messages[0]["content"][1]["text"],
            SYSTEM_PROMPT,
        )

    def test_parse_phrases_from_text_accepts_json_and_fallback_lines(self) -> None:
        self.assertEqual(
            _parse_phrases_from_text('["brown glasses", "beige case"]'),
            ["brown glasses", "beige case"],
        )
        self.assertEqual(
            _parse_phrases_from_text("1. brown glasses\n2. beige case"),
            ["brown glasses", "beige case"],
        )


if __name__ == "__main__":
    unittest.main()
