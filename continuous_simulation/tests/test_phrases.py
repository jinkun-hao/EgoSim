from __future__ import annotations

import unittest

from egowm_incremental.phrases import (
    DEFAULT_SCENE_PHRASES,
    has_custom_scene_phrases,
    merge_scene_phrases,
    normalize_scene_phrases,
)


class PhraseMergeTest(unittest.TestCase):
    def test_has_custom_scene_phrases_detects_non_default_manual_list(self) -> None:
        self.assertFalse(has_custom_scene_phrases(DEFAULT_SCENE_PHRASES))
        self.assertTrue(has_custom_scene_phrases(["left hand", "white mug"]))

    def test_normalize_scene_phrases_keeps_only_qwen_phrase_list(self) -> None:
        normalized = normalize_scene_phrases(["cup", "Cup", " cup in hand ", "", None])
        self.assertEqual(normalized, ["cup", "cup in hand"])

    def test_merge_scene_phrases_preserves_default_human_terms(self) -> None:
        merged = merge_scene_phrases(["white cup", "hand", "white cup in hand"])
        self.assertEqual(merged[:3], list(DEFAULT_SCENE_PHRASES))
        self.assertIn("white cup", merged)
        self.assertIn("white cup in hand", merged)

    def test_merge_scene_phrases_deduplicates_case_insensitively(self) -> None:
        merged = merge_scene_phrases(["Hand", "ARM", "person", "Cup", "cup"])
        self.assertEqual(merged, ["arm", "person", "hand", "Cup"])


if __name__ == "__main__":
    unittest.main()
