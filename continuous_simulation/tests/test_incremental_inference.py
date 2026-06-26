from __future__ import annotations

import unittest

from egowm_incremental.backends.incremental_inference import should_reuse_existing_generated_video


class IncrementalInferenceLogicTest(unittest.TestCase):
    def test_reuses_existing_video_when_reconstruction_is_enabled(self) -> None:
        self.assertTrue(
            should_reuse_existing_generated_video(
                out_video_exists=True,
                skip_existing=True,
                egosim_state_only=False,
                run_egosim_state=True,
            )
        )

    def test_does_not_reuse_existing_video_when_reconstruction_is_disabled(self) -> None:
        self.assertFalse(
            should_reuse_existing_generated_video(
                out_video_exists=True,
                skip_existing=True,
                egosim_state_only=False,
                run_egosim_state=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
