from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from egowm_incremental.backends.egosim_state_subprocess import parse_args


class EgoSimStateSubprocessDefaultsTest(unittest.TestCase):
    def test_defaults_match_legacy_quicktest_sensitive_filters(self) -> None:
        argv = [
            "egosim_state_subprocess.py",
            "--generated_video",
            "dummy.mp4",
            "--output_dir",
            "out",
            "--output_memory",
            "memory.npz",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.spatial_subsample, 1)
        self.assertEqual(args.tsdf_voxel_size, 0.0025)
        self.assertTrue(args.filter_body_parts)
        self.assertTrue(args.remove_small_clusters)


if __name__ == "__main__":
    unittest.main()
