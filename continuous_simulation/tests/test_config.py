from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from egowm_incremental.config import load_project_config
from egowm_incremental.paths import PROJECT_ROOT


class ConfigLoadTest(unittest.TestCase):
    def test_relative_paths_resolve_from_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "artifacts" / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "artifacts" / "models" / "Prior").mkdir(parents=True)
            (root / "artifacts" / "models" / "Qwen").mkdir(parents=True)
            (root / "artifacts" / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {Path('/usr/bin/python3')}
                        scene:
                          executable: {Path('/usr/bin/python3')}
                    models:
                      model_root: {root / 'artifacts' / 'models' / 'EgoSim-14B'}
                      qwen_vl_root: ./artifacts/models/Qwen
                      prior_depth_root: ./artifacts/models/Prior
                    data:
                      dataset_root: ./artifacts/data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            self.assertEqual(config.runtime.output_root, (root / "outputs").resolve())
            self.assertTrue(config.models.model_root.exists())
            self.assertTrue(config.backend.predict_phrases_script.exists())
            self.assertEqual(config.quicktest.spatial_subsample, 1)

    def test_default_qwen_vl_root_resolves_from_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "artifacts" / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "artifacts" / "models" / "Prior").mkdir(parents=True)
            (root / "artifacts" / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {Path('/usr/bin/python3')}
                        scene:
                          executable: {Path('/usr/bin/python3')}
                    models:
                      model_root: {root / 'artifacts' / 'models' / 'EgoSim-14B'}
                      prior_depth_root: ./artifacts/models/Prior
                    data:
                      dataset_root: ./artifacts/data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            self.assertEqual(
                config.models.qwen_vl_root,
                (root / ".." / "artifacts" / "models" / "qwen-vl" / "Qwen3-VL-4B-Instruct").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
