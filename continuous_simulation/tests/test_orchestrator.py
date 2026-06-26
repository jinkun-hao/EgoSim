from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from egowm_incremental.config import load_project_config
from egowm_incremental.orchestrator import prepare_quicktest_run
from egowm_incremental.paths import PROJECT_ROOT


class OrchestratorTest(unittest.TestCase):
    def test_recon_visualize_mode_defaults_to_qwen_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "models" / "Prior").mkdir(parents=True)
            (root / "models" / "Qwen").mkdir(parents=True)
            (root / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {sys.executable}
                        scene:
                          executable: {sys.executable}
                    models:
                      model_root: {root / 'models' / 'EgoSim-14B'}
                      qwen_vl_root: ./models/Qwen
                      prior_depth_root: ./models/Prior
                    data:
                      dataset_root: ./data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            run = prepare_quicktest_run(config, mode="recon_visualize")
            command = " ".join(run.command)
            self.assertIn("--incremental_mode recon_visualize", command)
            self.assertIn("--predict_phrases", command)
            self.assertIn("--qwen_model_path", command)
            self.assertNotIn("--no_predict_phrases", command)
            self.assertIn("--scene_phrases arm person hand", command)
            self.assertIn("--spatial_subsample 2", command)
            self.assertIn("incremental_quicktest_reconviz_", str(run.output_dir))

    def test_recon_visualize_mode_allows_explicit_qwen_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "models" / "Prior").mkdir(parents=True)
            (root / "models" / "Qwen").mkdir(parents=True)
            (root / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {sys.executable}
                        scene:
                          executable: {sys.executable}
                    models:
                      model_root: {root / 'models' / 'EgoSim-14B'}
                      qwen_vl_root: ./models/Qwen
                      prior_depth_root: ./models/Prior
                    data:
                      dataset_root: ./data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            run = prepare_quicktest_run(
                config,
                mode="recon_visualize",
                extra_args=["--no_predict_phrases"],
            )
            command = run.command
            self.assertNotIn("--qwen_model_path", command)
            self.assertNotIn("--predict_phrases", command)
            self.assertIn("--no_predict_phrases", command)
            self.assertIn("--scene_phrases", command)
            self.assertIn("arm", command)
            self.assertIn("person", command)
            self.assertIn("hand", command)
            self.assertIn("--spatial_subsample", command)
            self.assertEqual(command[command.index("--spatial_subsample") + 1], "2")
            self.assertIn("--incremental_mode", command)
            self.assertEqual(command[command.index("--incremental_mode") + 1], "recon_visualize")
            self.assertIn("incremental_quicktest_reconviz_", str(run.output_dir))

    def test_full_mode_defaults_to_qwen_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "models" / "Prior").mkdir(parents=True)
            (root / "models" / "Qwen").mkdir(parents=True)
            (root / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {sys.executable}
                        scene:
                          executable: {sys.executable}
                    models:
                      model_root: {root / 'models' / 'EgoSim-14B'}
                      qwen_vl_root: ./models/Qwen
                      prior_depth_root: ./models/Prior
                    data:
                      dataset_root: ./data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            run = prepare_quicktest_run(config, mode="full")
            command = " ".join(run.command)
            self.assertIn("--predict_phrases", command)
            self.assertIn("--qwen_model_path", command)
            self.assertNotIn("--no_predict_phrases", command)
            self.assertIn("--egosim_state_python", command)
            self.assertIn("--scene_phrases arm person hand", command)
            self.assertIn("--spatial_subsample 1", command)
            self.assertIn("incremental_quicktest_full_", str(run.output_dir))

    def test_custom_scene_phrases_are_forwarded_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "models" / "Prior").mkdir(parents=True)
            (root / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {sys.executable}
                        scene:
                          executable: {sys.executable}
                    models:
                      model_root: {root / 'models' / 'EgoSim-14B'}
                      prior_depth_root: ./models/Prior
                    data:
                      dataset_root: ./data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    quicktest:
                      scene_phrases:
                        - left hand
                        - white mug
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            run = prepare_quicktest_run(config, mode="recon_visualize")
            command = " ".join(run.command)
            self.assertIn("--scene_phrases left hand white mug", command)
            self.assertIn("--no_predict_phrases", command)

    def test_custom_scene_phrases_disable_qwen_even_when_flag_is_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models" / "EgoSim-14B").mkdir(parents=True)
            (root / "models" / "Prior").mkdir(parents=True)
            (root / "models" / "Qwen").mkdir(parents=True)
            (root / "data").mkdir(parents=True)

            config_path = root / "project.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    runtime:
                      output_root: ./outputs
                      python:
                        wan:
                          executable: {sys.executable}
                        scene:
                          executable: {sys.executable}
                    models:
                      model_root: {root / 'models' / 'EgoSim-14B'}
                      qwen_vl_root: ./models/Qwen
                      prior_depth_root: ./models/Prior
                    data:
                      dataset_root: ./data
                      metadata_path: {PROJECT_ROOT.parent / 'tests' / 'samples' / 'mini_sample' / 'continuous_generation' / 'metadata.csv'}
                    quicktest:
                      scene_phrases:
                        - left hand
                        - white mug
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_project_config(config_path)
            run = prepare_quicktest_run(config, mode="recon_visualize", extra_args=["--predict_phrases"])
            command = " ".join(run.command)
            self.assertIn("--scene_phrases left hand white mug", command)
            self.assertIn("--no_predict_phrases", command)
            self.assertNotIn("--qwen_model_path", command)


if __name__ == "__main__":
    unittest.main()
