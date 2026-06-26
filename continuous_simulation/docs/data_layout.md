# Data layout

Incremental inference resolves every relative path in the metadata CSV against `data.dataset_root` (the `--dataset_root` passed through `egowm_incremental.cli`).

## Dataset root layouts

The loader supports both of the following (and matches the original `open_source_incremental` Egodex tree as well as the bundled mini-sample):

1. **Egodex-style parent** — `dataset_root` is the dataset root that contains `process_result/`, optional `test_*/` video trees, and HDF5 shards:

   ```text
   <dataset_root>/
     process_result/…
     test_3dmem/…
     test_16fps_720p/…   # optional
   ```

2. **Process-result-only** — `dataset_root` points at the folder that *is* `process_result` (all clip assets live directly underneath):

   ```text
   <dataset_root>/
     test_3dmem/add_remove_lid/14/0_60/…
     test_3dmem/add_remove_lid/14/14_0_120.hdf5
     test_3dmem/add_remove_lid/GT_14_0_60.mp4
   ```

For each metadata path, the backend tries, in order:

- `<dataset_root>/<csv_path>`
- If `csv_path` starts with `process_result/`, also `<dataset_root>/<csv_path without that prefix>`
- Otherwise also `<dataset_root>/process_result/<csv_path>`

The first path that exists wins. If none exist, the first candidate is used (so error messages remain stable).

## Required metadata columns

- `video`
- `ego_prior_video`
- `hand_keypoint_video`
- `first_frame`
- `prompt`
- `task_name`
- `part_idx`
- `process_result_dir`
- `hdf5_path`
- `gt_process_result_dir`

If `gt_process_result_dir` resolves to a path that does not exist, it falls back to `process_result_dir` for the same clip.

## Example mini-sample

Bundled CSV:

- `../tests/samples/mini_sample/continuous_generation/metadata.csv`

Default `data.dataset_root` in `configs/project.yaml`:

- `../tests/samples/mini_sample/continuous_generation/process_result`

Example row (paths omit the redundant `process_result/` prefix because `dataset_root` already is that folder):

- `video = test_3dmem/add_remove_lid/GT_14_0_60.mp4` (optional `GT_` prefix on the clip stem; grouping strips it when parsing `video_id` / frame range)
- `process_result_dir = test_3dmem/add_remove_lid/14/0_60`
- `hdf5_path = test_3dmem/add_remove_lid/14/14_0_120.hdf5`

## Metadata location

The CSV does not need to live under `dataset_root`. Configure `data.metadata_path` to point at any readable file.
