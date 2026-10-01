# Paths changed by the move to `parameters/`

Every path in the settings is now relative to the directory the command is launched from
(the repository root): datasets under `data/`, weights and the torch hub code under
`checkpoints/`, secrets under `.secrets/`. They are read as given, not relative to the YAML
file. Adjust them to the machine; nothing else was changed.

| File | Key | Before | Now |
|---|---|---|---|
| `parameters/acquisition/default.yaml` | `data_root` | `/lustrehome/mvigone/data/signworld` | `data` |
| `parameters/acquisition/default.yaml` | `youtube.cookies_file` | `/lustrehome/mvigone/.secrets/youtube_cookies.txt` | `.secrets/youtube_cookies.txt` |
| `parameters/acquisition/default.yaml` | `youtube.pot_server_home` | `/lustrehome/mvigone/bgutil-ytdlp-pot-provider/server` | `third_party/bgutil-ytdlp-pot-provider/server` |
| `parameters/acquisition/default.yaml` | `sources.openasl.local_root` | `/lustrehome/mvigone/data/openASL` | `data/openASL` |
| `parameters/analysis/default.yaml` | `models_root` | `/lustrehome/mvigone/models` | `checkpoints` |
| `parameters/analysis/default.yaml` | `checkpoints.vjepa2_1_vitl_384.path` | `/lustrehome/mvigone/models/vjepa21/vjepa2_1_vitl_384.pt` | `checkpoints/vjepa21/vjepa2_1_vitl_384.pt` |
| `parameters/analysis/default.yaml` | `test_data.root` | `/lustrehome/mvigone/data/datiTest` | `data/datiTest` |
| `parameters/analysis/default.yaml` | `video_probes.hub_repo` | `/lustrehome/mvigone/cache/torch/hub/facebookresearch_vjepa2_main` | `checkpoints/hub/facebookresearch_vjepa2_main` |
| `parameters/model/worldsign.yaml` | `encoder.hub_repo` | `/lustrehome/mvigone/cache/torch/hub/facebookresearch_vjepa2_main` | `checkpoints/hub/facebookresearch_vjepa2_main` |
| `parameters/model/worldsign.yaml` | `encoder.checkpoint` | `/lustrehome/mvigone/models/vjepa21/vjepa2_1_vitl_384.pt` | `checkpoints/vjepa21/vjepa2_1_vitl_384.pt` |
| `parameters/model/worldsign.yaml` | `pose_encoder.checkpoint` | `/lustrehome/mvigone/data/datiTest/youtube_sl25/pose-teachers/sjepa.pt` | `data/datiTest/youtube_sl25/pose-teachers/sjepa.pt` |
| `parameters/model/openasl_trial.yaml` | `data.index` | `/lustrehome/mvigone/data/training/openasl/index.parquet` | `data/training/openasl/index.parquet` |
| `parameters/model/openasl_trial.yaml` | `data.embeddings` | `/lustrehome/mvigone/data/signworld/openasl/text/2fdc70f119730c4c` | `data/openasl/text/2fdc70f119730c4c` |
| `parameters/analysis/default.yaml` | `acquisition_config` | `configs/acquisition.yaml` | `parameters/acquisition/default.yaml` |
| `tests/worldsign.py` | `HUB` (tests that need the V-JEPA 2.1 code) | `/lustrehome/mvigone/cache/torch/hub/facebookresearch_vjepa2_main` | `checkpoints/hub/facebookresearch_vjepa2_main` |

The `.sub` files under `slurm/condor/` keep their absolute `initialdir`.

**Before the next cluster job.** The jobs start in the repository, so `data/` is now the
repository's own, empty folder: `fetch-media` would start downloading again and the analysis
jobs would not find their inputs. On ReCaS the old roots differ (`data_root` was
`/lustrehome/mvigone/data/signworld`, the other data paths were under `/lustrehome/mvigone/data`),
so link each subfolder (`data/youtube_sl25`, `data/openasl`, …, `data/datiTest`, `data/openASL`,
`data/training`) or set the values back, before submitting.
