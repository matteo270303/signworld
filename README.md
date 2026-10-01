# 🤟 signworld

Code for **WorldSign**, an energy-based world model for multilingual sign-language retrieval.
Every step, from the raw datasets to the trained model, is one `signworld` command.

## 🛠️ Installation

```bash
uv sync
uv run signworld --help       # or: uv run python main.py --help
```

Commands are run from the repository root: every path in `parameters/` is relative to it
(datasets under `data/`, weights and the V-JEPA 2.1 hub code under `checkpoints/`; see
`parameters/PATHS.md`).

- **YouTube** downloads from a datacenter IP need a Netscape cookies file of a logged-in
  account (`youtube.cookies_file`, default `.secrets/youtube_cookies.txt`, mode 600), `deno`
  on `PATH` and the server of the
  [bgutil proof-of-origin token provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
  (`youtube.pot_server_home`, default `third_party/bgutil-ytdlp-pot-provider/server`).
- **EmbeddingGemma** is gated: accept its licence on Hugging Face and export `HF_TOKEN`, or
  leave the token in `~/.secrets/hf_token`, which the cluster wrappers read.
- **BOBSL** credentials are read from `BOBSL_USERNAME` and `BOBSL_PASSWORD`.

## 📚 Datasets

| Source | Content | Access | Terms |
|---|---|---|---|
| `youtube_sl25` | 39,197 YouTube videos with manual captions, 25+ sign languages | public (video IDs) | YouTube Terms of Service |
| `openasl` | 98,417 ASL sentence clips from 2,043 YouTube videos | local copy, checked in place | CC BY-NC-ND 4.0 |
| `csl_news` | 1,985 h of Chinese Sign Language news clips (~935 GB) | public (Hugging Face Hub) | CC BY-NC 4.0 |
| `bobsl` | ~1,450 h of BSL-interpreted BBC broadcasts | BBC agreement and personal password | BBC terms of use |
| `phoenix14t` | German weather forecasts in DGS, gloss and translation (41.7 GB) | public | see homepage |
| `csl_daily` | Daily-life Chinese Sign Language, gloss and translation | agreement signed by staff | CSL release agreement |

```bash
uv run signworld sources      # the same list, with homepages
```

## ⬇️ Acquisition

Settings: `parameters/acquisition/default.yaml` (`--config` or `SIGNWORLD_CONFIG` for another).
Each dataset lands in `data/<source>/{metadata,raw,extracted,ledgers}`.

```bash
uv run signworld fetch-metadata csl_news                 # annotation files, with provenance
uv run signworld fetch-media csl_news --limit 100        # pilot run: at most 100 items
uv run signworld fetch-media youtube_sl25 --shard 0 --num-shards 4
uv run signworld status csl_news                         # settled, failing, still to fetch
uv run signworld probe-youtube youtube_sl25              # account refusal or network refusal?
```

`fetch-media` exits with 0 once the shard is settled, 3 while items remain and 4 when the host
refused the run; running it again resumes from the ledgers, with any shard count.

## 🧾 Manifests and data checks

Settings: `parameters/analysis/default.yaml` (`--config` or `SIGNWORLD_ANALYSIS_CONFIG`).
One manifest per dataset, `data/<source>/manifest/clips.parquet`; every check writes its JSON
report to `collaudo/<test>/`.

```bash
uv run signworld manifest build openasl
uv run signworld check durations openasl                 # caption durations, frame spacing
uv run signworld check split-duplicates openasl          # same caption across splits
uv run signworld check contamination youtube_sl25        # overlap with the benchmark clips
uv run signworld check checkpoint vjepa2_1_vitl_384      # PC5/PC6: checkpoint contents
```

Caption embeddings (EmbeddingGemma, pinned model, commit and prompt):

```bash
uv run signworld text embed openasl --device cuda
uv run signworld text verify openasl --device cuda       # A2: fingerprint and re-encoding
uv run signworld check text-geometry openasl             # PC1: geometry per MRL dimension
uv run signworld text collaudo openasl --device cuda     # embed + verify + text-geometry
```

Test clips (cut, cropped, 64 frames, poses) and the checks that read them:

```bash
uv run signworld testdata build youtube_sl25 --device cuda --shard 0 --num-shards 4
uv run signworld check frame-selection youtube_sl25      # A3: stored frame indices
uv run signworld check pose-alignment youtube_sl25 --device cuda   # A3: video against pose
uv run signworld check pose-quality youtube_sl25         # A4: shoulders, boxes, contact sheets
uv run signworld check unisign youtube_sl25              # PC5: Uni-Sign pose representation
```

Frozen-model checks:

```bash
uv run signworld check video-reproduction youtube_sl25 --encoder vjepa2_1_vitl   # A1 video
uv run signworld check pose-reproduction youtube_sl25                            # A1 pose
uv run signworld check predictor youtube_sl25 --encoder vjepa2_1_vitl            # PC6
```

## 🔬 Preliminary experiments

```bash
uv run signworld experiment pose-teachers youtube_sl25   # kinematics, Uni-Sign, MAMP, S-JEPA
uv run signworld experiment pose-spectrum youtube_sl25   # S-JEPA spectrum and whitening
uv run signworld experiment pose-isotropy youtube_sl25   # whitening, RBIG, SINF, flows
uv run signworld experiment video-features youtube_sl25 --run vjepa2_1_vitl-256 --shard 0 --num-shards 4
uv run signworld experiment video-probes youtube_sl25    # PC2, PC3, PC4 from the features
```

## 🏋️ Training and evaluation

A run is `parameters/model/worldsign.yaml` plus overlays laid in order: a loss arm or ablation
from `parameters/ablation/`, then a data overlay (`openasl_trial.yaml`, `toyworld.yaml`,
`gate.yaml`).

```bash
# stage 0: clips, poses and the training index
uv run signworld train materialize youtube_sl25 --output data/training/youtube_sl25 --shard 0 --num-shards 32
uv run signworld train index --manifest data/openasl/manifest/clips.parquet \
    --videos data/openASL/videos_256 --poses data/openASL/poses \
    --embeddings data/openasl/text/<fingerprint> --output data/training/openasl/index.parquet \
    -c parameters/model/worldsign.yaml -c parameters/model/openasl_trial.yaml

# training, on every GPU of the node
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 --no-python signworld train run \
    -c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml \
    -c parameters/model/openasl_trial.yaml --output runs/openasl-trial

# test of a trained run
uv run signworld train evaluate -c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml \
    -c parameters/model/openasl_trial.yaml --run runs/openasl-trial \
    --index data/training/openasl/index.parquet --split test --checkpoint final \
    --output runs/openasl-trial/test_final.json
```

Collaudo of the training (§4.13.1):

```bash
uv run signworld train toyworld --output data/toyworld           # synthetic clips
uv run signworld train toyworld-embed --output data/toyworld
uv run signworld train overfit -c parameters/model/worldsign.yaml -c parameters/model/openasl_trial.yaml \
    --output runs/overfit.json                                    # a few clips, every loss on
uv run signworld train benchmark -c parameters/model/worldsign.yaml -c parameters/model/openasl_trial.yaml \
    --output runs/benchmark                                       # PC7: step time, MFU, memory
```

## 🖥️ Cluster

**HTCondor (ReCaS).** Each `.sub` documents its own submission:

```bash
condor_submit -name ettore source=youtube_sl25 shards=4 slurm/condor/fetch_media.sub
condor_submit -name ettore -a 'dataset=youtube_sl25' slurm/condor/youtube_probe.sub
condor_submit -name ettore -a 'arguments = text embed openasl -c parameters/analysis/default.yaml' \
    slurm/condor/analysis.sub                                     # any analysis command
condor_submit -name ettore -a 'dataset=youtube_sl25' -a 'shards=4' slurm/condor/testdata_build.sub
condor_submit -name ettore -a 'arguments = youtube_sl25' slurm/condor/pose_collaudo.sub
condor_submit -name ettore -a 'arguments = youtube_sl25' slurm/condor/model_checks.sub
condor_submit -name ettore -a 'dataset=youtube_sl25' -a 'run=vjepa2_1_vitl-256' -a 'shards=4' \
    slurm/condor/video_features.sub
slurm/condor/submit_video_collaudo.sh youtube_sl25 4              # every video GPU job at once
condor_submit -name ettore -a 'source=youtube_sl25' -a 'output=data/training/youtube_sl25' \
    -a 'shards=32' slurm/condor/materialize.sub
condor_submit -name ettore slurm/condor/openasl_index.sub
condor_submit -name ettore -a 'output=data/toyworld' -a 'shards=16' slurm/condor/toyworld.sub
condor_submit -name ettore -a 'run=worldsign-A' -a 'gpus=2' \
    -a 'configs=parameters/model/worldsign.yaml parameters/ablation/arm_A.yaml' slurm/condor/train.sub
condor_submit -name ettore -a 'run=openasl-trial' \
    -a 'configs=parameters/model/worldsign.yaml parameters/ablation/arm_A.yaml parameters/model/openasl_trial.yaml' \
    slurm/condor/evaluate.sub
```

The `.sub` files still set an absolute `initialdir`: change it to your checkout.

**Slurm.** The `#SBATCH` headers (account, partition, resources) are placeholders; the jobs
read their caches from `$SCRATCH` and write logs to `out/`.

```bash
mkdir -p out
sbatch slurm/launch_fetch_media youtube_sl25 --shard 0 --num-shards 4
sbatch slurm/launch_analysis text embed openasl -c parameters/analysis/default.yaml
sbatch slurm/launch_train -c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml --output runs/worldsign-A
sbatch slurm/launch_evaluate -c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml \
    --run runs/worldsign-A --index data/training/openasl/index.parquet --split test --output runs/worldsign-A/test_best.json
```

## ♻️ Reproducibility

- **Pinned releases**: annotations and metadata by SHA-256, Hub datasets and models by revision.
- **Provenance**: `metadata/PROVENANCE.json` lists origin, SHA-256, size and time of every file.
- **Resumable**: every media outcome goes to `ledgers/shard-*.jsonl`; shards are a stable hash
  of the item key, so any run resumes with any shard count.
- **Polite fetching**: refusals never count against an item, pause the shard with a doubling
  cooldown and stop it after `refusals.max_consecutive`.

The project itself (architecture, losses, ablations) is described in `docs/`.

## 🗂️ Structure

```
signworld/     the package: cli, data, models, loss, metrics, experiment, logger
parameters/    settings: acquisition, analysis, model and ablation YAML files
slurm/         sbatch jobs; slurm/condor/ for HTCondor
tests/         pytest suite, mirroring the package
docs/          project documents (architecture, losses, ablations)
data/          datasets (local, not tracked)
checkpoints/   weights and hub code (local, not tracked)
notebooks/     exploration
collaudo/      check reports (local, not tracked)
runs/          training runs (local, not tracked)
main.py        python main.py <command>, the same as signworld <command>
```
