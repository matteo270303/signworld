# signworld

Code for **WorldSign**, an energy-based world model for multilingual sign-language retrieval.

This first milestone covers **data acquisition**: every dataset of §3 of the project document
is obtained reproducibly, resumably and at cluster scale.

## Datasets

| Source | Content | Access | Terms |
|---|---|---|---|
| `youtube_sl25` | 39,197 YouTube videos with manual captions, 25+ sign languages | public (video IDs) | YouTube Terms of Service |
| `openasl` | 98,417 ASL sentence clips from 2,043 YouTube videos | local copy, checked in place | CC BY-NC-ND 4.0 |
| `csl_news` | 1,985 h of Chinese Sign Language news clips (~935 GB) | public (Hugging Face Hub) | CC BY-NC 4.0 |
| `bobsl` | ~1,450 h of BSL-interpreted BBC broadcasts | BBC agreement and personal password | BBC terms of use |
| `phoenix14t` | German weather forecasts in DGS, gloss and translation (41.7 GB) | public | see homepage |
| `csl_daily` | Daily-life Chinese Sign Language, gloss and translation | agreement signed by staff | CSL release agreement |

`uv run signworld sources` prints the same list with homepages.

## Setup

```bash
uv sync
```

YouTube downloads from a datacenter IP additionally need:

- a Netscape cookies file of a logged-in account (`youtube.cookies_file`, kept outside the
  repository with mode 600);
- `deno` on `PATH` and the server of the
  [bgutil proof-of-origin token provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
  (`youtube.pot_server_home`).

BOBSL credentials are read from `BOBSL_USERNAME` and `BOBSL_PASSWORD`.

## Usage

```bash
uv run signworld fetch-metadata csl_news          # annotation files, with provenance
uv run signworld fetch-media csl_news             # media, resuming from the ledgers
uv run signworld status csl_news                  # settled, failing and remaining items
```

Settings live in `configs/acquisition.yaml`; another file can be passed with `--config` or
`SIGNWORLD_CONFIG`.

`fetch-media --limit N` fetches at most `N` items, for pilot runs. The command exits with 0
once every item of the shard is settled and with 3 while some remain.

On ReCaS, media are fetched in parallel shards on compute nodes (heavy downloads from the login
node get its IP flagged by YouTube):

```bash
uv sync
condor_submit -name ettore source=youtube_sl25 shards=4 scripts/condor/fetch_media.sub
```

A job whose shard is still incomplete waits an hour inside the job and runs again, up to 72
rounds, so one submission keeps going until everything is settled (ReCaS removes held jobs after
20 minutes, so Condor's own hold-and-release cannot be used). Stopping a cluster and submitting
with another shard count is safe: items are assigned to shards by a stable hash.

## Manifests and data checks

Each dataset gets its own manifest, `<data_root>/<source>/manifest/clips.parquet`, with one row
per captioned sentence clip; corpora are never merged at this stage. The checks of the
pre-training collaudo (§4.13.1 of the project document) and the preliminary controls (§4.12.1)
write JSON reports to `<data_root>/<source>/reports/`. Settings live in
`configs/analysis.yaml`.

```bash
uv run signworld manifest build openasl
uv run signworld check durations openasl              # caption durations, frame spacing T/32
uv run signworld check split-duplicates openasl       # same caption and duration across splits
uv run signworld check contamination youtube_sl25     # overlap with OpenASL evaluation clips
uv run signworld text embed openasl --device cuda     # EmbeddingGemma, pinned prompt
uv run signworld text verify openasl --device cuda    # prompt fingerprint and re-encoding
uv run signworld check text-geometry openasl          # PC1: geometry per MRL dimension
uv run signworld check checkpoint vjepa2_1_vitl_384   # PC5/PC6: checkpoint contents
```

Long commands belong on a compute node, where they survive a closed session:

```bash
condor_submit -name ettore -a 'arguments = text embed openasl -c configs/analysis.yaml' \
    scripts/condor/analysis.sub
```

EmbeddingGemma is a gated model: accept its licence on Hugging Face and export `HF_TOKEN`, or
leave the token in `~/.secrets/hf_token`, which the Condor wrapper reads.
Captions are encoded without a task prefix, the plain input VL-JEPA reports for its Y-Encoder;
model, resolved commit and prompt form the fingerprint every consumer checks.

For YouTube-SL-25 only the subtitle track written in the video's own language is kept
(`corpus/languages.py` maps each sign language to its written languages); translations into
other spoken languages are excluded by §3.8, and videos whose sign language the release marks
`???` contribute no captioned clip.

The metrics shared with training and evaluation (`signworld.metrics`: recall@k with a
duplicate-tolerant variant and bootstrap intervals, hubness, effective rank, IsoScore, collapse
alarm, SIGReg) are tested on synthetic cases with known answers, as the collaudo requires.

## Reproducibility

- **Pinned releases.** OpenASL annotations by SHA-256 of commit `c7d2350`, CSL-News by Hub
  revision, the YouTube-SL-25 metadata by SHA-256, PHOENIX-2014T by size.
- **Provenance.** Each dataset's `metadata/PROVENANCE.json` lists the origin, SHA-256, size and
  retrieval time of every annotation file.
- **Ledgers.** Every media outcome (`done`, `unavailable`, `failed`, `blocked`) is appended to
  `ledgers/shard-*.jsonl`. Items are assigned to shards by a stable hash of their key, so a
  re-run resumes with any shard count.
- **Safe transfers.** Downloads resume with HTTP Range requests and land under their final name
  only when complete; archives are extracted idempotently and rejected if a member would escape
  the destination.
- **Failures.** An item whose fetch fails `max_attempts` times is abandoned; removed and private
  videos are settled at once.
- **Run budget.** `run_budget_s` ends a run before the host starts refusing (YouTube refuses a
  shard after about an hour of continuous downloading), so the job rests and starts a new run.
- **Refusals.** Refusals (bot wall, HTTP 429, rejected credentials) never count against an item.
  Each consecutive refusal pauses the shard for a doubling cooldown, and `refusals.max_consecutive`
  of them stop it instead of escalating the block.
- **YouTube pacing.** Requests follow yt-dlp's recommended sleep intervals; each job reads a
  private copy of the cookies, rebuilds its client after a refusal (picking up refreshed cookies),
  and abandons a transfer stalled for `item_timeout_s`.

Data are laid out as `<data_root>/<source>/{metadata,raw,extracted,ledgers}`.

## Development

```bash
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```
