#!/usr/bin/env bash
# HTCondor job: fetch one shard of a dataset's media on a compute node.
# Usage: fetch_media.sh <source> <shard> <num_shards> [max_rounds] [round_cooldown_s]
#                       [max_cooldown_s]
#
# ReCaS removes held jobs after 20 minutes, so the job resumes itself for at most max_rounds
# rounds. After a run that ended on its time budget or on failures to retry (exit code 3) it
# waits round_cooldown_s. After a run the host refused (exit code 4) the wait doubles, up to
# max_cooldown_s: in September 2026 retrying every 40 minutes kept YouTube's block on the
# workers alive for days. Exit codes: 0 shard complete, 3 items left, 2 setup error.
set -euo pipefail

source_name="$1"
shard="$2"
num_shards="$3"
max_rounds="${4:-72}"
round_cooldown_s="${5:-3600}"
max_cooldown_s="${6:-86400}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
signworld="${repo_dir}/.venv/bin/signworld"
readonly EXIT_INCOMPLETE=3
readonly EXIT_REFUSED=4

if ! "${repo_dir}/.venv/bin/python" -c '' 2>/dev/null; then
  echo "error: no working environment in ${repo_dir}/.venv; run 'uv sync' there first" >&2
  exit 2
fi

# shellcheck source=youtube_env.sh
source "${repo_dir}/scripts/condor/youtube_env.sh"

cd "${repo_dir}"
cooldown_s="${round_cooldown_s}"
for ((round = 1; round <= max_rounds; round++)); do
  echo "$(date -Is) host=$(hostname) source=${source_name} shard=${shard}/${num_shards} round=${round}/${max_rounds}"
  status=0
  "${signworld}" fetch-media "${source_name}" --shard "${shard}" --num-shards "${num_shards}" \
    || status=$?
  if ((status == EXIT_REFUSED)); then
    cooldown_s=$((cooldown_s * 2 < max_cooldown_s ? cooldown_s * 2 : max_cooldown_s))
  elif ((status == EXIT_INCOMPLETE)); then
    cooldown_s="${round_cooldown_s}"
  else
    exit "${status}"
  fi
  if ((round < max_rounds)); then
    echo "$(date -Is) items remain; next round in ${cooldown_s} s"
    sleep "${cooldown_s}"
  fi
done
exit "${EXIT_INCOMPLETE}"
