#!/usr/bin/env bash
# HTCondor job: which way out still reaches YouTube from a worker node. One metadata request
# per video and variant (cookies or not, IPv4 or IPv6); nothing is downloaded.
# Usage: youtube_probe.sh <source> [--video ID ...]
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=youtube_env.sh
source "${repo_dir}/scripts/condor/youtube_env.sh"

cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/signworld" probe-youtube "$@"
