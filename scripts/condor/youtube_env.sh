# Sourced by the YouTube jobs: the environment the proof-of-origin token provider needs, and
# a record of the addresses this node reaches the internet from.

# deno runs the YouTube proof-of-origin token provider. Its compilation cache stays shared,
# while the provider's own cache file moves to the job's scratch directory, because parallel
# jobs would overwrite each other's copy.
export PATH="${HOME}/.deno/bin:${PATH}"
export DENO_DIR="${DENO_DIR:-${HOME}/.cache/deno}"
job_scratch="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"
export XDG_CACHE_HOME="${job_scratch}/cache"
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}"

# YouTube rate-limits by address: record which one this node reaches the internet from.
public_ipv4="$(curl -4 -s --max-time 10 https://api.ipify.org || echo unknown)"
public_ipv6="$(curl -6 -s --max-time 10 https://api6.ipify.org || echo none)"
echo "$(date -Is) host=$(hostname) public addresses: ipv4=${public_ipv4} ipv6=${public_ipv6}"
