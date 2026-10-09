#!/usr/bin/env bash
# HTCondor job: every run of the CVPR plan on the toy world (collaudo §4.13.1), end to end:
# training, evaluation on the held-out combinations, the run's report, then the comparison.
# It checks the code of every arm and ablation, not their results: the toy world is too
# simple for the arms to differ.
# Usage: ablations_toy.sh <toy world root, built by toyworld.sub> <gpus> [run ...]
#   runs: A V B0 C ESP-6 ESP-2 G1 ESP-4 D4 (all by default)
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
toy="$1"; gpus="$2"; shift 2
runs=("$@")
[[ ${#runs[@]} -gt 0 ]] || runs=(A V B0 C ESP-6 ESP-2 G1 ESP-4 D4)
export HF_HOME="${HF_HOME:-${HOME}/cache/hf}" HF_HUB_OFFLINE=1 TORCH_HOME="${HOME}/cache/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
signworld="${repo_dir}/.venv/bin/signworld"
cd "${repo_dir}"
out="${repo_dir}/runs/toy"
mkdir -p "${out}"

# Stage 0 of the toy world, once: caption embeddings, the index, GloFND's thresholds.
[[ -d "${toy}/text" ]] || "${signworld}" train toyworld-embed --output "${toy}" --device cuda
embeddings="$(ls -d "${toy}"/text/*/ | head -1)"
index="${toy}/index.parquet"
[[ -f "${index}" ]] || "${signworld}" train index --manifest "${toy}/manifest/clips.parquet" \
  --materialized "${toy}" --embeddings "${embeddings}" --output "${index}" \
  -c parameters/model/worldsign.yaml -c parameters/model/toyworld.yaml
negatives="${toy}/negatives"
[[ -d "${negatives}" ]] || "${signworld}" text false-negatives --index "${index}" \
  --embeddings "${embeddings}" --output "${negatives}" --device cuda
data="${out}/toy-data.yaml"
printf 'data: {index: %s, embeddings: %s}\nlosses: {false_negatives: %s}\n' \
  "${index}" "${embeddings%/}" "${negatives}" > "${data}"

declare -A overlays=(
  [A]="arm_A.yaml" [V]="arm_V.yaml" [B0]="arm_B0.yaml" [C]="arm_C.yaml"
  [ESP-6]="arm_A.yaml esp6_video_target.yaml" [ESP-2]="arm_A.yaml esp2_no_physical.yaml"
  [G1]="arm_A.yaml g1_global.yaml" [ESP-4]="arm_A.yaml esp4_latent.yaml"
  [D4]="arm_A.yaml d4_seed1.yaml"
)
compare=()
for name in "${runs[@]}"; do
  configs=(-c parameters/model/worldsign.yaml -c parameters/model/toyworld.yaml -c "${data}")
  for file in ${overlays[${name}]}; do configs+=(-c "parameters/ablation/${file}"); done
  run="${out}/${name}"
  echo "$(date -Is) ${name}: ${configs[*]}"
  "${repo_dir}/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="${gpus}" \
    --no-python "${signworld}" train run "${configs[@]}" --output "${run}"
  "${signworld}" train evaluate "${configs[@]}" --run "${run}" --index "${index}" \
    --split val_channel --clips 1000 --masks 2 --output "${run}/evaluation.json"
  "${signworld}" train report --run "${run}" --evaluation "${run}/evaluation.json" \
    --split val_channel
  compare+=(--run "${run}" --evaluation "${run}/evaluation.json")
done
"${signworld}" train compare "${compare[@]}" --output "${out}/compare"
echo "$(date -Is) done: ${out}"
