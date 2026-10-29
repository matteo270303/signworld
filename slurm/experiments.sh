# Gli esperimenti di docs/worldsign-ablation.md: nome -> file di configurazione, in ordine.
# Letto da slurm/launch_experiment_exe e da slurm/batch/*.sh (source, non si esegue).
#
# ESP-1 (prima ondata, in parallelo): A-gate B0 B C V. A-gate è il braccio A con gli stop F1-F3.
# Su θ* (seconda ondata): esp2 esp4 esp6 g1 d4, sopra il braccio THETA (default A finché ESP-1
# non decide); la cartella del run porta il braccio, es. runs/esp2-on-A.

BASE=(parameters/model/worldsign.yaml parameters/model/youtube_sl25.yaml parameters/model/youtube_sl25_run.yaml)
THETA=${THETA:-A}
EXPERIMENTS=(A-gate A B0 B C V esp2 esp4 esp6 g1 d4)
WAVE1=(A-gate B0 B C V)
WAVE2=(esp2 esp4 esp6 g1 d4)
# `pilot` non sta in EXPERIMENTS: lo lancia solo slurm/batch/pilot.sh.
# Una config per memoria diversa (B0 pesa come B, d4 e A come A-gate): 8 nodi, il massimo di boost_qos_dbg.
DEBUG_SET=(A-gate B C V esp2 esp4 esp6 g1)

experiment_files() {  # nome -> file di configurazione, uno per riga
  local arm=parameters/ablation/arm_$THETA.yaml files
  case $1 in
    A-gate) files=(parameters/ablation/arm_A.yaml parameters/model/gate.yaml) ;;
    A | B0 | B | C | V) files=("parameters/ablation/arm_$1.yaml") ;;
    esp2) files=("$arm" parameters/ablation/esp2_no_physical.yaml) ;;
    esp4) files=("$arm" parameters/ablation/esp4_latent.yaml) ;;
    esp6) files=("$arm" parameters/ablation/esp6_video_target.yaml) ;;
    g1) files=("$arm" parameters/ablation/g1_global.yaml) ;;
    d4) files=("$arm" parameters/ablation/d4_seed1.yaml) ;;
    pilot) files=(parameters/ablation/arm_A.yaml parameters/model/pilot.yaml) ;;
    *)
      echo "esperimento sconosciuto: $1 (noti: ${EXPERIMENTS[*]})" >&2
      return 1
      ;;
  esac
  printf '%s\n' "${BASE[@]}" "${files[@]}"
}

run_dir() {  # debug|train, nome -> cartella del run
  local root=runs
  [ "$1" = debug ] && root=runs/debug
  case $2 in
    esp2 | esp4 | esp6 | g1 | d4) echo "$root/$2-on-$THETA" ;;
    pilot) echo "$root/pilot-A" ;;
    *) echo "$root/$2" ;;
  esac
}
