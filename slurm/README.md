# Run su Leonardo

Un solo script: `slurm/launch_experiment.sh` (aiuto: lancialo senza argomenti). Ogni esperimento
occupa **un nodo Booster da 4 A100 64 GB** (batch 128 = 32 clip per GPU, DDP). Esperimenti diversi
vanno in parallelo su nodi diversi. Gli esperimenti sono elencati in `slurm/experiments.sh`.

| Nome | Config | Ondata |
|---|---|---|
| `A-gate` | braccio A + stop F1-F3 (`gate.yaml`) | 1 (run di gate) |
| `B0`, `B`, `C`, `V` | bracci di ESP-1 | 1 |
| `esp2`, `esp4`, `esp6`, `g1`, `d4` | ablation sopra `THETA` (default A) | 2, su θ\* |
| `A` | braccio A senza stop | — |

`wave1` = `A-gate B0 B C V`.

## 0. Dalla macchina locale, prima di entrare

```bash
git push origin refactor/structure        # il clone su Leonardo deve avere questi file
scripts/sync_dataset.sh -p 8 -c           # se non è già fatto: dati in /leonardo_scratch/large/userexternal/ecolonna
```

## 1. Login node: setup (nessun costo)

```bash
mkdir -p $WORK/$USER && cd $WORK/$USER
git clone -b refactor/structure https://github.com/matteo270303/signworld.git && cd signworld
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh    # poi riapri la shell
export UV_CACHE_DIR=$WORK/$USER/uv-cache   # i wheel di torch fuori da $HOME (50 GB)
slurm/launch_experiment.sh setup
saldo                                       # budget prima di partire
```

`setup` fa, nell'ordine:

1. `uv sync --frozen`.
2. I link `data/youtube-sl-25` e `data/training` verso `$SCRATCH` (`DATA_ROOT=` per cambiarli).
3. Il codice hub di V-JEPA 2 al commit `204698b`.
4. I pesi V-JEPA 2.1 (~5 GB) con il controllo dello SHA-256.
5. L'indice di debug.
6. La risoluzione delle config di tutti gli 11 esperimenti.

Si può rilanciare: salta quello che c'è già.

**Perché `$WORK`.** `runs/` (checkpoint) deve durare tutto il run. `$SCRATCH` cancella i file non
letti da 40 giorni e `$HOME` ha 50 GB. I dati restano in `$SCRATCH`, dove il training li legge
a ogni epoca.

## 2. Debug del carico (~1 ora, 8 nodi)

```bash
slurm/launch_experiment.sh debug            # A-gate B C V esp2 esp4 esp6 g1, un nodo ciascuno
squeue -u $USER                             # due job sw-debug: il secondo parte dopo il primo
slurm/launch_experiment.sh report           # quando sono finiti entrambi
```

**Cosa gira.** Il `train run` vero: stesso modello, 4 GPU in DDP, 32 clip per GPU, stessi frame.
Cambia solo l'indice, ridotto a 16 passi per epoca con un'epoca per stage
(`parameters/model/debug_cluster.yaml`). Ogni esperimento attraversa P, F0 e F.

- Il primo job da 30 min si ferma 6 minuti prima della fine e salva `latest`.
- Il secondo job riprende da lì. Così il debug prova sul cluster anche il resume: nel log del
  secondo job c'è `[run] resuming at step N`.

Gli 8 esperimenti coprono ogni configurazione con memoria diversa. B0 pesa come B, A e d4 come
A-gate. Sono 8 nodi, il massimo per job di `boost_qos_dbg`.

**Cosa legge `report`** per ogni esperimento:

- Verdetto: `OK` / `RISCHIO OOM` / `OOM`. È `RISCHIO OOM` quando una GPU supera il 90 % della
  memoria (nvidia-smi, `runs/debug/<esp>/gpu.csv`, ogni 30 s).
- Secondi per passo (mediana) e picco allocato sulla GPU 0 per stage. Il picco è cumulativo e
  include il preflight P13, che fa già 30 passi completi di stage F. In locale (Blackwell,
  32 clip, braccio A) vale 48,3 GiB: su una A100 da 64 GB resterebbero ~14 GiB di margine, da
  confermare sul cluster.
- Durata del run vero, contando solo i passi.

| Esito | Cosa fare |
|---|---|
| `OK` | si lancia |
| `RISCHIO OOM` / `OOM` | non lanciarlo. Strade: backward per ramo (`docs/worldsign-gpu.md`, codice da scrivere) o 2 nodi (serve il lancio multi-nodo) |
| `stage F non raggiunto` | `slurm/launch_experiment.sh debug <esp>`: riprende da dove era rimasto |
| `FAILED` | leggi `out/sw-debug-<esp>-*.log`, poi `rm runs/debug/<esp>/FAILED` |

`boost_qos_dbg` ammette 2 job per utente fra running e pending: un nuovo `debug` va lanciato
quando i due precedenti sono finiti. Per rifare un debug da zero: `rm -rf runs/debug/<esp>`.

## 3. Produzione

```bash
slurm/launch_experiment.sh train A-gate                 # run di gate
grep "\[stop\] F2" out/sw-train-A-gate-*.log           # F2 passato: fine dello stage F0
slurm/launch_experiment.sh train B0 B C V               # il resto della prima ondata
# oppure tutta la prima ondata subito:  slurm/launch_experiment.sh train wave1
THETA=B slurm/launch_experiment.sh train esp2 esp6 g1 d4   # seconda ondata, θ* deciso
```

`docs/worldsign-ablation.md` chiede i bracci dopo F2 della run di gate (fail-fast). Lanciare
`wave1` subito guadagna giorni sulla scadenza CVPR, ma rischia 5 run invece di 1 se il gate fallisce.

**Resume.** `boost_qos_lprod` dura **4 giorni**, non 2 (pagina Leonardo della doc CINECA). Ha un
massimo di 8 nodi / 32 GPU per utente e per progetto: la prima ondata ne usa 5. Con `train` ogni
esperimento è una catena di `SEGMENTS` job (default 6, cioè 24 giorni), legati da `afterok`.

- Ogni job si ferma `STOP_MARGIN_MIN` minuti (default 45) prima del walltime e salva `latest`.
  Il trainer lo fa da sé (`SIGNWORLD_STOP_AT`), e il job esce con 0.
- Il job seguente riprende dallo stesso passo, nello stesso ordine di clip.
- A run finito (`final.pt`), i segmenti rimasti escono subito.
- Un errore (OOM, NaN, gate fallito) scrive `runs/<esp>/FAILED` e cancella il resto della catena.
- Calcolo dei segmenti: giorni stimati da `report` / 3,9, più 1 di margine.
- Con la QOS normale da 24 h: `QOS=normal TIME=24:00:00 SEGMENTS=25 slurm/launch_experiment.sh train ...`

**Non cambiare il codice durante un run.** Un resume con codice diverso viene rifiutato
(provenance). Se serve davvero: `ALLOW_CODE_CHANGE=1`; la modifica viene registrata.

## 4. Controllo

```bash
slurm/launch_experiment.sh status       # coda + ultimo passo di ogni run (stage, s/passo, GiB, ETA)
tail -f out/sw-train-A-gate-<job>.log
slurm/launch_experiment.sh cancel C     # tutti i job di C, catena compresa
saldo
```

Variabili: `ACCOUNT` (default `IscrC_SPARTAN`), `MAIL` (mail a fine o errore di ogni job),
`QOS`, `TIME`, `SEGMENTS`, `STOP_MARGIN_MIN`, `THETA`, `DATA_ROOT`.

## Costi

| Cosa | Nodo-ore |
|---|---|
| debug (8 nodi × 2 × 30 min) | 8 (32 GPU-ore) |
| un run, stima non misurata (`.bemyagent/docs/drafts/ytsl25-risorse-slurm.md`) | ~410–530 |

La doc CINECA non dice quanto costa una GPU-ora Booster: controllare `saldo` prima e dopo il debug.

## Non ancora provato su Leonardo

Provato in locale: il trainer che si ferma alla scadenza e riprende (test + run vero su GPU),
l'indice di debug, le config di tutti gli esperimenti, i comandi `sbatch` generati. Da verificare
al primo debug:

- il formato della fine del job da `squeue -o %e`. Se fallisce, il log dice `ATTENZIONE: fine del
  job non letta`;
- un `srun` per nodo nello stesso job;
- che il progetto possa usare `boost_qos_lprod`;
- `uv sync` entro i 10 minuti di CPU del login node. Se viene ucciso, si rilancia.

## Aperti

- `docs/worldsign-ablation.md` elenca A₀ in ESP-1, ma `arm_A0.yaml` non esiste e il codice non
  ha il braccio (`Arm = A, V, B0, B, C`). Al suo posto c'è V.
- Il braccio C gira senza le soglie GloFND (`losses.false_negatives: null`): esclude solo le
  clip dello stesso video.
