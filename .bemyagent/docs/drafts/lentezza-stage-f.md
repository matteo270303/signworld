# Perché lo stage F è lento, e quanto costa un'epoca

## Misure del 23/11/2026 (task 1.8, `signworld train profile`)

Run `runs/youtube_sl25-A-1gpu-b40`: 945.688 clip di train, batch 40, 23.642 passi per epoca.
Nello stage P misura **3,34 s/passo di wall-clock, cioè 21,9 h/epoca**. È l'epoca più leggera.

**Dati: non sono il collo di bottiglia.** Una clip letta da sola costa 19,5 ms, di cui 13,6 ms (70 %)
per decodificare i 64 frame con decord. Il loader del run (16 worker) dà ~1.130 clip/s, contro le
~12 clip/s che la GPU consuma: un margine di circa 100×.

**GPU, per sezione** (start-end con `cuda.synchronize`; batch 4, GPU 0 condivisa col run al 100 %:
contano i rapporti, non i tempi assoluti):

| sezione | P | F0 | F |
|---|---|---|---|
| passo (s, b4) | 0,756 | 1,570 (2,1×) | 2,065 (2,7×) |
| encoder intero senza grad (semantico) | **69 %** | 32 % | 23 % |
| encoder mascherato ×2 (fisico, forward) | — | 13 % | 9 % |
| predittore fisico ×2 (forward) | — | 11 % | 7 % |
| backward (in F con il ricalcolo del checkpointing) | 10 % | 35 % | **54 %** |
| posa, query, testo, augment, AdamW, copia su GPU | ~20 % | ~6 % | ~6 % |

Su phoenix14t, a GPU libera, F/P valeva 3,7× (sotto). Quindi **un'epoca di F su YouTube-SL-25 a
batch 40 dura circa 60-80 h, non 24**. Il piano a 15 epoche (2 P + 1 F0 + 12 F) fa ~34-44 giorni su
una GPU. Da confermare a GPU libera:

```bash
uv run --no-sync signworld train profile -c parameters/model/worldsign.yaml \
  -c parameters/model/youtube_sl25.yaml -c parameters/ablation/arm_A.yaml \
  -c parameters/model/youtube_sl25_run.yaml -c parameters/model/local_single_gpu_max.yaml \
  --output tmp/profile-b40      # copiarci prima runs/youtube_sl25-A-1gpu-b40/statistics.pt
```

**Esperimenti sulle leve** (stessa GPU condivisa, batch 2):

- Activation checkpointing in F: con 1,20 s/passo e 4,7 GiB, senza 0,97 s e 24,6 GiB.
  Toglierlo vale -19 %, ma costa ~10 GiB per clip invece di ~2: a 96 GB starebbero ~8 clip.
- `torch.compile` sul forward dell'encoder senza grad: 214 → 186 ms (1,15×). Il guadagno è limitato
  dal context `torch.backends.cuda.sdp_kernel()` in ogni blocco Meta, che esaurisce il limite di
  ricompilazione.
- SDPA a 8.192 token: flash fa 3,8 ms in forward e 12,8 ms in fwd+bwd; cuDNN 3,8 e 16,2;
  efficient 7,9 e 35. Flash è già il più veloce: l'ipotesi 2 (backend sbagliato) cade, almeno
  per l'encoder.

### Leve, in ordine di guadagno

1. **Quante epoche di F.** Ogni epoca di F tolta risparmia 60-80 h. `epochs: 15` è
   `[Aperto: PC7]`, da fissare sul costo per passo, e c'è già `patience: 3`. È la leva più grande,
   ed è una scelta di progetto.
2. **La passata semantica sulla clip intera** (8.192 token, attention quadratica): vale il 69 % di
   P e il ~23 % di F. Meno frame o meno risoluzione solo per questa passata (4.096 token → ~2,3×)
   cambia il metodo (§3.6): decisione dell'utente.
3. **Checkpointing selettivo** (solo su una parte dei blocchi): fra 0 e -19 % in F, pagato in
   memoria. Da misurare con `profile` alla stessa dimensione di batch.
4. **`torch.compile` dell'encoder**, dopo aver tolto il context `sdp_kernel` per blocco (come fa
   `drop_cudnn_attention`): ≥ 1,15× sulle passate dell'encoder, cioè circa -10 % del passo. Va
   verificato con P13 e con le tolleranze numeriche.
5. **Non conviene**: dataloader (margine 100×, i worker si possono anzi ridurre), backend SDPA,
   AdamW, augment e copia su GPU (< 3 %), diagnostiche (stima ~2-3 % dell'epoca: validazione ogni
   4.000 passi, rare ogni 16.000).

Le leve 3 e 4 insieme, che non cambiano il metodo, valgono circa -20 % in F (stima), cioè ~48-64 h/epoca.

### Batch 40, GPU libera (23/11/2026): la leva 4 è fatta (task 1.9)

| stage | eager s/passo | compilato | h/epoca eager → compilato |
|---|---|---|---|
| P | 3,32 | 2,58 (-22 %) | 21,8 → 16,9 |
| F0 | 8,90 | 6,26 (-30 %) | 58,4 → 41,1 |
| F | 12,49 | 8,37 (-33 %) | 82,0 → 54,9 |

In eager, F/P vale 3,8× a batch 40: confermata la stima alta, 82 h per epoca di F. Il piano a 15
epoche passa da ~1.090 h (~45 giorni) a ~730 h (~31 giorni) su una GPU. La leva 3 (checkpointing
selettivo) e le leve 1-2 restano aperte: l'utente le ha escluse perché toccano il modello.

---

## Diagnosi di ottobre (ipotesi; la 2 è smentita per l'encoder, la 5 è confermata)

Run `runs/phoenix14t-A` (1 GPU, batch 32, fermata allo step 1.320). Dati da `metrics.jsonl`:

| Stage | Cosa gira (curriculum.py, video_branch.py) | s/step (solo passo) | s/step (con validazione) | Memoria |
|---|---|---|---|---|
| P | encoder ViT-L sulla clip intera **senza gradiente** + posa + semantico | 2,84 | 4,2 | 48,3 GiB |
| F0 | + 2 passate dell'encoder sui token visibili (maschere 8×0,15 e 2×0,7), ancora senza gradiente, + predittore fisico fwd/bwd | 7,23 | 7,25 | 48,3 GiB |
| F | + LoRA del video e del predittore: **backward attraverso il ViT-L** nelle 2 passate di contesto (con activation checkpointing) | 10,39 | 12,49 | 54,1 GiB |

## Ipotesi, in ordine di probabilità

1. **Strutturale, atteso (la più probabile).** Ogni stage aggiunge lavoro. Stima dei FLOP in avanti per clip:
   ViT-L 24×1024 ≈ 0,6 GFLOP/token di parte lineare, più l'attention 4·n²·d per blocco.
   - passata intera (n = 8.192): ~5 TF lineare + ~6,6 TF attention ≈ **11,5 TF**
   - contesto 8×0,15 (37,5 % visibile, n ≈ 3.072): ≈ 2,8 TF
   - contesto 2×0,7 (19 % visibile, n ≈ 1.556): ≈ 1,2 TF

   In F le due passate di contesto (~4 TF) diventano ×4: ricalcolo del checkpointing + backward ≈ 2× forward.
   Sono circa 12 TF in più per clip, cioè quanto un'altra passata intera e mezza. Il rapporto F/P di
   circa 3,7× è coerente. In `docs/worldsign-gpu.md` il passo completo era 0,55 s/clip (17,6 s a 32 clip):
   **qui 0,32 s/clip, quindi F non è anomalo, è P a essere leggero.**
2. **Il salto di F0 è più grande di quanto dicono i FLOP.** Mi aspettavo ~1-1,5 s in più, ne vedo 4,4.
   Candidati:
   - il predittore fisico (larghezza 384, profondità 4) fa attention su tutti gli 8.192 token, due volte
     (una per tipo di maschera): con head dim piccolo i kernel rendono poco;
   - la rimozione di cuDNN (`drop_cudnn_attention`) potrebbe lasciare il backend mem-efficient o math
     invece di flash, in alcuni punti;
   - gli hook su `context_levels`, la gather/scatter delle maschere e tanti kernel piccoli.
   Questo è il punto dove c'è margine da recuperare.
3. **Validazione frequente.** `validation_every: 25` in `phoenix14t_trial.yaml` aggiunge ~2,1 s/step
   ammortizzati in F (12,49 contro 10,39), e ~1,4 s in P. È configurazione, non lentezza del modello.
4. **GPU condivisa** con altri job (in passato ~25 GB occupati da altri sulla GPU 0). Durante questa run
   non l'ho visto, ma non l'ho monitorato in F. Da escludere.
5. **Poco probabili.** Il dataloader: in P gli stessi dati vanno a 11 clip/s, quindi non è lui. La
   memoria: 54 di 96 GiB, nessuna pressione sull'allocatore.

## Verifiche (da lanciare a mano)

Si lavora su una **copia** della run, che è già in stage F, per non toccare l'originale:

```bash
cp -r runs/phoenix14t-A tmp/profF
CFG="-c parameters/model/worldsign.yaml -c parameters/ablation/arm_A.yaml -c parameters/model/phoenix14t_trial.yaml -c parameters/model/local_single_gpu.yaml"
```

**V1. Uso della GPU e GPU condivisa (ipotesi 1 e 4).** Due terminali:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync signworld train run $CFG --output tmp/profF
nvidia-smi dmon -i 0 -s um -d 5           # sm% ~100 => compute-bound (strutturale); sm% basso => attese
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # deve esserci solo il nostro processo
```

**V2. Quali kernel pesano (ipotesi 2).** Dopo ~5 minuti di riscaldamento, 60 s di traccia:

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda,nvtx --delay 300 --duration 60 -o tmp/stageF \
  uv run --no-sync signworld train run $CFG --output tmp/profF
nsys stats --report cuda_gpu_kern_sum tmp/stageF.nsys-rep | head -30
```

Come leggerlo:
- in cima `flash_fwd`/`flash_bwd` e GEMM: tutto a posto, ipotesi 1;
- `efficient_attention`/`fmha_cutlass` o kernel `math` (softmax + bmm separati): backend sbagliato,
  ipotesi 2;
- tanti kernel piccoli (`index`, `gather`, `copy`) in alto: overhead delle maschere.

**V3. Separare encoder e predittore (ipotesi 2), senza toccare il codice.** Stesso V2 su una copia ferma
in F0: `cp -r` di un checkpoint di F0, se ce n'è uno in `checkpoints/`, altrimenti una run nuova fino allo
step ~450. La differenza F0 − P è il costo del predittore più le passate di contesto senza gradiente.

**V4. Peso della validazione (ipotesi 3).** Rilancio con validazione rara su un'altra copia e confronto
della colonna `wall`:

```bash
printf 'training:\n  validation_every: 1000\n' > tmp/rare_val.yaml
uv run --no-sync signworld train run $CFG -c tmp/rare_val.yaml --output tmp/profF_rareval
```

(La configurazione cambia, quindi serve una cartella nuova: riparte da zero e serve solo a misurare P.)

## Cosa si può guadagnare (se le verifiche confermano)

- Ipotesi 1: niente da correggere. Si accelera solo con più GPU o con meno token (risoluzione, maschere).
- Ipotesi 2: forzare flash nel predittore o raggruppare le due passate di maschera. Il salto di F0 suggerisce
  2-3 s/step.
- Ipotesi 3: alzare `validation_every` nelle trial: circa -2 s/step in F.

---

## Task 1.11 (24/11/2026): lavoro ridondante tolto dal passo, metodo invariato

GPU 1-3 in modalità "Prohibited": tutto misurato sulla GPU 0, **condivisa con il run `-r3` al 100 %**. Contano i rapporti (A/B alternati), non i tempi assoluti.

**Passata semantica (encoder intero senza grad, 4 clip, `scripts/perf/bench_encoder.py`):** eager 0,516 s → compilato (1.9) 0,431 → + LoRA fusa 0,368 → + RoPE fuso **0,339 (−21 %)**. Senza LoRA: 0,365, quindi ora la LoRA senza gradiente non costa nulla. `max_autotune` −3,5 %: scartato.

**Passo intero (`train profile`, 8 clip):**

| stage | 1.9 | 1.11 | Δ | picco |
|---|---|---|---|---|
| P | 1,143 | 0,945 | −17 % | 7,8 GiB |
| F0 | 2,690 | 2,207 | −18 % | 11,7 GiB |
| F | 3,765 | 3,276 | −13 % | 14,1 GiB |

Se i rapporti reggono a b40 (sono **da rimisurare a GPU libera**): P 2,58 → ~2,14 s, F0 6,26 → ~5,1, F 8,37 → ~7,3. Il piano a 15 epoche passa da ~730 h a ~635 h (−4 giorni su una GPU).

**Dove va ora un passo F** (traccia dei kernel, 4 clip, `scripts/perf/trace_step.py`): attention flash 39 % (forward 22 %, backward 17 %; testa da 32 nei predittori, poco efficiente), GEMM 29 %, elementwise (gelu, RoPE, LayerNorm) ~15 %. Sono kernel già al limite: dentro F non resta molto lavoro inutile da togliere.

### Leve rimaste

| leva | stima | cambia il metodo? |
|---|---|---|
| `latest` più frequente (oggi solo ogni 4.000 passi ≈ 2,9 h in P, 9 h in F); r3 ha perso 3.920 passi al resume | 0 per passo, fino a ore per interruzione | no |
| validazione più leggera: oggi ~540 s ogni 4.000 passi (0,27 s/clip, 4× il training con backward), cioè ~5 % in P e ~1,6 % in F. In P rifà il ramo fisico non addestrato e una seconda passata semantica sul rumore | −3-5 % in P | solo le letture diagnostiche |
| FP8 sulle GEMM del ViT congelato (passata semantica + base dei LoRA) | GEMM 2× → −10-15 % | sì: numerica delle feature |
| compile del predittore semantico (eager, ~7 % di P) | −2-3 % in P | no |
| LoRA con forward fuso anche in addestramento (autograd custom: un prodotto in forward e nel ricalcolo) | −3-5 % in F, stima | no, ma rischio con checkpoint + compile |
| leve 1-3 sopra (epoche di F, token della passata semantica, checkpointing selettivo) | le più grandi | sì |

### Multi-GPU (solo analisi, agente in sola lettura)

DDP non è il collo di bottiglia: in F ~22 M parametri addestrabili, cioè 88 MB di allreduce fp32, < 0,1 % del passo e sovrapposto al backward. Le collective di SIGReg e SyncBN sono piccole, ma fanno da barriera a metà forward: lì emergono gli sbilanciamenti fra i rank (maschere diverse per rank → numero di token visibili diverso). Efficienza attesa 90-95 %: le ottimizzazioni single-GPU si trasferiscono quasi 1:1.

Vincolo vero: 32 clip/GPU in F non entrano in A100 da 64 GB (`docs/worldsign-gpu.md`). Leve a basso sforzo: `static_graph=True` (0-2 %); `diagnostic_clips` diviso per il numero di GPU; un solo allreduce vettoriale in `_log_step`; ID dei video come tensore invece di `gather_objects` nel braccio C.
