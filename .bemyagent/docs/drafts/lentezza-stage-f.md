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
