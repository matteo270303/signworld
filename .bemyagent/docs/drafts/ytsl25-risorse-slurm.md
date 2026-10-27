# Risorse Slurm per una run ytsl25 (batch 128, Leonardo)

Data: 26/10/2026. Misure locali: GPU 0 (RTX PRO 6000 Blackwell, 95 GiB), batch 32, dati reali
(indice del 26/10), probe in `scratchpad/probe_stages.py`, risultati in `scratchpad/probe_stages_result.json`.

## Misure sulla GPU 0 (batch 32 per GPU, un processo)

| Stage | s/passo | VRAM picco allocata | VRAM riservata | RSS host picco |
|---|---|---|---|---|
| P | 2,70 | 26,65 GiB | 27,11 GiB | 16,25 GiB |
| F0 | 7,22 | 42,55 GiB | 43,46 GiB | 16,25 GiB |
| F | 9,80 | 48,27 GiB | 49,14 GiB | 16,25 GiB |

- `prepare()` (indice, embedding, statistiche): 34,8 s.
- Il picco RSS è quello di un processo dopo `prepare()` e durante i passi. I worker del data loader non sono misurati.
- La doc `docs/worldsign-gpu.md` riporta 17,9 s e 65,8 GiB per 32 clip in stage F, con batch sintetico. Con dati reali la misura è ~1,8× più veloce e ~17 GiB più leggera. Causa non chiarita.

## Ottimizzazioni testate

- **Checkpointing di attivazione spento:** OOM in stage F a batch 32 (93 GiB richiesti, 95 disponibili). Resta acceso.
- **Early stop:** già previsto (`patience: 3` epoche in F), ma le epoche sono fisse a 15 per confrontare i bracci allo stesso numero di epoche. Attivarlo cambia il protocollo.
- **Validazione ogni 4.000 passi:** ~28 validazioni nella run. Nello smoke la validazione di 2.000 clip ha richiesto ~7,5 min su questa GPU: costo minore, da misurare.
- **Frame 64 → 32 o meno maschere:** ridurrebbe il costo, ma cambia il modello. È un'ablation, non un'ottimizzazione.
- **torch.compile, altri kernel:** non testati.

## Conteggio dei passi (4 GPU × 32 clip, batch 128)

- 945.688 clip di train / 128 = 7.388 passi per epoca; 15 epoche = 110.820 passi.
- P: epoche 1–2 = 14.776 passi. F0: epoca 3 = 7.388. F: epoche 4–15 = 88.656.

## Stima su Leonardo (4× A100 64 GB, un nodo)

Le misure sono su Blackwell. Il fattore Blackwell → A100 è ancora una stima (1,5–2×), non misurata.

| Stage | s/passo su A100 | Passi | Durata |
|---|---|---|---|
| P | 4,0–5,4 | 14.776 | 0,7–0,9 giorni |
| F0 | 10,8–14,4 | 7.388 | 0,9–1,2 giorni |
| F | 14,7–19,6 | 88.656 | 15–20 giorni |
| **Totale** | | | **17–22 giorni (centrale ~20)** |

- Sincronizzazione DDP tra le 4 GPU non misurata: aggiunge tempo, non quantificato.
- ~20 giorni × 24 h / 24 h per job ≈ 20 job da 24 h concatenati con resume.
- Node-hour: ~480 per run (~1.900 GPU-hour).
- VRAM su A100 64 GB: 48,3 GiB allocati + contesto CUDA. Entra senza il backward per ramo. Da confermare: la misura è su dati reali, la doc diceva 65,8 GiB.

## RAM host

- Misurata: 16,25 GiB per processo (picco RSS, con indice ed embedding caricati).
- Non misurata: worker del data loader (8 per GPU sul cluster).
- Con 4 processi: ~65 GiB solo per i processi principali. `--mem=492000` è molto più di quanto serve, ma i worker non sono ancora misurati.

## Sulla GPU 0 (per riferimento)

Con batch 32 (configurazione locale, non quella dell'articolo): 29.552 passi per epoca, 15 epoche.
Tempo stimato: P 2 epoche × 29.552 × 2,70 s + F0 1 × 29.552 × 7,22 s + F 12 × 29.552 × 9,80 s ≈ **44 giorni**.
Il batch 128 non entra su questa GPU (servono ~4× la VRAM di batch 32 in stage F), quindi la run dell'articolo va sul cluster.

## Prima di lanciare su Leonardo

1. Confermare la misura F su A100 con un job breve: il fattore 1,5–2× è la stima più incerta.
2. Misurare la RAM con `sacct` (MaxRSS) durante quel job, worker inclusi.
3. Verificare la frequenza di salvataggio di `latest.pt`: non l'ho trovata nel codice con la grep.
4. Il calendario: invio CVPR 16/11/2026 (da README). Una run da ~20 giorni, più le altre ondate, non rientra prima di quella data.

## Limiti

- Dati reali, batch 32, un processo: non è una run distribuita. La sincronizzazione tra GPU non è misurata.
- `tmp/full_probe.py` e `tmp/sweep/` citati nella doc non sono nel repo: la discrepanza 17,9 s / 65,8 GiB non è verificabile da qui.
