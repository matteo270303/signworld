# Panoramica

**signworld** è il codice di ricerca di **WorldSign**, un world model a energia per il retrieval multilingua della lingua dei segni (video ↔ didascalia). Copre l'intera pipeline — acquisizione dei dataset, manifest e controlli sui dati («collaudo»), esperimenti preliminari, addestramento e valutazione — ogni passo è un comando della CLI `signworld`. Destinatari: il gruppo di ricerca che lo esegue in locale e sui cluster ReCaS (HTCondor) / Slurm. Motivazioni complete del progetto: `docs/worldsign-progetto.md`.

## Glossario

Glossario completo: `docs/worldsign-progetto.md` §1.7. Termini specifici del progetto:

| Termine | Significato |
|---|---|
| WorldSign | il modello: encoder di posa + encoder video V-JEPA 2.1 + predictor fisico e semantico + ramo testuale |
| Livello 0 / posa | encoder di posa addestrato da zero; produce il bersaglio per passo `s_t ∈ ℝ¹⁹²` |
| Livello 1 / fisico | predizione mascherata dei bersagli di posa a partire dal video (energia `E_fis`) |
| Livello 2 / semantico | clip → `ŷ ∈ ℝ²⁵⁶` confrontato con l'embedding della didascalia `ẽ` (energia `E_sem`) |
| SIGReg | regolarizzatore gaussiano isotropo contro il collasso (`signworld/loss/sigreg.py`) |
| braccio / ablation | overlay di loss/config in `parameters/ablation/` |
| collaudo | controlli di accettazione su dati e modelli; report in `collaudo/<test>/` |
| PC1…PC7, A1…A4 | pre-controlli / controlli di accettazione numerati, richiamati dai comandi CLI |
| ledger | JSONL per shard con gli esiti dei download, rende l'acquisizione riprendibile |
| manifest | `data/<source>/manifest/clips.parquet`, uno per dataset |
| materialize | stadio 0: taglio delle clip, pose e indice di addestramento |
| source | adattatore di un dataset: `youtube_sl25`, `openasl`, `csl_news`, `bobsl`, `phoenix14t`, `csl_daily` |

## Metriche di successo

- Retrieval R@1 / R@5 / R@10 (video↔testo) sugli split di test dei benchmark — obiettivi ?
- Nessun collasso: effective rank / isotropia degli embedding (vedi `signworld/metrics/`)
- Throughput PC7: tempo per passo, MFU, memoria (`signworld train benchmark`)

## Struttura del repository

```
signworld/     pacchetto: cli, data, models, loss, metrics, experiment, logger
parameters/    impostazioni YAML: acquisition, analysis, model, ablation
scripts/       wrapper di esecuzione locale (materialize, train per dataset)
slurm/         job sbatch; slurm/condor/ per HTCondor (ReCaS)
tests/         suite pytest che rispecchia il pacchetto
docs/          documenti di progetto: architettura, loss, ablation, problemi
data/ checkpoints/ runs/ logs/ collaudo/   locali, non tracciati
```

## Avvio rapido

```bash
uv sync
uv run signworld --help
uv run pytest
uv run ruff check . && uv run mypy signworld
```

Eseguire dalla radice del repository — tutti i percorsi in `parameters/` sono relativi a essa (`parameters/PATHS.md`).

## Variabili d'ambiente

| Nome | Default | Descrizione |
|---|---|---|
| `SIGNWORLD_CONFIG` | `parameters/acquisition/default.yaml` | impostazioni di acquisizione |
| `SIGNWORLD_ANALYSIS_CONFIG` | `parameters/analysis/default.yaml` | impostazioni di analisi |
| `HF_TOKEN` | — (oppure `~/.secrets/hf_token`) | EmbeddingGemma (accesso riservato) |
| `BOBSL_USERNAME` / `BOBSL_PASSWORD` | — | credenziali di download BOBSL |
| `LOCAL_RANK` / `WORLD_SIZE` | impostate da torchrun | addestramento distribuito |

## Porte

Nessuna — CLI batch, nessun servizio.
