# Decisioni e problemi

## Indice delle decisioni

> Le decisioni in linea stanno qui. Le scelte complesse, a livello di sistema, hanno un ADR dedicato in `decisions/`, collegato qui.

| # | Decisione | Sintesi | File |
|---|---|---|---|
| 1 | Addestramento per livello | ogni loss aggiorna solo i pesi del suo livello | `docs/worldsign-gerarchia.md` |
| 2 | Percorsi relativi al repository | tutti i percorsi in `parameters/` sono relativi alla radice | `parameters/PATHS.md` |

### Decisioni in linea

#### Overlay di configurazione

- **Problema**: molte run condividono una configurazione di base.
- **Decisione**: `parameters/model/worldsign.yaml` + braccio di ablation + overlay dei dati, fusi in ordine.
- **Compromesso**: la configurazione effettiva è visibile solo dopo la fusione.

## Lezioni di ingegneria

- **Percorsi sul cluster**: su ReCaS le sottocartelle di `data/` vanno collegate con symlink alle vecchie radici prima di sottomettere i job (`parameters/PATHS.md`).
- **Rifiuti in acquisizione**: i rifiuti dell'host non contano mai contro un elemento; codici di uscita 0 completato / 3 elementi rimanenti / 4 rifiutato.

## Problemi noti

I problemi di ricerca aperti dell'ultima run di addestramento (19/10/2026, passo 10.770) sono tracciati in `docs/problems.md` (13 punti: il bersaglio di posa non contiene la velocità, bersaglio anisotropo, predizione fisica debole, fuga di informazione dai box, deriva della LoRA dell'encoder, collasso delle query, R@1 basso, …).
