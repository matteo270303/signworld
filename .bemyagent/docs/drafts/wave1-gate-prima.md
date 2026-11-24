# Wave 1: solo A-gate, o tutta l'ondata subito? (24/11/2026)

**Raccomandazione: prima solo A-gate, poi B0 B C V quando A-gate passa F2.**
Al più tardi, dopo F1 e qualche centinaio di passi senza crash, se il tempo stringe.

## Perché

- **La scadenza CVPR (16/11) è passata.** In `slurm/README.md` §3 l'unico argomento per lanciare
  `wave1` subito era guadagnare giorni su quella data.
- **I bracci senza gate non si fermano da soli.** `gate_stops` è solo in `gate.yaml`. Se A-gate scrive
  `FAILED`, B0 B C V vanno avanti finché non si lancia `scancel` a mano.
- **F1 e F2 non dipendono dal braccio.** Livello fisico, ancora e SIGReg della posa sono uguali in
  tutti i bracci (`docs/worldsign-ablation.md`). Un F1 o F2 fallito colpisce tutti e 5 i run.
  Il pilota ha fallito F1: rank di `s` 70 → 23, IsoScore 0,01 (`drafts/pilot-A-crash.md`).
- **Il codice di 1.9-1.14 non è mai girato sul cluster.** A-gate è già crashato tre volte (timeout
  NCCL, `KeyError` al passo ~330, SIGABRT del 18/11). Se lo stesso bug colpisce 5 run, si pagano
  5 code e 5 run da buttare.
- **Una correzione costa un riavvio.** Un resume con codice diverso viene rifiutato
  (provenance). Una correzione a monte di una fermata riparte comunque da zero
  (`worldsign-progetto.md` §4.13.5).

## Costo dell'attesa

F2 arriva a fine F0 (epoca 3). Stima su A100, non misurata: P 0,7-0,9 giorni + F0 0,9-1,2 giorni,
quindi **~2 giorni più la coda**, con `torch.compile` forse meno. Una run intera dura ~17-22 giorni
(~410-530 nodo-ore). Aspettare costa il ~10 % del tempo totale e protegge ~1.600-2.100 nodo-ore.

## Scartato

- **Tutta `wave1` subito**: aveva senso solo per CVPR. Rischia 5 run invece di 1 (README §3).

## Decisione dell'utente (24/11): tutto subito, scadenza IJCAI 2027

La data ufficiale non è ancora uscita. Stime: invio intorno al 4-15/01/2027. L'abstract di IJCAI 2026
scadeva il 12/01, il paper il 19/01. Restano ~7 settimane. Un run dura ~17-22 giorni (stima non
misurata), quindi c'è tempo per **una sola tornata**: ESP-1 e le ablation devono girare insieme.
Il criterio fail-fast del gate passa in secondo piano.

1. `slurm/batch/debug.sh`: 1 h, prima prova sul cluster di 1.9-1.14. Dal report si ricava la
   durata reale. Se supera ~25 giorni, si tagliano le epoche di F (`epochs: 15`, decisione
   dell'utente) prima di lanciare.
2. `slurm/batch/wave1.sh A B0 B C V` (5 nodi) + `slurm/batch/wave2.sh esp2 esp6 g1` (3 nodi, su A):
   8 nodi, il massimo. Catene intere (`SEGMENTS` 6): il run deve arrivare in fondo.
3. **A invece di A-gate**: le soglie F1-F3 non sono calibrate (il pilota è fallito su F1) e uno stop
   falso costerebbe giorni. I criteri si scrivono comunque nel log (`[stop] F1 ... passed|FAILED`,
   `trainer.py` `_stop_point`). Si leggono a mano e si fermano i run solo se il problema è reale.
4. Le ablation girano su A, non su θ\*: nel paper diventano "ablation intorno alla configurazione di
   riferimento". Fuori `d4` ed `esp4`.
5. (24/11) Confermato dall'utente: 15 epoche (già in `worldsign.yaml` e `youtube_sl25.yaml`), batch 128 e il resto dei parametri invariati.
