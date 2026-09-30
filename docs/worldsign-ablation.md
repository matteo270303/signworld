# WorldSign — Ablation

Il catalogo delle ablation: **quelle concordate** per la sottomissione a CVPR 2027 e **tutte quelle possibili**, con la domanda a cui rispondono, come si lanciano e quanto costano.

Documenti collegati:
- `worldsign-progetto.md`: ipotesi H1–H7 (§5.1), previsioni (§5.3), piano (§4.14); i riferimenti [Lett. N] rimandano alla sua bibliografia;
- `worldsign-architettura.md` e `worldsign-loss.md`: i componenti e i termini che le ablation cambiano.

Stato: 30/9/2026.

---

## 1. Regole comuni

Valgono per ogni run, concordata o no [Nostra scelta, §4.14 del progetto]:

- **ViT-L** sempre; **stesso batch effettivo, 128**; stesse epoche, quindi stessi dati visti.
- **Una variabile alla volta** rispetto alla configurazione di riferimento. Solo ESP-1 confronta più loss.
- **θ\* = il braccio migliore di ESP-1** secondo la metrica per decidere, la media di R@1 T2V e V2T sullo split held-out channel. Le ablation successive si applicano a θ\*.
- **Significatività:** intervalli bootstrap sulle query per ogni R@k. Con un solo seed per run vale la non sovrapposizione degli intervalli.
- **Costo** in unità **L = un addestramento completo su ViT-L, braccio A** (≈ 1 L per run, salvo indicazione). Il valore in GPU-ore si misura in PC7.
- **Come si lancia:** configurazione di base + il file del braccio + il file dell'ablation, in quest'ordine:

```
condor_submit -a 'run=<nome>' -a 'gpus=2' \
  -a 'configs=configs/model/worldsign.yaml configs/model/ablations/arm_X.yaml configs/model/ablations/<ablation>.yaml' \
  scripts/condor/train.sub
```

**Legenda dello stato:**

| Stato | Significato |
|---|---|
| **Concordata** | nel piano per CVPR |
| **Facoltativa** | nel piano, se il tempo lo consente |
| **Proposta** | possibile, non pianificata; va approvata |
| **Esclusa** | valutata e scartata dal progetto, riportata per completezza |

**Come si realizza:** *config* = basta un file YAML; *codice* = serve codice nuovo, indicato.

---

## 2. Ablation concordate

### 2.1 ESP-1 — Loss (5 run)

```
                                senza SIGReg_sem        con SIGReg_sem
solo allineamento                    A₀                       A
allineamento + L_unif                B₀                       B
InfoNCE, batch 128                                            C
```

| Run | Braccio | `L_pred_sem` | `SIGReg_sem` | File | Stato |
|---|---|---|---|---|---|
| 1 | **A** (run di gate) | `E_sem` | `½ [SIGReg({ŷ}) + SIGReg({ẽ})]` | `arm_A.yaml` | Concordata |
| 2 | **A₀** | `E_sem` | — | `arm_A0.yaml` | Concordata |
| 3 | **B₀** | `E_sem + L_unif` [Lett. 40] | — | `arm_B0.yaml` | Concordata |
| 4 | **B** | `E_sem + L_unif` | `½ [SIGReg({ŷ}) + SIGReg({ẽ})]` | `arm_B.yaml` | Concordata |
| 5 | **C** | InfoNCE, batch 128, clip dello stesso video escluse | `½ [SIGReg({ŷ}) + SIGReg({ẽ})]` | `arm_C.yaml` | Concordata |

Livello fisico, ancora e SIGReg sulla posa sono identici in tutti i bracci.

| Confronto | Domanda | Previsione (§5.3 del progetto) |
|---|---|---|
| A₀ → A | quanto porta SIGReg | A ≫ A₀ |
| A₀ → B₀ | quanto porta l'uniformity a coppie | B₀ > A₀ |
| **A contro B₀** | **vincolo distribuzionale contro uniformity a coppie**: la domanda centrale (H1) | A ≈ B₀ |
| B contro A e B₀ | i due meccanismi si sommano? | B ≈ A |
| A contro C | cosa aggiungono i negativi, a parità di batch e di dati visti | A ≈ C entro l'incertezza |

### 2.2 Ablation su θ\*

| Run | Esperimento | Differenza rispetto a θ\* | Domanda | File | Costo | Stato |
|---|---|---|---|---|---|---|
| 6 | **ESP-2** | nessun livello fisico: niente encoder di posa, predictor fisico, ancora e SIGReg sulla posa; solo il passaggio semantico | il livello fisico migliora il semantico? (H3) | `esp2_no_physical.yaml` | ≈ 0,85 L | Concordata |
| 7 | **ESP-3** | encoder di posa S-JEPA **congelato**: niente LoRA e niente layer finale; SIGReg sulla posa solo diagnostica; l'ancora addestra solo le sue teste | serve un target di posa addestrabile e reso isotropo? (H2) | `esp3_pose_frozen.yaml` | ≈ 1 L | Concordata |
| 8 | **ESP-4** | `K = 4` ipotesi dalle 8 query, energia libera rilassata (ε = 0,05), a zero parametri aggiuntivi | l'ambiguità delle didascalie richiede una variabile latente? (H7) | `esp4_latent.yaml` | ≈ 1 L | Facoltativa |

**Totale concordato: ≈ 6,85 L** (≈ 7,85 L con ESP-4), in due ondate: i cinque bracci in parallelo, poi le ablation su θ\*.

---

## 3. Ablation possibili

Tutte **Proposte**. La priorità è una **[Nostra valutazione]**, basata su quanto la risposta conta per la tesi, sul rischio che la scelta attuale sia sbagliata e sul costo.

### 3.1 Mascheramento

#### M1 — Maschera guidata dagli articolatori, contro multi-blocco casuale · priorità **alta**

**Idea.** Oggi i blocchi sono in posizioni casuali: spesso nascondono sfondo e corpo fermo, e le mani restano visibili. Una maschera guidata dalla posa nasconde proprio le regioni degli articolatori, i riquadri di mani e volto già calcolati per la lettura. Il predictor deve allora predire dal contesto **dove sta l'informazione fonologica**.

**Precedenti.**
- In S-JEPA e MAMP la maschera è guidata dal moto dei giunti [Lett. 99, 100]; il nostro S-JEPA è pre-addestrato così.
- Nel video, maschere guidate dal moto o apprese: MGMAE (ICCV 2023), AdaMAE (CVPR 2023).
- Nel segnato, SignBERT+ maschera le mani sulla posa.
- V-JEPA mostra che il disegno della maschera pesa molto: tubi casuali al 90 % 51,5 su K400 contro 72,9 del multi-blocco [Lett. 30].

**Varianti** (ognuna rispetta il vincolo: **maschera identica in ogni frame della clip**, cioè un tubo):

| Variante | Maschera |
|---|---|
| M1a · articolatori | unione nel tempo dei riquadri di **entrambe le mani** (o mano dominante + volto), nascosta come tubo; blocchi casuali in più fino al rapporto di V-JEPA |
| M1b · miscela | metà dei passi con la maschera multi-blocco di oggi, metà con M1a: conserva la distribuzione di V-JEPA la metà delle volte |
| M1c · probabilistica | i blocchi multi-blocco si posizionano con probabilità proporzionale all'occupazione (o al moto) degli articolatori nella patch, come la maschera di MAMP sui giunti |

**Vincoli tecnici (codice nuovo).**
- La posizione delle mani cambia da clip a clip, quindi la maschera **non si può più condividere nel batch**.
- Per non tornare alla troncatura di V-JEPA, che rompe i tubi (`worldsign-architettura.md` §4.1), ogni clip deve nascondere **lo stesso numero di patch**: per esempio le `k` patch con occupazione più alta nell'unione dei riquadri, completate a caso fino a `k`.
- Serve un nuovo `MaskPolicy` che riceva i riquadri del batch.

**Rischi.**
1. **Fuga di informazione:** la forma e la posizione della maschera dicono dove sono le mani, e `s_{t,a}` contiene anche la posizione rispetto alle spalle. Il predictor può leggere una parte del target dalla maschera stessa. Mitigazioni: allargare le regioni, mescolare con blocchi casuali (M1b), e un **controllo** con blocchi casuali della stessa area.
2. **`L_ctx` sugli articolatori si svuota:** nei riquadri delle mani restano pochi token visibili.
3. **La plausibilità** `Ē_fis` va misurata con la stessa distribuzione di maschere dell'addestramento.

**Lettura.** Se `E_fis`, errore in keypoint e probe fonologici migliorano senza perdere R@1 held-out channel, la maschera si adotta. Se migliora solo `E_fis`, è probabile la fuga (1). **Costo** ≈ 1 L per variante.

#### M2 — Maschera per clip, contro condivisa nel batch · priorità bassa

Oggi una maschera per tipo è condivisa dalle 64 clip di una GPU. Varianti per clip che conservano i tubi:
- **equalizzata:** si nascondono tubi in più fino al minimo del batch. Rapporti 62 % → 79 % (brevi), 81 % → 95 % (lunghe);
- **tubi scartati:** come V-JEPA, ma togliendo tubi interi. Si perde il 43 % / 74 % dei token visibili.

**Domanda:** la varietà di maschere dentro un batch conta, a parità di rapporto? **Codice:** poco (`masking.py`).

#### M3 — Tipi e rapporti di maschera · priorità bassa

Solo brevi, solo lunghe, entrambe (oggi); scala dei blocchi. V-JEPA l'ha già studiato sul video naturale [Lett. 30]; da noi cambia il target (la posa), non il video. **Config:** `masking.specs`.

#### M4 — Maschera causale (predire il futuro dal passato) · priorità bassa

Nascondere gli ultimi passi. In V-JEPA peggiora le rappresentazioni [Lett. 30], ma darebbe un'energia di **prevedibilità** del futuro, utile per la produzione (H6). **Codice:** poco (blocchi con estensione temporale parziale, ancorati alla fine).

### 3.2 Loss fisica

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| F1 | pesi del **pre-training** di V-JEPA 2.1: visibili pesati `1/√d_min`, λ_ctx con warm-up 15k–30k su 252k | il cooldown è la scelta giusta per partire da un modello già addestrato? | config: `physical.weight_distance: true`, `lambda_progressive: true` | bassa |
| F2 | **niente `L_ctx`** (λ_ctx = 0): solo i token nascosti | il termine denso di V-JEPA 2.1 serve anche con un target per articolatore? | config: `physical.context_lambda: 0` | **media** |
| F3 | target **senza LayerNorm** | la normalizzazione di V-JEPA 2.1 serve con un target reso isotropo da SIGReg? | codice: un'opzione in `physical_energy` | bassa |
| F4 | soglia dei riquadri **1,0** invece di 0,3 | quali keypoint devono definire il riquadro di lettura? È una decisione aperta | config: `physical.box_threshold: 1.0` | **media** |
| F5 | lettura con **attention pooling** appreso invece della media nel riquadro | la media perde informazione (dita, orientamento)? | codice: una testa di lettura nuova | bassa |
| F6 | **niente ancora** (`L_anchor`) | l'ancora serve a tenere il target informativo? (H2) | codice: un'opzione in `WorldSign` | **media** |
| F7 | `SIGReg_posa` **sull'unione** dei quattro articolatori | il vincolo per articolatore conta davvero (§8.4 della loss)? | codice: poco | bassa |
| F8 | errore **L2** invece di L1 | L1 porta alla mediana, L2 alla media (§5.5 del progetto) | codice: poco | bassa |

### 3.3 Target di posa

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| P1 | **layer finale**: (a) nessuno, (b) uno condiviso dai 4 articolatori, (c) inizializzato dallo **sbiancamento per articolatore** già misurato | il target diventa isotropo, e a che prezzo? | config: `pose_encoder.final_layer: false` (a); codice per (b) e (c) | **alta** |
| P2 | layer finale sullo **schedule principale** invece di warm-up + coseno a 0 al 50 % | fermare il target a metà run aiuta o toglie? | codice: poco (lo schedule del gruppo) | media |
| P3 | LoRA di posa a **×0,1** o **×1** invece di ×0,05 | quanto può muoversi il target? VL-JEPA trova l'ottimo fra ×0,05 e ×0,10 [Lett. 34] | config: `pose_encoder.learning_rate_multiplier` | bassa |
| P4 | target **EMA** dell'encoder di posa | la soluzione classica contro un target mobile serve anche qui? (H2) | codice: una copia EMA | bassa |
| P5 | **altro encoder di posa** (MAMP, Uni-Sign) come target | la scelta di PC5, fatta con letture lineari, regge nell'addestramento? | codice: caricare un altro teacher | bassa |

**Perché P1 ha priorità alta [Nostra misura].**
- Lo spettro di S-JEPA è molto schiacciato: il 90 % della varianza sta in 17 direzioni su 256, il 99 % in circa 60 (spettro del 22/9).
- Lo sbiancamento porta l'IsoScore a 0,82–0,85 senza perdere R². Quello salvato ha pesi fino a circa 15 per le mani (192 direzioni) e fino a 2,4·10⁶ per corpo e volto (tutte le 256: le direzioni quasi vuote vanno amplificate moltissimo).
- Partendo dall'identità, con Adam che sposta ogni peso di al più circa `lr` per passo, il layer finale non può arrivarci in una run.
- La variante (c) parte già isotropa. Da decidere prima di θ\* se il criterio F1 (IsoScore ≥ 0,8 alla fine dello stadio 1) resta.

### 3.4 Encoder video e predictor fisico

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| V1 | fusione **`linear`** o **`residual`** invece di `mlp` | serve la capacità dell'MLP, o basta partire esattamente dal predictor rilasciato? Decisione aperta di PC6 | config: `fusion.kind` | **media** |
| V2 | **un solo livello** (blocco 24) invece di 6/12/18/24 | la fusione multilivello di V-JEPA 2.1 porta qualcosa sulla posa? | codice: poco | bassa |
| V3 | LoRA a rango **8** o **32** | capacità contro overfitting sui canali | config: `encoder.lora.rank`; attenzione al tetto di 22 M | bassa |
| V4 | predictor fisico **da zero** invece del rilasciato | il riuso del predictor di V-JEPA 2.1 aiuta (§4.4.5 del progetto)? | codice: poco | media |
| V5 | LayerNorm dell'encoder **congelate** | le 0,10 M di norme addestrabili servono? | config: `encoder.train_norms: false` | bassa |

### 3.5 Livello semantico e testo

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| S1 | **scarto casuale del 50 % dei token** nel passaggio semantico | il ripiego di §4.9 del progetto (costo del passo ≈ 1,8 invece di 3,7) costa accuratezza? Da fare se PC7 lo rende necessario | codice: poco | **media** |
| S2 | predictor semantico a **2 o 6 blocchi**; **1 o 16 query** | la capacità del predictor semantico | config: `semantic.depth`, `semantic.queries` | bassa |
| S3 | **niente centratura per lingua** | la centratura toglie la lingua dal target (§4.4.4 del progetto)? | codice: un'opzione nel ramo testuale | bassa |
| S4 | **altro encoder testuale** (Qwen3-Embedding, SONAR, BGE-M3) | VL-JEPA misura +5,4 in retrieval con Qwen3-8B [Lett. 34] | nuovo precalcolo degli embedding | bassa |
| S5 | `λ_S` = 0,02 o 0,1; **256** direzioni | sensibilità di SIGReg; LeJEPA la dichiara bassa [Lett. 35] | config: `losses.sigreg_weight`, `losses.sigreg_directions` | bassa |
| S6 | `D` = MSE fra i vettori invece di `1 − cos` | energia sensibile alla norma o solo alla direzione | codice: poco | bassa |
| S7 | testa testuale **lineare**, **assente** o **whitening fisso** | — | — | **Esclusa** (§4.4.4 del progetto: decisione dalla letteratura) |

### 3.6 Dati e addestramento

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| D1 | **64 frame uniformi** invece della selezione guidata dal moto | la selezione aiuta nell'addestramento, oltre che nelle letture del collaudo? | una materializzazione diversa dei frame | bassa |
| D2 | **senza gli stadi 1a e 2a** | le teste casuali distorcono le feature pre-addestrate [Lett. 95]? | config: `training.stages` a 0 | bassa |
| D3 | schedule **coseno** invece di warm-up, fase costante e cooldown | V-JEPA 2 contro la ricetta standard | codice: poco | bassa |
| D4 | **secondo seed** di θ\* | la variabilità fra seed, per leggere gli scarti di ESP-2 ed ESP-3 | nessun codice | **media** (Aperto nel progetto) |

---

## 4. Se il budget lo consente

Con le date di CVPR (registrazione il 10/11, invio il 16/11/2026) le ablation in più vanno scelte prima di lanciare θ\*. In ordine **[Nostra valutazione]**:

| Ordine | Ablation | Perché | Costo | Codice |
|---|---|---|---|---|
| 1 | **P1c** layer finale inizializzato dallo sbiancamento | senza, l'isotropia del target, cioè H2 e il criterio F1, è difficile da raggiungere (§3.3) | ≈ 1 L | poco |
| 2 | **M1b** maschera miscela casuale + articolatori | la domanda sul mascheramento più legata al segnato; M1b limita la fuga di informazione di M1a | ≈ 1 L | medio |
| 3 | **F2** senza `L_ctx` | dice se il termine denso serve con un target per articolatore; solo configurazione | ≈ 1 L | nessuno |
| 4 | **S1** scarto del 50 % dei token | se PC7 mostra che serve, meglio misurarne il costo prima | ≈ 0,5 L | poco |

Le altre ablation restano per un lavoro successivo, o per la versione estesa del paper.
