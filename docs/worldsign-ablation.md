# WorldSign — Ablation

Il catalogo delle ablation: **quelle concordate** per la sottomissione a CVPR 2027 e **tutte quelle possibili**, con la domanda a cui rispondono, come si lanciano e quanto costano.

Documenti collegati:
- `worldsign-progetto.md`: ipotesi H1–H7 (§5.1), previsioni (§5.3), piano (§4.14); i riferimenti [Lett. N] rimandano alla sua bibliografia;
- `worldsign-architettura.md` e `worldsign-loss.md`: i componenti e i termini che le ablation cambiano;
- `worldsign-posa.md` e `worldsign-gerarchia.md`: l'encoder di posa e la gerarchia per livello di θ\*.

Stato: 3/10/2026.

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
  -a 'configs=parameters/model/worldsign.yaml parameters/ablation/arm_X.yaml parameters/ablation/<ablation>.yaml' \
  slurm/condor/train.sub
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

In θ\* l'encoder di posa è quello di worldSign adattato, **addestrato da zero insieme al resto del modello** con invarianza, ancora e SIGReg (`worldsign-posa.md`), e la gerarchia è addestrata **per livello** (`worldsign-gerarchia.md`). S-JEPA è replicato a parte, non collegato [Decisioni del 2–3/10/2026].

| Run | Esperimento | Differenza rispetto a θ\* | Domanda | File | Costo | Stato |
|---|---|---|---|---|---|---|
| 6 | **ESP-2** | nessun livello fisico: niente encoder di posa, predictor fisico, ancora e SIGReg sulla posa; solo il passaggio semantico | il livello fisico migliora il semantico? (H3) | `esp2_no_physical.yaml` | ≈ 0,85 L | Concordata |
| 7 | **ESP-6** | **senza encoder di posa: bi-encoder video–video** invece di video–posa. Il target del livello fisico è il **target encoder di V-JEPA 2.1** sulla clip intera; niente encoder di posa, ancora e SIGReg sulla posa | la posa come target del livello fisico serve, rispetto al target video di V-JEPA 2.1? | `esp6_video_target.yaml` (da creare; serve codice) | ≈ 1 L, più il forward del target encoder [da misurare in PC7] | Concordata |
| 8 | **ESP-4** | `K = 4` ipotesi dalle 8 query, energia libera rilassata (ε = 0,05), a zero parametri aggiuntivi | l'ambiguità delle didascalie richiede una variabile latente? (H7) | `esp4_latent.yaml` | ≈ 1 L | Facoltativa |

**ESP-6: con e senza encoder di posa.** Sulla posa si confrontano solo due condizioni: **con** encoder di posa (θ\*, video–posa) e **senza** (ESP-6, video–video). Tutto il resto resta come in θ\*: encoder video con LoRA, fusione, predictor fisico, maschera, `E_fis` e livello semantico.

Da decidere prima del codice **[Aperto]**:
- **il target encoder:** quello rilasciato e congelato, oppure una copia EMA dell'encoder video con LoRA, come nel pre-training di V-JEPA 2.1;
- **quale rete fa da target:** i ViT-L distillati sono addestrati sull'ultimo layer di un teacher più grande (`worldsign-progetto.md` §4.4.1);
- **la lettura:** predizione per token, come V-JEPA 2.1, oppure la lettura per passo di θ\* (medie nei 4 riquadri, concatenate).

### 2.3 ESP-8 — Regolarizzazione su θ\*

Con l'encoder di posa addestrato da zero, il regolarizzatore è ciò che impedisce al target di posa di collassare: la sola energia predittiva si minimizza con un target costante. ESP-8 confronta due famiglie di regolarizzatori, a parità di tutto il resto.

| Run | Regolarizzatore | Forma | File | Costo | Stato |
|---|---|---|---|---|---|
| — | **SIGReg** (θ\*) | vincolo sull'intera distribuzione: gaussiana isotropa, test di Epps–Pulley su direzioni casuali [Lett. 35] | è θ\* | 0 | Concordata |
| 9 | **VICReg** | vincolo sui momenti del secondo ordine: **varianza** (deviazione standard di ogni dimensione sopra una soglia γ) e **covarianza** (penalità sui termini fuori diagonale) [VICReg, Bardes, Ponce e LeCun, ICLR 2022, https://arxiv.org/abs/2105.04906] | `esp8_vicreg.yaml` (da creare; serve codice) | ≈ 1 L | **Rimandata** (3/10) |

**Dove si applica.** VICReg prende il posto di SIGReg **sugli stessi tensori** di θ\*: le 4 viste del bersaglio di posa, passo per passo, al posto di `SIGReg_posa`, e `ŷ` ed `ẽ`, al posto di `SIGReg_sem`, se il braccio scelto come θ\* lo prevede. La variabile è una sola: la famiglia del regolarizzatore.

**Domanda.** Quale dei due tiene informativo il target di posa addestrato da zero e disperso lo spazio semantico?

Da decidere prima del codice **[Aperto]**:
- **l'invarianza:** VICReg ha tre termini. Sulla posa l'invarianza c'è già (`L_inv` fra le 4 viste, `worldsign-posa.md` §4.2); sul livello semantico il suo ruolo lo può svolgere `E_sem`. Proposta: varianza e covarianza in più dove oggi c'è SIGReg;
- **i pesi:** VICReg usa 25 / 25 / 1 per invarianza, varianza e covarianza; va fissato il peso dei suoi termini rispetto a quelli predittivi (per SIGReg è λ = 0,04, tarato in `worldsign-loss.md` §8.5);
- **l'espansore:** VICReg applica i termini all'uscita di un MLP espansore; qui si può usarlo oppure vincolare direttamente `s`, `ŷ` ed `ẽ`, come fa SIGReg.

**Totale concordato: ≈ 6,85 L** (≈ 7,85 L con ESP-4; +1 L con ESP-8, rimandata), più il forward del target encoder di ESP-6, in due ondate: i cinque bracci in parallelo, poi le ablation su θ\*.

---

## 3. Ablation possibili

Tutte **Proposte**. La priorità è una **[Nostra valutazione]**, basata su quanto la risposta conta per la tesi, sul rischio che la scelta attuale sia sbagliata e sul costo.

### 3.1 Mascheramento

#### M1 — Maschera guidata dagli articolatori, contro multi-blocco casuale · priorità **alta**

**Idea.** Oggi i blocchi sono in posizioni casuali: spesso nascondono sfondo e corpo fermo, e le mani restano visibili. Una maschera guidata dalla posa nasconde proprio le regioni degli articolatori, i riquadri di mani e volto già calcolati per la lettura. Il predictor deve allora predire dal contesto **dove sta l'informazione fonologica**.

**Precedenti.**
- In S-JEPA e MAMP la maschera è guidata dal moto dei giunti [Lett. 99, 100]; la nostra replica di S-JEPA la usa (`worldsign-posa.md` §7).
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
1. **Fuga di informazione:** la forma e la posizione della maschera dicono dove sono le mani, e `s` contiene anche la posizione rispetto alle spalle. Il predictor può leggere una parte del target dalla maschera stessa. Mitigazioni: allargare le regioni, mescolare con blocchi casuali (M1b), e un **controllo** con blocchi casuali della stessa area.
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

#### M4 — Maschera causale (predire il futuro dal passato) · **scartata il 6/10**

Nascondere gli ultimi passi, per un livello fisico che preveda in avanti come LeWorldModel. È stata valutata nel confronto con LeJEPA e LeWorldModel (`worldsign-gerarchia.md` §4.1) e scartata:
- in V-JEPA peggiora le rappresentazioni [Lett. 30];
- con l'encoder sul 75 % dei token visibili costerebbe più del doppio del passaggio fisico.

### 3.2 Loss fisica

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| F1 | pesi del **pre-training** di V-JEPA 2.1: visibili pesati `1/√d_min`, λ_ctx con warm-up 15k–30k su 252k | il cooldown è la scelta giusta per partire da un modello già addestrato? | config: `physical.weight_distance: true`, `lambda_progressive: true` | bassa |
| F2 | **niente `L_ctx`** (λ_ctx = 0): solo i token nascosti | il termine denso di V-JEPA 2.1 serve anche con un target per passo? | config: `physical.context_lambda: 0` | **media** |
| F3 | target **senza LayerNorm** | la normalizzazione di V-JEPA 2.1 serve con un target reso isotropo da SIGReg? | codice: un'opzione in `physical_energy` | bassa |
| F4 | soglia dei riquadri **1,0** invece di 0,3 | quali keypoint devono definire il riquadro di lettura? È una decisione aperta | config: `physical.box_threshold: 1.0` | **media** |
| F5 | lettura con **attention pooling** appreso invece della media nel riquadro | la media perde informazione (dita, orientamento)? | codice: una testa di lettura nuova | bassa |
| F6 | **niente ancora** (`L_anchor`) | l'ancora serve a tenere il target informativo? (H2) | codice: un'opzione in `WorldSign` | **media** |
| F8 | **MSE su `sg(s)` senza LayerNorm**, la loss di LeWorldModel, invece della L1 su `LN(sg(s))` di V-JEPA 2.1 **[facoltativa, 6/10]** | con un bersaglio reso N(0, I) da SIGReg la LayerNorm è quasi ridondante, e la MSE è invariante per rotazione mentre la L1 privilegia la base, arbitraria, di `s`; L1 porta alla mediana, L2 alla media (§5.5 del progetto) | codice: un'opzione in `physical_energy` | bassa |

F7 (`SIGReg_posa` sull'unione dei quattro articolatori) è superata: il bersaglio è un vettore per passo che li concatena (3/10).

### 3.3 Target di posa

Le varianti P1–P5 di prima (layer finale, schedule del layer finale, LoRA di posa, altro teacher pre-addestrato) riguardavano l'S-JEPA pre-addestrato e sono superate dall'encoder da zero (3/10). Restano:

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| P1 | target **EMA** dell'encoder di posa | una copia lenta del bersaglio aiuta il video a inseguirlo? | codice: una copia EMA | bassa |
| P2 | **S-JEPA** (la replica, adattata ai nostri dati) come livello 0 | un bersaglio appreso con la predizione latente è migliore di quello con invarianza e ancora? | codice: l'adattamento di `models/sjepa` | bassa |

### 3.4 Encoder video e predictor fisico

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| V1 | fusione **`linear`** o **`residual`** invece di `mlp` | serve la capacità dell'MLP, o basta partire esattamente dal predictor rilasciato? Decisione aperta di PC6 | config: `fusion.kind` | **media** |
| V2 | **un solo livello** (blocco 24) invece di 6/12/18/24 | la fusione multilivello di V-JEPA 2.1 porta qualcosa sulla posa? | codice: poco | bassa |
| V3 | LoRA a rango **8** o **32** | capacità contro overfitting sui canali | config: `encoder.lora.rank`; attenzione al tetto di 30 M | bassa |
| V4 | predictor fisico **da zero** invece del rilasciato | il riuso del predictor di V-JEPA 2.1 aiuta (§4.4.5 del progetto)? | codice: poco | media |
| V5 | LayerNorm dell'encoder **congelate** | le 0,10 M di norme addestrabili servono? | config: `encoder.train_norms: false` | bassa |

### 3.5 Livello semantico e testo

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| S1 | **scarto casuale del 50 % dei token** nel passaggio semantico | il ripiego di §4.9 del progetto (costo del passo ≈ 1,2 invece di 1,7, con il passaggio semantico ormai senza backward nell'encoder) costa accuratezza? Da fare se PC7 lo rende necessario | codice: poco | **media** |
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
| D2 | **senza lo stadio F₀**: teste fisiche e LoRA entrano insieme | le teste casuali distorcono le feature pre-addestrate [Lett. 95]? | codice: poco (ammettere 0 epoche in `training.stages`) | bassa |
| D3 | schedule **coseno** invece di warm-up, fase costante e cooldown | V-JEPA 2 contro la ricetta standard | codice: poco | bassa |
| D4 | **secondo seed** di θ\* | la variabilità fra seed, per leggere gli scarti di ESP-2 ed ESP-6 | nessun codice | **media** (Aperto nel progetto) |

### 3.7 Gerarchia

| ID | Variante rispetto a θ\* | Domanda | Come | Priorità |
|---|---|---|---|---|
| G1 | **globale**: `E_sem` raggiunge la LoRA video (lo schema prima del 3/10) | il livello semantico deve plasmare l'encoder? (`worldsign-gerarchia.md` §8) | config: `semantic.trains_encoder: true` | **media** |

---

## 4. Se il budget lo consente

Con le date di CVPR (registrazione il 10/11, invio il 16/11/2026) le ablation in più vanno scelte prima di lanciare θ\*. In ordine **[Nostra valutazione]**:

| Ordine | Ablation | Perché | Costo | Codice |
|---|---|---|---|---|
| 1 | **M1b** maschera miscela casuale + articolatori | la domanda sul mascheramento più legata al segnato; M1b limita la fuga di informazione di M1a | ≈ 1 L | medio |
| 2 | **F2** senza `L_ctx` | dice se il termine denso serve con un target per passo; solo configurazione | ≈ 1 L | nessuno |
| 3 | **S1** scarto del 50 % dei token | se PC7 mostra che serve, meglio misurarne il costo prima | ≈ 0,5 L | poco |

P1c, prima al primo posto, è superata con l'S-JEPA pre-addestrato (3/10).

Le altre ablation restano per un lavoro successivo, o per la versione estesa del paper.
