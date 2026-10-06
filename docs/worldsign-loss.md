# WorldSign — Energie, loss e regolarizzazioni

Formalizzazione completa di **cosa minimizza** WorldSign: le due energie, ogni termine della loss, come si combinano nei bracci di ESP-1 e nelle ablation, e ogni regolarizzazione. Tutto corrisponde al codice in `signworld/` (`loss/worldsign.py`, `models/worldsign/model.py`, `loss/sigreg.py`).

Documenti collegati:
- `worldsign-progetto.md`: motivazioni, con i riferimenti [Lett. N] della sua bibliografia;
- `worldsign-architettura.md`: i moduli e i layer che producono le quantità usate qui;
- `worldsign-posa.md` e `worldsign-gerarchia.md`: il livello della posa, gli stop-gradient fra i livelli e gli stadi;
- `worldsign-ablation.md`: le varianti.

Stato: 3/10/2026 (encoder di posa da zero, gerarchia per livello). **[Aperto]** indica un valore da calibrare in PC7.

---

## 1. Il quadro: un modello a energia regolarizzato

Un **modello a energia** (EBM) assegna a ogni coppia `(x, y)` un numero `E(x, y)`: basso se `y` è compatibile con `x`, alto altrimenti [Lett. 27, 28]. Non modella una probabilità normalizzata, quindi non serve una funzione di partizione. Si usa per **scegliere**: data `x`, la risposta è `argmin_y E(x, y)`.

WorldSign ha **due energie** su un encoder video condiviso:

| Energia | Coppia | Significato | Uso |
|---|---|---|---|
| `E_fis(v, p ; m)` | video `v`, posa `p`, maschera `m` | quanto il video, con una parte nascosta, prevede il movimento del corpo che ne segue | addestramento del livello fisico; plausibilità di una sequenza |
| `E_sem(v, c)` | video `v`, didascalia `c` | quanto il significato predetto dalla clip coincide con quello della frase | addestramento del livello semantico; retrieval, riconoscimento, localizzazione |

**Il problema di ogni EBM è il collasso.** Se si abbassa solo l'energia delle coppie osservate, la soluzione banale mette tutto a energia bassa: ogni rappresentazione costante. Due famiglie di rimedi [Lett. 27, 28]:
- **contrastiva:** si alza esplicitamente l'energia di coppie negative (InfoNCE). Serve un campionamento di negativi, e l'informazione cresce solo come `log N` [Lett. 38, 39];
- **regolarizzata:** si limita il **volume** di spazio che può avere energia bassa, senza negativi.

**WorldSign è regolarizzato** (tesi H1). Il volume a energia bassa è limitato da:
- **SIGReg**, che impone alle rappresentazioni la distribuzione `N(0, I)` (§8);
- il **basso rango** delle LoRA su encoder congelati (§10);
- l'**ancora** di ricostruzione, che impedisce al target di posa di perdere informazione (§4);
- lo **stop-gradient fra i livelli**: `E_fis` non può spostare il bersaglio di posa, `E_sem` non tocca l'encoder video (`worldsign-gerarchia.md` §3).

Il braccio C di ESP-1 (InfoNCE) è il confronto contrastivo, a parità di batch.

**Notazione.** Il documento di progetto usa `λ` sia per il peso dei token visibili sia per quello di SIGReg; qui si distinguono:

| Simbolo | Valore | Significato |
|---|---|---|
| `λ_ctx` | 0,5 | peso del termine sui token visibili in `E_fis` (V-JEPA 2.1) |
| `λ_S` | 0,05 | peso di SIGReg nell'obiettivo dei livelli sopra la posa (LeJEPA) |
| `λ_P` | 0,04 | peso di SIGReg nel livello di posa (`worldsign-posa.md` §4.2) |
| `𝓑` | 128 clip | il batch effettivo, su tutte le GPU (64 per GPU) |
| `m ∈ {brevi, lunghe}` | | le due maschere di ogni passo |
| `t = 1 … 32` | | passo (2 frame) |
| `a ∈ {corpo, sx, dx, volto}` | | articolatore: il suo riquadro nella lettura dal video |
| `C = 192`, `d = 512` | | larghezza del bersaglio di posa e dello spazio semantico |

---

## 2. Le quantità di partenza

| Quantità | Definizione | Dove |
|---|---|---|
| `s_t = G_ω(p)_t` | bersaglio di posa: l'encoder di posa sulla sequenza pulita, un vettore per passo | arch. §5 |
| `z_{v,t} = G_ω(v_v)_t`, `v = 0…3` | lo stesso sulle 4 viste: la pulita (`z_0 = s`) e tre estrazioni di disturbi (camera, rumore, giunti nascosti) | `worldsign-posa.md` §4.1 |
| `ŝ_t(v, m)` | lettura per passo: medie dei token predetti (visibili e nascosti) nei 4 riquadri, concatenate, `Linear(1.536 → 192)` | arch. §4.4 |
| `n^m_t(m)`, `w^v_t(m)` | token nascosti nei riquadri del passo e somma dei pesi dei visibili | arch. §4.4 |
| `c_{t,j}` | peso del giunto `j` al passo `t`: 1 se presente in un frame del tubelet, altrimenti 0 | arch. §2.2 |
| `c_t = (1/69) Σ_j c_{t,j}` | confidenza del passo | `step_confidence` |
| `p̂_{t,j}` | keypoint medio del passo, in unità di spalla | arch. §2.2 |
| `ŷ(v) ∈ ℝ^{K×512}` | predizione semantica (K = 1; K = 4 in ESP-4) | arch. §6 |
| `ẽ(c) ∈ ℝ^{512}` | target testuale: testa MLP su EmbeddingGemma centrato per lingua | arch. §7 |

---

## 3. Energia fisica `E_fis`

È la loss di V-JEPA 2.1 [Lett. 32], portata dal token al passo (`worldsign-gerarchia.md` §4). Seguiamo il **cooldown** di V-JEPA 2.1 (`cooldown-256px-64f.yaml`): clip da 64 frame, partendo da un modello già addestrato, come facciamo noi.

**Nessuna SIGReg a questo livello** (`worldsign-gerarchia.md` §4.1): agisce solo attraverso il bersaglio `s`. La MSE senza LayerNorm di LeWorldModel è un'ablation facoltativa (F8, `worldsign-ablation.md` §3.2).

**Target normalizzato e stoppato.** Come V-JEPA 2.1 fa con i token del teacher, il target si normalizza sui suoi canali, **senza parametri affini**, e arriva con il gradiente fermato:

```
s̄_t  =  LN( sg(s_t) )  =  ( s_t − mean_c(s_t) ) / sqrt( var_c(s_t) + 10⁻⁵ )
```

**Errore di un passo:** L1 media sui canali (`loss_exp = 1` di V-JEPA 2.1):

```
e_t(m)  =  (1/C) · ‖ ŝ_t(v, m) − s̄_t ‖₁
```

**Per una maschera `m`:**

```
ω_{b,t}(m)  =  c_{b,t} · ( n^m_{b,t}(m) + λ_ctx · w^v_{b,t}(m) )

                   Σ_(b,t)  ω_{b,t}(m) · e_{b,t}(m)
E_fis(v, p ; m)  =  ──────────────────────────────────          λ_ctx = 0,5, costante dal primo passo (cooldown)
                       Σ_(b,t)  ω_{b,t}(m)

w^v_{b,t}(m)  =  Σ_{i visibile nei riquadri del passo} ω_i     ω_i = 1 (cooldown; nostro default)
                                                                ω_i = 1/√d_min(i) (pre-training di V-JEPA 2.1, opzione)
```

`b` scorre sulle clip del batch; `d_min(i)` è la distanza euclidea, sulla griglia (passo, riga, colonna), dal token nascosto più vicino.

**Corrispondenza con V-JEPA 2.1 [Nostra scelta, adattamento di Lett. 32].** V-JEPA 2.1 calcola `L_pred` come media sui token nascosti e `L_ctx` come media sui visibili pesata per `ω_i`. Da noi un passo rappresenta i token dei suoi riquadri, quindi pesa quanti ne contiene: ogni token nascosto conta una volta, ogni visibile `λ_ctx · ω_i`. Due letture separate (solo nascosti, solo visibili) non funzionerebbero più: un riquadro senza token nascosti darebbe una parte vuota, mentre il bersaglio contiene comunque quell'articolatore, quindi resterebbe un errore irriducibile. L'unico peso nostro è la confidenza `c_t`: un passo senza giunti non entra.

**Nell'addestramento** le due maschere del passo pesano uguale:

```
E_fis(v, p)  =  ½ · [ E_fis(v, p ; m_brevi)  +  E_fis(v, p ; m_lunghe) ]
```

Ogni GPU calcola il rapporto sulle sue 64 clip e DDP media i gradienti delle GPU, come in V-JEPA. Il risultato coincide con il rapporto sull'intero batch quando i denominatori delle GPU sono uguali, e ci è vicino altrimenti.

**Plausibilità (solo in valutazione):**

```
Ē_fis(v, p)  =  (1/M) · Σ_{i=1..M}  E_fis(v, p ; m_i)          M maschere casuali
```

Una sola maschera potrebbe non coprire la parte anomala. Le maschere sono tubi su tutta la clip, quindi l'energia misura la coerenza **dentro la finestra**, con contesto prima e dopo, non la prevedibilità del futuro dal solo passato.

**Dove va il gradiente** (`worldsign-gerarchia.md` §3):
- **lato predizione:** testa di lettura, LoRA del predictor, fusione, LoRA e LayerNorm dell'encoder video (attraverso il contesto);
- **lato target: nessuno.** `s` arriva con lo stop-gradient: il video non può spostare il bersaglio. L'encoder di posa si addestra solo con i suoi termini (§4), che ne impediscono il collasso.

**Scala.** Il target ha media 0 e varianza 1 sui canali di ogni vettore, quindi `e` è di ordine 1: predire il vettore nullo costa la media di `|s̄|`, circa 0,8 per canali quasi gaussiani.

---

## 4. I termini della posa

L'encoder di posa si addestra da zero con tre termini, nella forma di LeJEPA [Lett. 35] (`worldsign-posa.md` §4).

### 4.1 Ancora di ricostruzione `L_anchor`

```
                 Σ_(b,t) Σ_j  c_{t,j} · ‖ D( s_t )_j − p̂_{t,j} ‖²
L_anchor  =  ─────────────────────────────────────────────────────
                       σ²  ·  Σ_(b,t) Σ_j  c_{t,j}

σ²  =  Σ_{t,j} c_{t,j} · ‖ p̂_{t,j} − μ_j ‖²  /  Σ_{t,j} c_{t,j}          μ_j = media pesata del giunto j
```

- `D` è un solo `Linear(192 → 138)` dal vettore del passo ai 69 giunti (arch. §5.1), sulla sola vista pulita.
- **`σ²` fissa la scala:** è la varianza dei keypoint attorno alla media di ciascun giunto, calcolata **una volta sulle clip di training** (20.000 clip, `statistics_clips`), un buffer salvato con il checkpoint. **Predire la media di ogni giunto vale esattamente 1.**
- **A cosa serve:** tiene la cinematica nel bersaglio, che senza di essa potrebbe ridursi a ciò che è invariante alle viste; dà un errore in keypoint interpretabile.

### 4.2 Invarianza `L_inv`

```
μ_{b,t} =  ¼ Σ_v z_{v,b,t}
L_inv   =  Σ_(b,t) [c_{b,t} > 0] · ¼ Σ_v (1/C) ‖ μ_{b,t} − z_{v,b,t} ‖²  /  Σ_(b,t) [c_{b,t} > 0]
```

È il termine predittivo di LeJEPA **nella sua forma**: ogni vista si avvicina al centro delle quattro, con la media sui canali. Con due viste varrebbe `‖s − s̃‖²/4C`. La pipeline delle viste è in `worldsign-posa.md` §4.1.

---

## 5. Energia semantica `E_sem`

```
E_sem(v, c)  =  1 − cos( ŷ(v), ẽ(c) )          ∈ [0, 2]
```

**Uso** [Lett. 34]:

```
testo → video      v*          =  argmin_v  E_sem(v, c)
video → testo      c*          =  argmin_c  E_sem(v, c)
riconoscimento     etichetta*  =  argmin_ℓ  E_sem(v, testo di ℓ)
profilo temporale  profilo(t)  =  E_sem( v[t, t+L], c )          dove la frase viene segnata
```

Con `K` ipotesi (ESP-4) la similarità di una clip è quella della sua ipotesi migliore: `max_k cos(ŷ_k, ẽ)`, cioè `min_k E_k`.

**Perché le coppie sbagliate restano ad energia alta senza negativi [Nostra argomentazione, su Lett. 35].** Se SIGReg rende `ŷ` ed `ẽ` circa `N(0, I_d)`, due vettori di coppie indipendenti hanno coseno `0 ± 1/√d`, cioè energia ≈ 1. L'allineamento porta le coppie corrispondenti a coseno ≈ 1, cioè energia ≈ 0. Il divario nasce dall'isotropia, non da un negativo. Il «circa» conta: didascalie simili non sono indipendenti.

**Il rischio: predire la media [Lett. 81].** Se la clip non determina del tutto la frase, una regressione è minimizzata dalla media delle frasi compatibili, che può non somigliare a nessuna. Per questo ESP-1 confronta L_unif e InfoNCE, ed esiste ESP-4.

### 5.1 La loss semantica di ogni braccio

| Braccio | `L_pred_sem` |
|---|---|
| A₀, A | `(1/|𝓑|) Σ_b E_sem(v_b, c_b)` |
| B₀, B | `(1/|𝓑|) Σ_b E_sem(v_b, c_b)  +  L_unif` |
| C | `L_InfoNCE` |
| ESP-4 (su θ\*) | energia libera rilassata (§5.2) al posto di `E_sem` |

### 5.2 Energia libera (solo ESP-4, facoltativa)

Le 8 query si dividono in `K = 4` gruppi da 2; ogni gruppo dà un'ipotesi `ŷ_k` [Lett. 81, 80].

```
E_k(v, c)     =  1 − cos( ŷ_k(v), ẽ(c) )
F_sem(v, c)   =  min_k  E_k(v, c)                                         inferenza

L_F           =  (1 − ε) · min_k E_k  +  ε · (1/K) Σ_k E_k               addestramento, ε = 0,05 [Aperto: PC7]
```

Con il solo minimo, le ipotesi che non vincono mai non ricevono gradiente e restano inutilizzate. Il termine `ε` ne dà a tutte una piccola parte.

---

## 6. Uniformity `L_unif` (bracci B₀ e B)

Wang e Isola [Lett. 40], sulla sfera unitaria:

```
L_unif(X)  =  log  ( 1 / (|X|·(|X|−1)/2) )  Σ_{i<j}  exp( − t · ‖ x̂_i − x̂_j ‖² )        x̂ = x / ‖x‖,   t = 2

L_unif     =  ½ · [ L_unif({ŷ}) + L_unif({ẽ}) ]
```

- Si calcola **sull'intero batch di 128** (tutte le GPU, raccolto con una collettiva dotata di backward): ogni coppia conta.
- Con `K` ipotesi, tutte le `ŷ_k` sono punti di `{ŷ}`.
- È **a coppie**: spinge i punti ad allontanarsi l'uno dall'altro. SIGReg è invece **distribuzionale**: confronta la distribuzione con una gaussiana. Il confronto A contro B₀ è la domanda centrale della tesi.

---

## 7. InfoNCE (braccio C)

```
ℓ_ij  =  cos( ŷ_i, ẽ_j ) / τ                      τ appresa: si ottimizza log τ, inizio τ = 0,07
ℓ_ij  =  −∞   se i ≠ j e le clip i e j vengono dallo stesso video          (P15)

L_InfoNCE  =  ½ · [  (1/|𝓑|) Σ_i  −log softmax_j(ℓ_ij)[i]   +   (1/|𝓑|) Σ_j  −log softmax_i(ℓ_ij)[j]  ]
```

- Simmetrico (testo → video e video → testo), **sulle 128 coppie dell'intero batch**, raccolte da tutte le GPU con la stessa collettiva.
- Due clip dello stesso video hanno spesso didascalie legate e lo stesso segnante: come negativi insegnerebbero a separare ciò che è simile per davvero, quindi si escludono.
- C ha anche `SIGReg_sem`, come A: isola il contributo dei negativi.

---

## 8. SIGReg

LeJEPA [Lett. 35]: una statistica che misura quanto un insieme di vettori si discosta da `N(0, I)`.

### 8.1 Definizione

```
direzioni:     v_1 … v_M  uniformi sulla sfera unitaria,  M = 1.024,  nuove a ogni passo
proiezioni:    u_{m,n}  =  ⟨ v_m , x_n ⟩,   n = 1 … N
f. caratt.:    φ̂_m(τ)   =  (1/N) Σ_n  e^{ i τ u_{m,n} }

Epps–Pulley:   EP_m  =  N · ∫_{−5}^{5}  | φ̂_m(τ) − e^{−τ²/2} |²  ·  e^{−τ²/2}  dτ

SIGReg(X)  =  (1/M) · Σ_m  EP_m
```

- `e^{−τ²/2}` è la funzione caratteristica di `N(0, 1)` ed è anche il **peso** dell'integrale.
- Il modulo quadro si calcola in reali: `|φ̂ − φ|² = (mean_n cos τu − e^{−τ²/2})² + (mean_n sin τu)²`.
- L'integrale usa la regola dei trapezi su **17 nodi** in `[−5, 5]`; il calcolo è sempre in float32.
- **Perché basta una statistica 1-D (Cramér–Wold):** una distribuzione è determinata dalle sue proiezioni monodimensionali. Se tutte sono `N(0, 1)`, la congiunta è `N(0, I)`.

### 8.2 Il fattore N

Per un campione davvero gaussiano il valore atteso è **costante**, qualunque sia `N`:

```
E[ SIGReg ]  =  ∫ (1 − e^{−τ²}) e^{−τ²/2} dτ  =  √(2π) − √(2π/3)  ≈  1,06
```

Per ogni altra distribuzione cresce **proporzionalmente a `N`**. È la forma di LeJEPA, a cui si riferiscono `λ_S` e `λ_P`.

**Vale solo per campioni indipendenti.** Con campioni correlati il valore minimo sale anche se la distribuzione è esattamente `N(0, I)`, e il gradiente spinge a separarli. Su dati sintetici: con correlazione 0,9 fra passi consecutivi di una clip, SIGReg sui 32 passi insieme vale 10; calcolato per passo resta 1,04. Per questo `SIGReg_posa` si calcola passo per passo (§8.4).

### 8.3 Su più GPU

Ogni GPU calcola le **somme** di `cos(τu)` e `sin(τu)` sui suoi campioni, e il loro numero. Somme e conteggi si sommano su tutte le GPU (`all_reduce`, con backward) prima di dividere, quindi `φ̂` è la funzione caratteristica **di tutto il batch** e `N` è il numero totale di campioni.

Le direzioni sono le stesse su tutte le GPU: vengono da un generatore con lo stesso seme, legato al passo. Un test a due processi verifica valore e gradienti uguali a quelli di un solo processo sull'intero batch.

### 8.4 Dove si applica

```
SIGReg_sem   =  ½ · [ SIGReg({ŷ}) + SIGReg({ẽ}) ]                         bracci A, B, C
SIGReg_posa  =  media su v = 0…3 e t = 1…32 di  SIGReg({ z_{v,b,t} : c_{b,t} > 0 }_b)   tutti i bracci
```

| Termine | Campioni `N` per valutazione | Direzioni | Nota |
|---|---|---|---|
| `SIGReg({ŷ})` | 128 (128·K in ESP-4) | condivise con `{ẽ}` | ogni modalità separatamente, come LeJEPA fa con le viste |
| `SIGReg({ẽ})` | 128 | condivise con `{ŷ}` | spinge la testa testuale, l'unica parte mobile del ramo |
| `SIGReg({z_{v,·,t}})`, per vista e passo | ≤ 128: le clip presenti al passo t | condivise fra viste e passi | come LeWorldModel; un passo conta se ha almeno due clip presenti su tutte le GPU; tutti i 128 gruppi in un solo calcolo e una sola `all_reduce` |

**Perché per modalità e per vista [Nostra argomentazione].** SIGReg garantisce qualcosa solo sull'insieme su cui è calcolato. Sull'unione di più gruppi, uno può perdere varianza lungo alcune direzioni ed essere compensato dagli altri senza che il test se ne accorga. Separatamente, ogni insieme ha la garanzia piena. Sulla posa il bersaglio è un vettore per passo che concatena i quattro articolatori (`worldsign-posa.md` §3): non ci sono più insiemi per articolatore.

**Perché per passo sulla posa [revisione del 6/10].** I 32 passi di una clip non sono campioni indipendenti (§8.2). LeJEPA applica SIGReg a ogni vista sui campioni del batch, e LeWorldModel a ogni istante separatamente.

**Cosa SIGReg non fa.** Non allinea: se `z ~ N(0, I)` allora anche `R·z` lo è, per ogni rotazione `R`. Quale clip corrisponda a quale frase lo decidono solo i termini predittivi.

**Il modality gap.** Se `ŷ` ed `ẽ` sono entrambe ≈ `N(0, I)` hanno la stessa distribuzione marginale: stessa media (0), nessun classificatore le distingue.

---

## 9. L'obiettivo completo

```
L  =  (1 − λ_P) · ( L_inv + L_anchor )  +  λ_P · SIGReg_posa                    λ_P = 0,04
   +  (1 − λ_S) · ( E_fis + L_pred_sem )   +  λ_S · SIGReg_sem                     λ_S = 0,05
```

Con gli stop-gradient fra i livelli ogni termine aggiorna solo il suo livello: la posa (`L_inv`, `L_anchor`, `SIGReg_posa`), il fisico (`E_fis`), il semantico (`L_pred_sem`, `SIGReg_sem`). Con Adam i pesi **fra** livelli non contano; contano solo i rapporti dentro ciascun livello (`worldsign-gerarchia.md` §3). Per questo ogni livello può avere il suo λ (`Objective.weights`).

| Braccio / ablation | termini della posa | `E_fis` | `L_pred_sem` | `SIGReg_sem` |
|---|---|---|---|---|
| **A** (gate) | sì | sì | `E_sem` | sì |
| **A₀** | sì | sì | `E_sem` | — |
| **B₀** | sì | sì | `E_sem + L_unif` | — |
| **B** | sì | sì | `E_sem + L_unif` | sì |
| **C** | sì | sì | `L_InfoNCE` | sì |
| **ESP-2** (su θ\*) | — | — | come θ\* | come θ\* |
| **ESP-4** (su θ\*, facoltativa) | sì | sì | `L_F` (§5.2) | `{ŷ_k}` tutte le ipotesi |

**Pesi presi dalla letteratura, nessuna calibrazione:**
- `λ_S = 0,05`: la forma di LeJEPA, *«default robusto»*, prestazioni *«stabili al variare di λ»* [Lett. 35]; nel paper vale per 8–10 viste, da rivedere con il livello semantico;
- `λ_P = 0,04`: lo 0,02 di LeJEPA per 4 viste con 256 campioni, raddoppiato per i ≤ 128 campioni di ogni passo; dentro la zona stabile di LeWorldModel, da 0,01 a 0,2 (`worldsign-posa.md` §4.2);
- pesi uguali fra i termini predittivi di un livello: sommare con pesi uguali *«eguaglia o supera gli ottimizzatori multi-task complessi»* [Lett. 96]. Le scale sono confrontabili per costruzione: `L_inv` di ordine 1 (s vicino a `N(0, I)`), `L_anchor` = 1 per la media.

**Per stadio del curriculum** (`worldsign-gerarchia.md` §6) entrano solo i termini dei livelli eseguiti, con la stessa formula:

| Stadio | Epoche | Livelli | Termini |
|---|---|---|---|
| P | 1–2 | posa, semantico | posa, `L_pred_sem`, `SIGReg_sem` |
| F₀ | 3 | + fisico (teste nuove) | tutti |
| F e cooldown | 4 → fine | tutti (con le LoRA) | tutti |

---

## 10. Regolarizzazioni

| Regolarizzazione | Dove | Valore | Cosa impedisce |
|---|---|---|---|
| **SIGReg** | `ŷ`, `ẽ`; le 4 viste della posa, passo per passo | `λ_S = 0,05`, `λ_P = 0,04`, 1.024 direzioni, 17 nodi | collasso; anisotropia; modality gap |
| **Stop-gradient fra i livelli** | `sg(s)` in `E_fis`; `sg(Enc(x))` nel passaggio semantico | — | che il video sposti il proprio bersaglio; che il semantico deformi l'encoder adattato dalla fisica |
| **Congelamento + LoRA** | encoder video, predictor fisico (r = 16) | `B = 0` all'inizio, scala `α/r = 1` **[Aperto: α]** | allontanarsi dai pesi pre-addestrati: i pesi congelati non memorizzano il corpus, gli aggiornamenti restano a basso rango |
| **Weight decay** (AdamW, disaccoppiato) | matrici addestrabili, LoRA comprese | 0,04 costante, come V-JEPA 2.1 | pesi grandi; 0 su norme, bias, scalari, posizioni e query apprese |
| **Dropout** | predictor semantico (attenzione, residui, MLP), testa testuale; **non** nell'encoder di posa, il cui `sg(s)` è un bersaglio | 0,1 **[Aperto: PC7]** | co-adattamento nei moduli da zero |
| **Stochastic depth** (DropPath) | rami residui del predictor semantico | 0,1 **[Aperto: PC7]** | dipendenza da singoli blocchi |
| **LayerScale** | rami residui del predictor semantico | init 1e-4 **[Aperto: PC7]** | aggiornamenti grandi all'inizio: ogni blocco parte vicino all'identità |
| **Ancora** | `s` → keypoint | peso uguale agli altri termini della posa | perdita di informazione cinematica nel target |
| **Invarianza fra viste** | le 4 viste della posa contro il loro centro | peso uguale | dipendenza dal rumore del rilevatore, dal punto di vista e dai giunti persi |
| **Schedule della posa** | encoder di posa | picco 3e-4, warm-up di 2 epoche (lo stadio P), coseno a 0 | un target che accelera o continua a muoversi mentre il video lo insegue |
| **Aumentazioni** | video e keypoint | jitter ±10 %, colore ±0,2 **[Aperto]**, niente flip | scorciatoie su inquadratura e colore |
| **Curriculum** | stadi P e F₀ | P 2 epoche (il warm-up della posa), F₀ un'epoca; warm-up di 2 epoche per ogni gruppo che entra | che la LoRA insegua un bersaglio casuale; che le teste casuali distorcano le feature pre-addestrate [Lett. 95] |
| **Early stopping** | metrica held-out channel | pazienza 3 epoche, contata nello stadio F, poi cooldown dal checkpoint migliore | overfitting sui canali |
| **Split per canale** | dati | 10 % dei canali per lingua dei segni **[Aperto]** | misurare la generalizzazione su clip dello stesso segnante |

**Cosa non c'è, per scelta:** nessuna EMA, nessuno stop-gradient dentro un livello, nessun clipping del gradiente (come V-JEPA 2.1), nessuna calibrazione dei pesi delle loss.

**Ottimizzazione** (`worldsign-gerarchia.md` §6.2).
- AdamW `β = (0,9; 0,999)`, bf16 con pesi in fp32, batch effettivo 128; un gruppo per famiglia del curriculum.
- Ogni famiglia tranne la posa: warm-up lineare di 2 epoche dalla sua entrata, poi costante; la posa: warm-up di 2 epoche (lo stadio P), poi coseno a 0. Su tutto, il cooldown di V-JEPA 2: lineare a 0 sul 5 % dei passi, dal checkpoint migliore.
- Learning rate base **[Aperto: PC7, {1e-4, 2e-4, 5e-4}]**; posa 3e-4.

---

## 11. Dove si calcola ogni termine con più GPU

| Termine | Calcolo | Aggregazione fra GPU |
|---|---|---|
| `E_fis`, `L_anchor`, `L_inv`, `E_sem`, `L_F` | media sulle clip della GPU | DDP media i gradienti: media globale (esatta a denominatori uguali) |
| `L_unif`, `L_InfoNCE` | tutte le coppie del batch intero | `all_gather` con backward; ogni GPU calcola lo stesso valore |
| `SIGReg_sem`, `SIGReg_posa` | funzione caratteristica del batch intero | `all_reduce` di somme e conteggi, con backward |

Per un termine calcolato identico su ogni GPU a partire da `y = Σ_r x_r`, il backward della collettiva dà a ogni GPU `W · ∂L/∂y`, e la media di DDP divide per `W`: il gradiente è esattamente quello del termine. Per questo **non c'è accumulo di gradienti**: SIGReg, InfoNCE e L_unif non si scompongono su micro-batch, e il batch di 128 deve essere uno.
