# WorldSign — Encoder di posa (livello 0)

L'encoder di posa produce il **bersaglio del livello fisico**: un vettore per passo che descrive lo stato del segnante. È addestrato **da zero, insieme al resto del modello**, con termini propri; il livello fisico lo legge con lo stop-gradient.

Documenti collegati:
- `worldsign-gerarchia.md`: i livelli, il flusso del gradiente, il calendario degli stadi, il monitoraggio;
- `worldsign-architettura.md` e `worldsign-loss.md`: il resto del modello e degli obiettivi;
- `worldsign-ablation.md`: ESP-6 (bersaglio video) ed ESP-8 (regolarizzatore).

Stato: 3/10/2026. Decisioni prese nella revisione della metodologia del 2–3/10 **[Nostra scelta]**.

---

## 1. Decisioni

| Punto | Decisione |
|---|---|
| Pre-training | nessuno: S-JEPA non esiste come modello pre-addestrato. L'encoder si addestra da zero nella run di WorldSign |
| Architettura | quella dell'encoder di posa di worldSign (`worldSign/worldsign/models/pose_encoder.py`), adattata al tubelet di V-JEPA e ai nostri articolatori (§3) |
| Segnale di addestramento | **senza maschera**: invarianza fra viste + SIGReg + ancora, nella forma di LeJEPA [Lett. 35] (§4) |
| Articolatori | **concatenati subito dopo gli encoder spaziali**, come in worldSign: un vettore per passo, a 512 nel transformer temporale |
| Uscita | `s ∈ (B, 32, C)`, **C = 256**, dalla proiezione finale 512 → 256 |
| Rapporto con il livello fisico | `E_fis` legge `LN(sg(s))`: il video non può spostare il bersaglio |
| Budget | il tetto dei parametri addestrabili sale da 22 a **30 M** (§3) |
| S-JEPA | replica a sé del paper (§7), pronta per un adattamento futuro; **non collegata** a WorldSign |
| ESP-8 (VICReg) | rimandata |

**Perché lo stop-gradient.** Un bersaglio addestrabile che riceve il gradiente dell'energia che lo deve predire ha una soluzione banale: la costante annulla `E_fis`. Con il gradiente fermato, il collasso del bersaglio dipende solo dai termini della posa, che lo impediscono per costruzione (§4.2).

---

## 2. Ingresso

L'encoder legge `pose_tokens` (B, 32, 69, 6), lo stesso tensore che il dataset già produce: per ogni passo (2 frame del tubelet) e giunto, x, y e presenza dei due frame, in unità di spalla.

**Ricanonicalizzazione per frame**, come worldSign: origine fra le spalle e unità pari alla loro distanza, frame per frame. Un frame senza entrambe le spalle resta nel riferimento della clip (origine 0, scala 1), che i dati hanno già.

**9 canali per giunto e per frame:**

| Canali | Contenuto |
|---|---|
| 0–1 | posizione globale (x, y) |
| 2–3 | posizione locale rispetto alla radice della parte, divisa per il mezzo ingombro della parte nel frame: la coordinata più grande fra le posizioni locali e gli ossi della parte, così entrambi restano entro ±1 |
| 4–5 | osso: vettore verso il giunto padre, con la stessa scala |
| 6–7 | velocità: spostamento globale dal frame precedente, in unità di spalla (0 al frame 0 e se uno dei due frame manca) |
| 8 | valido (presenza) |

Il decimo canale di worldSign, la confidenza di RTMW, non c'è: i dati materializzati hanno solo la presenza (soglia di punteggio 1,0). Un giunto assente ha 0 in tutti i canali tranne «valido».

**Perché anche gli ossi nel mezzo ingombro [correzione del 6/10].** Se manca la radice della parte (il polso, la punta del naso), le posizioni locali valgono 0 e da sole darebbero un ingombro al minimo (0,001). Gli ossi fra i giunti ancora visibili verrebbero divisi per quel valore: su OpenASL il 5,5 % delle clip superava 100, fino a 5.710, e la verifica P13 divergeva. worldSign non aveva il problema perché non azzerava le posizioni locali.

**Tubelet:** i 2 frame del passo si concatenano, quindi **(B, 32, 69, 18)**, allineato ai token di V-JEPA (tubelet di 2 frame).

**Parti, radici e padri** (indici COCO-WholeBody):

| Articolatore | Giunti | Radice | Padri |
|---|---|---|---|
| Corpo | 9: naso 0, orecchie 3–4, spalle 5–6, gomiti 7–8, polsi 9–10 | l'origine fra le spalle | orecchie e spalle → naso; gomiti → spalle; polsi → gomiti |
| Mano sinistra | 21: 91–111 | polso 91 | le cinque catene delle dita |
| Mano destra | 21: 112–132 | polso 112 | le cinque catene delle dita |
| Volto | 18: 9 punti della mascella (23, 25, …, 39), 8 del labbro interno (83–90), punta del naso 53 | punta del naso | la catena della mascella e quella del labbro, ciascuna appesa alla punta del naso |

---

## 3. Architettura

```
(B,32,69,18)
 ├─ corpo (9)  ─► Linear(18→128) + tipo giunto(9×128)  ─► transformer spaziale ─► media giunti ─► (B,32,128)
 ├─ mano sx(21)─► Linear(18→128) + tipo giunto(21×128) ─► transformer spaziale ─► media giunti ─► (B,32,128)
 ├─ mano dx(21)─► Linear(18→128) + tipo giunto(21×128) ─► transformer spaziale ─► media giunti ─► (B,32,128)
 └─ volto (18) ─► Linear(18→128) + tipo giunto(18×128) ─► transformer spaziale ─► media giunti ─► (B,32,128)
                                         (pesi NON condivisi fra le parti)
CONCATENAZIONE [corpo, sx, dx, volto] ─► (B,32,512)
 ─► + posizione temporale (32×512) ─► transformer temporale ─► (B,32,512)
 ─► H: LayerNorm(512) + Linear(512→256) ─► s  (B,32,256)
```

| Blocco | Configurazione | Parametri |
|---|---|---|
| 4 encoder spaziali | 2 strati pre-norm, d = 128, 8 teste, FFN 512, dropout 0,1 | 1,60 M |
| Posizione temporale | 32 × 512 | 0,02 M |
| Transformer temporale | 2 strati pre-norm, d = 512, 8 teste, FFN 2048, dropout 0,1 | 6,30 M |
| H | LayerNorm(512) + Linear(512 → 256) | 0,13 M |
| **Encoder** | | **≈ 8,06 M** |
| Decoder dell'ancora | Linear(256 → 2 · 69 = 138), uno solo | 0,04 M |

Rispetto a worldSign cambiano:
- il tubelet in ingresso, 2 frame per passo (32 passi invece di 64 frame);
- le parti, che sono i nostri 4 articolatori;
- il merge: worldSign porta subito la concatenazione a 256 e lavora nel temporale a 256; qui il temporale lavora a 512 e la riduzione a 256 avviene **dopo**, nella proiezione finale;
- l'uscita, 256 invece di 1.024: in worldSign la posa si mescolava ai token video nella TC-PR, qui è un bersaglio.

**Budget [Nostra scelta, 3/10].** Con l'encoder di posa (≈ 8,1 M con il decoder) gli addestrabili arrivano a **≈ 29,9 M**: il tetto passa da 22 a **30 M**. Il tetto era motivato dal rischio di overfitting sui video (§3.4 del progetto); l'encoder di posa si addestra sulla posa con invarianza, SIGReg e ancora, non sulle didascalie.

---

## 4. Addestramento

### 4.1 Viste

A ogni passo l'encoder fa **due forward**: sulla sequenza **pulita** `p` e su una **vista** `p̃ = A(p)`.

**Le augmentation toccano solo i disturbi:**
- rotazione nel piano attorno all'origine fra le spalle, angolo uniforme in **±10°** (rollio della camera);
- rumore gaussiano sulle coordinate dei giunti presenti, **σ = 0,01** unità di spalla (rumore del rilevatore).

**Non si usano:**
- traslazione e scala, che la ricanonicalizzazione per frame annulla;
- giunti persi simulati;
- il flip, che scambia la mano dominante.

Le augmentation geometriche della clip (ritaglio, stessa geometria su video e keypoint) restano quelle della pipeline e precedono le viste.

### 4.2 Termini

```
s  = G_ω(p)      s̃ = G_ω(p̃)                                      (B,32,256) ciascuno
c_{b,t} = media della presenza dei 69 giunti al passo t           passi validi: c_{b,t} > 0

L_inv       = media sui passi validi di  (1/256) ‖ s_{b,t} − s̃_{b,t} ‖²
L_anchor    = Σ_{b,t,j} c_{b,t,j} ‖ D(s_{b,t})_j − p̂_{b,t,j} ‖²  /  ( σ² Σ_{b,t,j} c_{b,t,j} )
SIGReg_posa = ½ [ SIGReg({s_{b,t}}) + SIGReg({s̃_{b,t}}) ]        sui passi validi di tutte le GPU

L_0 = (1 − λ)·( L_inv + L_anchor ) + λ·SIGReg_posa                λ = 0,05
```

- **`L_inv`** è il termine predittivo di LeJEPA con due viste: la sequenza pulita e la vista devono avere la stessa rappresentazione.
- **`L_anchor`** è l'ancora del progetto (`worldsign-loss.md` §4) con un decoder lineare unico 256 → 138:
  - p̂ è la posizione di ogni giunto mediata sui due frame del passo;
  - σ² è la varianza dei keypoint calcolata una volta sulle clip di training, così che predire la media valga 1.

  Tiene la cinematica nel bersaglio. È la «testa ausiliaria» di LeCun (§4.5.2).
- **`SIGReg_posa`** spinge s verso la gaussiana isotropa: impedisce il collasso, anche solo dimensionale, e rende tutte le direzioni ugualmente pesate nella L1 di `E_fis`.

**Il collasso è escluso per costruzione:**
- un s costante annulla `L_inv`, ma non `L_anchor` (non ricostruisce i keypoint) e non `SIGReg_posa` (non è gaussiano);
- non c'è EMA né stop-gradient interno alla posa: è la ricetta di LeJEPA.

**Cosa aspettarsi.** Con viste deboli, l'astrazione resta vicina alla cinematica, ma la rappresentazione è robusta al rumore del rilevatore e distribuita su tutte le dimensioni.

### 4.3 Ottimizzazione

| Voce | Valore |
|---|---|
| Gruppo | proprio (famiglia `pose`: encoder e decoder) |
| LR di picco | **3e-4** (il valore di worldSign) |
| Schedule | warm-up lineare sul **20 %** dei passi della run, poi **coseno fino a 0** alla fine pianificata; il cooldown della run lo moltiplica |
| Weight decay | quello della run (0,04), non su norme, bias e posizioni |
| Attiva | dal passo 0 (stadio P, `worldsign-gerarchia.md` §6) |

Il coseno fa rallentare il bersaglio proprio mentre il video lo insegue, come il momentum dell'EMA che sale verso 1 in V-JEPA e S-JEPA.

### 4.4 ESP-8, rimandata

Con VICReg al posto di SIGReg, `L_0` diventerebbe `25·L_inv + 25·V(s) + 1·Cov(s) + w_a·L_anchor`, con i pesi di VICReg. Restano aperti il peso dell'ancora `w_a` e l'espansore (`worldsign-ablation.md` §2.3).

---

## 5. Collegamento al livello fisico

```
pose_tokens ─► G_ω ─► s (B,32,256) ──sg──► LN ──────────────────────────────────────────┐
clip ─► maschere ─► V-JEPA 2.1-L + LoRA (token visibili) ─► fusione ─► predictor fisico  │
     ─► media dei token predetti in ciascuno dei 4 riquadri al passo t                   │
     ─► concatenazione (B,32,4·384) ─► Linear(1536→256) ─► ŝ ─► E_fis(ŝ, LN(sg(s))) ◄────┘
```

La lettura dal video rispecchia il lato posa: una media per articolatore, poi la concatenazione dei 4 articolatori. La definizione di `E_fis` per passo è in `worldsign-gerarchia.md` §4.

---

## 6. Monitoraggio

La posa entra nell'addestramento al passo 0. Lo stadio P esiste per verificarla prima che il video la insegua (`worldsign-gerarchia.md` §7). Cosa si legge:

| Lettura | Cadenza | Allarme |
|---|---|---|
| `inv_posa`, `anchor`, `sigreg_posa` | ogni passo (log) | picchi della loss |
| `s_rank`, `s_std`, `s_isoscore`, `s_sigreg` di s sui passi validi | letture frequenti | rango o deviazione sotto 0,5 × il passo 0; `s_sigreg` oltre 1,5 × il passo 0 |
| quote e coseni del gradiente di `inv_posa`, `anchor`, `sigreg_posa` sull'encoder di posa | letture frequenti | `pose_cos_anchor_sigreg_posa` < −0,3 per 5 letture: SIGReg combatte l'ancora |
| sul batch sonda fisso: IsoScore, rango, SIGReg di s; R² di posizione e velocità di ogni articolatore da s; CKA con il passo 0 | validazione | R² sceso oltre 0,02 sotto **il suo massimo** |
| learning rate del gruppo `pose` | ogni passo di log | — |
| assenza di gradiente di `E_fis` sulla posa | P17, prima di lanciare | la run non parte |

**Fermata F1, a fine stadio P (fine dell'epoca 1):**
- `anchor` e `sigreg_posa` in calo;
- rango di s > 0,5 × il passo 0;
- IsoScore di s ≥ 0,8;
- R² di posizione delle due mani da s ≥ 0,9 [Aperto: PC7].

---

## 7. S-JEPA: replica a sé del paper

Codice: `signworld/models/sjepa/`. Riproduce il metodo del paper [Lett. 99] sui suoi dati (scheletri NTU). Non è collegato a WorldSign: serve come base per un adattamento futuro. Il codice ufficiale non risulta pubblicato. Le parti che S-JEPA eredita da MAMP seguono il codice ufficiale di MAMP [Lett. 100] (https://github.com/maoyunyao/MAMP).

| Voce | Valore | Fonte |
|---|---|---|
| Input | (B, T = 120, V = 25, C = 3); taglio casuale nell'intervallo [0,5, 1] della sequenza valida, poi ricampionamento lineare a 120 (0,9 al test) | paper |
| Embedding | segmenti di l = 4 frame: Linear(l·C → 256) per giunto e segmento, più posizioni spaziali e temporali apprese separate; 30 × 25 = 750 token | paper |
| View encoder | 8 blocchi, d = 256, 8 teste, FFN 1024; vede solo i token visibili della vista | paper |
| Predictor | Linear(256 → 256) + 5 blocchi; mask token nelle posizioni nascoste, con posizioni proprie | paper; posizioni come il decoder di MAMP |
| Target encoder | EMA del view encoder, sulla sequenza originale intera; λ da 0,9999 a 1 con coseno | paper |
| Maschera | motion-aware: moto con passo m = l e padding replicato; intensità per token `I = Σ|M|`; `π = softmax(I/τ)`; estrazione Gumbel top-K; rapporto 0,9; τ = 0,80 (NTU-60), 0,75 (NTU-120) | paper; τ dal codice di MAMP |
| Viste | rotazione attorno all'asse verticale (Rodrigues, asse dal bacino alle spalle), traslazione, flip | paper; ampiezze non indicate, valori di partenza come in MAMP (θ = 0,3 rad) **[Aperto]** |
| Loss | cross-entropia fra softmax(R_p/0,1) e softmax((R_t − c)/0,06), centratura con β = 0,9 | paper |
| Ottimizzazione | AdamW, wd 0,05, beta (0,9, 0,95); LR da 0 a 1e-3 in 20 epoche, poi coseno fino a 5e-4; 1.200 epoche; batch 256 | paper |
| Uso a valle | solo il target encoder | paper |

---

## 8. Codice

| Modulo | Contenuto |
|---|---|
| `signworld/data/pose/skeleton.py` | parti, radici e padri dei 69 giunti |
| `signworld/models/worldsign/pose_features.py` | `JointFeatures`: ricanonicalizzazione, 9 canali, tubelet |
| `signworld/models/worldsign/pose_encoder.py` | `PartEncoder`, `PoseEncoder` |
| `signworld/models/worldsign/pose_branch.py` | `PoseViews`, `KeypointDecoder`, `PoseBranch` (encoder, decoder, viste, scala dell'ancora) |
| `signworld/loss/worldsign.py` | `invariance`, `Objective.pose_sigreg` |
| `signworld/models/sjepa/` | la replica di S-JEPA |

Configurazione: `pose_encoder` in `parameters/model/worldsign.yaml`.

---

## 9. Aperto

- Ampiezze delle viste (±10°, σ 0,01): valori di partenza, da rivedere dopo le prime run.
- Soglia di R² della fermata F1 (0,9) [PC7].
- ESP-8: peso dell'ancora ed espansore.
