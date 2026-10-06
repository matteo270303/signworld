# WorldSign — Encoder di posa (livello 0)

L'encoder di posa produce il **bersaglio del livello fisico**: un vettore per passo che descrive lo stato del segnante. È addestrato **da zero, insieme al resto del modello**, con termini propri; il livello fisico lo legge con lo stop-gradient.

Documenti collegati:
- `worldsign-gerarchia.md`: i livelli, il flusso del gradiente, il calendario degli stadi, il monitoraggio;
- `worldsign-architettura.md` e `worldsign-loss.md`: il resto del modello e degli obiettivi;
- `worldsign-ablation.md`: ESP-6 (bersaglio video) ed ESP-8 (regolarizzatore).

Stato: 6/10/2026. Decisioni prese nella revisione della metodologia del 2–3/10 **[Nostra scelta]**; viste, SIGReg per passo, λ, dimensione di `s` e dropout rivisti il 6/10 sul confronto con LeJEPA e LeWorldModel [Lett. 35] (§4.4).

---

## 1. Decisioni

| Punto | Decisione |
|---|---|
| Pre-training | nessuno: S-JEPA non esiste come modello pre-addestrato. L'encoder si addestra da zero nella run di WorldSign |
| Architettura | quella dell'encoder di posa di worldSign (`worldSign/worldsign/models/pose_encoder.py`), adattata al tubelet di V-JEPA e ai nostri articolatori (§3) |
| Segnale di addestramento | **senza predizione mascherata**: invarianza fra **4 viste** nella forma di LeJEPA (distanza dal centro), **SIGReg per passo** come LeWorldModel, ancora (§4) |
| Articolatori | **concatenati subito dopo gli encoder spaziali**, come in worldSign: un vettore per passo, a 512 nel transformer temporale |
| Uscita | `s ∈ (B, 32, C)`, **C = 192**, dalla proiezione finale 512 → 192 (§3) |
| Dropout | **nessuno**: il bersaglio `sg(s)` non deve cambiare con un'estrazione del dropout |
| λ della posa | **0,04**, proprio del livello (§4.2) |
| Rapporto con il livello fisico | `E_fis` legge `LN(sg(s))`: il video non può spostare il bersaglio |
| Budget | il tetto dei parametri addestrabili sale da 22 a **30 M** (§3) |
| S-JEPA | replica a sé del paper (§7), pronta per un adattamento futuro; **non collegata** a WorldSign |
| ESP-8 (VICReg) | rimandata |

**Perché lo stop-gradient.** Non serve contro il collasso: SIGReg sul bersaglio basta a escludere la costante, ed è la tesi di LeWorldModel, che addestra encoder e predictor insieme senza stop-gradient. Serve a tenere il bersaglio **cinematico**: il video non deve poterlo piegare verso ciò che gli è facile predire. La conseguenza da sapere: `s` non viene reso predicibile dal video, quindi `E_fis` può restare alto sulle parti di `s` che il video non vede.

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
 ─► H: LayerNorm(512) + Linear(512→192) ─► s  (B,32,192)
```

| Blocco | Configurazione | Parametri |
|---|---|---|
| 4 encoder spaziali | 2 strati pre-norm, d = 128, 8 teste, FFN 512, senza dropout | 1,60 M |
| Posizione temporale | 32 × 512 | 0,02 M |
| Transformer temporale | 2 strati pre-norm, d = 512, 8 teste, FFN 2048, senza dropout | 6,30 M |
| H | LayerNorm(512) + Linear(512 → 192) | 0,10 M |
| **Encoder** | | **≈ 8,03 M** |
| Decoder dell'ancora | Linear(192 → 2 · 69 = 138), uno solo | 0,03 M |

Rispetto a worldSign cambiano:
- il tubelet in ingresso, 2 frame per passo (32 passi invece di 64 frame);
- le parti, che sono i nostri 4 articolatori;
- il merge: worldSign porta subito la concatenazione a 256 e lavora nel temporale a 256; qui il temporale lavora a 512 e la riduzione a 192 avviene **dopo**, nella proiezione finale;
- l'uscita, 192 invece di 1.024: in worldSign la posa si mescolava ai token video nella TC-PR, qui è un bersaglio;
- il dropout, che qui non c'è: worldSign lo usava a 0,1.

**Perché C = 192 [revisione del 6/10].**
- **LeWorldModel** è l'analogo più vicino: uno stato per passo, SIGReg per passo con N = 128, usato direttamente come bersaglio. Nella sua ablation su Push-T il successo vale 42 % a 8 dimensioni, 74 % a 24, 92 % a 96, **94 % a 192** e 92 % a 384.
- **LeJEPA** (Tab. 1d) preferisce proiezioni piccole sotto SIGReg: 64 è la migliore, 1.024 la peggiore. Lì però la proiezione non si usa a valle, mentre `s` sì.
- **OpenASL** (250 clip): i keypoint di un passo (i 138 valori dell'ancora) hanno il 99 % della varianza in 23 componenti lineari e il 99,9 % in 67. Con lo spostamento dentro il passo (276 valori) servono 47 e 125 componenti.
- Con 192 la parte lineare al 99,9 % entra con margine. Con 256 SIGReg dovrebbe riempire 131 direzioni oltre quella con contenuto non cinematico, cioè amplificare il rumore; con 192 le direzioni in più scendono a 67.

**Budget [Nostra scelta, 3/10; aggiornato il 6/10].** Con l'encoder di posa (≈ 8,06 M con il decoder) gli addestrabili arrivano a **≈ 29,8 M**: il tetto passa da 22 a **30 M**. Il tetto era motivato dal rischio di overfitting sui video (§3.4 del progetto); l'encoder di posa si addestra sulla posa con invarianza, SIGReg e ancora, non sulle didascalie.

---

## 4. Addestramento

### 4.1 Viste

A ogni passo l'encoder vede **4 viste** in **un solo passaggio** su 4B sequenze: la sequenza **pulita** `p = v_0` e **tre estrazioni indipendenti** `v_1, v_2, v_3 = A(p)` della stessa pipeline, come le viste di LeJEPA. L'encoder non ha statistiche di batch, quindi le clip non si mescolano.

**La pipeline `A` tocca solo i disturbi** (per clip, dal generatore della GPU):

| Trasformazione | Probabilità | Valore | Fonte |
|---|---|---|---|
| Rotazione nel piano attorno all'origine fra le spalle | sempre | ±10° | rollio della camera (SPOTER usa ±13°) |
| Scala orizzontale (rapporto d'aspetto) | 0,5 | da 0,85 a 1,15 | SPOTER, schiacciamento fino al 15 % |
| Shear `x ← x + h·y` | 0,5 | \|h\| ≤ 0,1 | **[Nostra scelta]**: Moliner et al. usano ±1, ma su scheletri 3D visti da qualunque angolo |
| Rumore gaussiano sui giunti presenti | sempre | mani 0,01; corpo 0,003; volto 0,002 unità di spalla | il jitter misurato su OpenASL (sotto) |
| Giunti nascosti (presenza a 0) | 0,5 | 9 giunti per 8 passi consecutivi, fra i 57 di dita e volto | PSTL nasconde 9 giunti; quantità moderata come chiedono AimCLR e HiCLR |

- Rotazione, scala e shear formano una sola mappa lineare per tutta la clip: imitano una telecamera.
- I giunti nascosti escludono il corpo e le radici (polsi, punta del naso): una radice mancante azzererebbe le posizioni locali di tutta la parte.
- **Il jitter misurato.** Su 200 clip di OpenASL, dalle seconde differenze su tre frame video consecutivi (un limite superiore, perché include il moto vero): mani ≤ 0,008–0,009 unità di spalla, rispetto al polso; corpo ≤ 0,003; volto ≤ 0,002.
- **I giunti mancanti** su OpenASL sono rari: una mano manca del tutto nello 0,1–0,2 % dei frame, in parte nell'1,4–4,3 %. Nascondere giunti è quindi soprattutto un compito predittivo, utile anche per corpora più sporchi.

**Non si usano:**
- traslazione e scala globale, che la ricanonicalizzazione per frame annulla (non annulla invece shear e rapporto d'aspetto);
- il flip orizzontale, che scambia la mano dominante;
- il flip temporale, che inverte la direzione del movimento;
- le rotazioni grandi: l'orientamento è fonologico;
- il passa-basso temporale, che cancella i cambi rapidi di configurazione;
- il mixing fra scheletri, la rotazione del braccio di SPOTER (cambia la posa, non un disturbo) e la prospettiva.

Le augmentation geometriche della clip (ritaglio, stessa geometria su video e keypoint) restano quelle della pipeline e precedono le viste.

**Perché 4 viste [revisione del 6/10].**
- **Più di due viste aiutano, fino a quattro.**
  - LeJEPA: da 2 a 4 viste guadagna 1,2 punti; da 4 a 8 nessun guadagno (tabella di λ). Il suo codice ufficiale usa 4 viste.
  - W-MSE, uno sbiancamento come SIGReg: 4 viste meglio di 2, da +0,4 a +1,5 punti.
  - Negli scheletri, tre rami battono due: PSTL (+3,3 e +5,2 punti per vista mascherata), HiCLR (77,6 contro 74,0).
  - SSL-SLR, sulla lingua dei segni, perde molto se si toglie la sequenza originale.
- **Le viste deboli da sole non bastano.** BYOL su scheletri dà 51,1 con le sole viste conservative e 79,6 con quelle aggressive (Moliner et al.).
- **Quelle troppo forti fanno danno.** AimCLR passa da 75,0 a 71,3; HiCLR, con augmentation forti usate direttamente, scende a 56,7.

### 4.2 Termini

```
z_v = G_ω(v_v),  v = 0…3;   s = z_0 (la vista pulita)              (B,32,192) ciascuna
c_{b,t} = media della presenza dei 69 giunti al passo t (vista pulita)   passi validi: c_{b,t} > 0
μ_{b,t} = ¼ Σ_v z_{v,b,t}                                          il centro delle viste

L_inv       = media sui passi validi di  ¼ Σ_v ‖ μ_{b,t} − z_{v,b,t} ‖² / 192
L_anchor    = Σ_{b,t,j} c_{b,t,j} ‖ D(s_{b,t})_j − p̂_{b,t,j} ‖²  /  ( σ² Σ_{b,t,j} c_{b,t,j} )
SIGReg_posa = media su v e t di  SIGReg({ z_{v,b,t} : c_{b,t} > 0 }_b)   N_t ≤ 128, su tutte le GPU

L_0 = (1 − λ_P)·( L_inv + L_anchor ) + λ_P·SIGReg_posa             λ_P = 0,04
```

- **`L_inv`** è il termine predittivo di LeJEPA nella sua forma: ogni vista si avvicina al centro delle quattro, con la media sui canali. Con due viste varrebbe `‖s − s̃‖²/4C`.
- **`L_anchor`** è l'ancora del progetto (`worldsign-loss.md` §4) con un decoder lineare unico 192 → 138, sulla sola vista pulita:
  - p̂ è la posizione di ogni giunto mediata sui due frame del passo;
  - σ² è la varianza dei keypoint calcolata una volta sulle clip di training, così che predire la media valga 1.

  Tiene la cinematica nel bersaglio. È la «testa ausiliaria» di LeCun (§4.5.2).
- **`SIGReg_posa`** spinge s verso la gaussiana isotropa: impedisce il collasso, anche solo dimensionale, e rende tutte le direzioni ugualmente pesate nella L1 di `E_fis`.
  - È calcolato **per passo e per vista**: a ogni passo t i campioni sono le clip presenti lì, indipendenti fra loro.
  - Le stesse 1.024 direzioni servono tutte le viste e tutti i passi; sono nuove a ogni step e uguali su ogni GPU.
  - Un passo conta se ha almeno due clip presenti su tutte le GPU insieme.
- **Il bersaglio** di `E_fis` è `sg(s)` della vista pulita. Senza dropout è deterministico.

**Il collasso è escluso per costruzione:**
- un s costante annulla `L_inv`, ma non `L_anchor` (non ricostruisce i keypoint) e non `SIGReg_posa` (non è gaussiano);
- non c'è EMA né stop-gradient interno alla posa: è la ricetta di LeJEPA.

**Perché λ_P = 0,04 [revisione del 6/10].**
- LeJEPA con 4 viste usa 0,02, sia nella tabella di λ (ResNet-50, IN-100) sia nel codice ufficiale, con 256 campioni per SIGReg. Nella stessa tabella l'ottimo cresce con le viste: 0,01 per 2, 0,02 per 4, 0,05 per 8.
- Da noi ogni SIGReg ha al più 128 campioni. La statistica cresce con N quando la distribuzione non è gaussiana, quindi con 128 SIGReg spinge la metà che con 256: per lo stesso equilibrio λ/(1 − λ) raddoppia, e λ_P = 0,04.
- Gli altri parametri non chiedono correzioni:
  - le direzioni, i 32 passi e le 4 viste entrano come medie;
  - la statistica di ogni direzione non dipende da C;
  - l'ancora e le viste più miti di quelle su ImageNet spingono in direzioni opposte, senza una base quantitativa.
- LeWorldModel, anch'esso con N = 128 per passo, usa 0,09 (circa 0,083 nella nostra forma) e su Push-T resta stabile fra 0,01 e 0,2. Il valore 0,04 sta dentro questa zona.
- Il livello semantico resta a 0,05 finché non lo si rivede.

**Cosa aspettarsi.** Con viste che toccano solo i disturbi, l'astrazione resta vicina alla cinematica, ma la rappresentazione è robusta al rumore del rilevatore, al punto di vista e ai giunti persi, ed è distribuita su tutte le dimensioni.

### 4.3 Ottimizzazione

| Voce | Valore |
|---|---|
| Gruppo | proprio (famiglia `pose`: encoder e decoder) |
| LR di picco | **3e-4** (il valore di worldSign) |
| Schedule | warm-up lineare sul **20 %** dei passi della run, poi **coseno fino a 0** alla fine pianificata; il cooldown della run lo moltiplica |
| Weight decay | quello della run (0,04), non su norme, bias e posizioni |
| Attiva | dal passo 0 (stadio P, `worldsign-gerarchia.md` §6) |

Il coseno fa rallentare il bersaglio proprio mentre il video lo insegue, come il momentum dell'EMA che sale verso 1 in V-JEPA e S-JEPA.

### 4.4 Confronto con LeJEPA e LeWorldModel [revisione del 6/10]

| Punto | LeJEPA / LeWorldModel | Prima | Ora |
|---|---|---|---|
| Insieme su cui si calcola SIGReg | campioni indipendenti: per vista (LeJEPA), per passo (LeWM, App. D e H) | tutti i passi insieme: ≈ 4.096 campioni correlati | per passo e per vista, N ≤ 128 |
| Viste | LeJEPA: 4 nel codice ufficiale, 8–10 nel paper | 2, deboli | 4, ognuna con disturbi diversi |
| Forma dell'invarianza | distanza dal centro, media sui canali | `‖s − s̃‖²/C`, cioè 4 volte quella di LeJEPA | quella di LeJEPA |
| λ | 0,02 con 4 viste (LeJEPA); 0,09 (LeWM) | 0,05 condiviso | 0,04 per la posa |
| Dropout dell'encoder che dà il bersaglio | assente in entrambi | 0,1 | assente |
| Dimensione dello stato | LeWM: ottimo a 192 | 256 | 192 |

**Perché per passo.** Su dati sintetici con distribuzione esattamente N(0, I) a ogni passo, SIGReg sui passi messi insieme vale 1,05 se i passi sono indipendenti, 2,2 con correlazione 0,5 fra passi consecutivi, 10 con 0,9 e 22 con 0,98. Calcolato per passo resta a 1,03–1,06. Sui passi insieme, quindi, SIGReg spingeva anche a rendere diversi fra loro i passi di una clip, cioè `s` ruvido nel tempo. LeWM osserva l'effetto opposto: le traiettorie si raddrizzano proprio perché SIGReg non agisce lungo il tempo.

**Rimandata al livello semantico:** la quadratura. Il codice ufficiale di entrambi i paper integra su [0, 3] sfruttando la simmetria, con 17 nodi; noi su [−5, 5]. Le ablation dei paper non vedono differenze, e la quadratura è condivisa con `SIGReg_sem`.

### 4.5 ESP-8, rimandata

Con VICReg al posto di SIGReg, `L_0` diventerebbe `25·L_inv + 25·V(s) + 1·Cov(s) + w_a·L_anchor`, con i pesi di VICReg. Restano aperti il peso dell'ancora `w_a` e l'espansore (`worldsign-ablation.md` §2.3).

---

## 5. Collegamento al livello fisico

```
pose_tokens ─► G_ω ─► s (B,32,192) ──sg──► LN ──────────────────────────────────────────┐
clip ─► maschere ─► V-JEPA 2.1-L + LoRA (token visibili) ─► fusione ─► predictor fisico  │
     ─► media dei token predetti in ciascuno dei 4 riquadri al passo t                   │
     ─► concatenazione (B,32,4·384) ─► Linear(1536→192) ─► ŝ ─► E_fis(ŝ, LN(sg(s))) ◄────┘
```

La lettura dal video rispecchia il lato posa: una media per articolatore, poi la concatenazione dei 4 articolatori. La definizione di `E_fis` per passo è in `worldsign-gerarchia.md` §4.

---

## 6. Monitoraggio

La posa entra nell'addestramento al passo 0. Lo stadio P esiste per verificarla prima che il video la insegua (`worldsign-gerarchia.md` §7). Cosa si legge:

| Lettura | Cadenza | Allarme |
|---|---|---|
| `inv_posa`, `anchor`, `sigreg_posa` | ogni passo (log) | picchi della loss |
| `s_rank`, `s_std`, `s_isoscore` di s sui passi validi; `s_sigreg` passo per passo, come la loss (≈ 1 per una gaussiana) | letture frequenti | rango o deviazione sotto 0,5 × il passo 0; `s_sigreg` oltre 1,5 × il passo 0 |
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
| `signworld/models/worldsign/pose_branch.py` | `PoseViews` (la pipeline `A`), `KeypointDecoder`, `PoseBranch` (encoder, decoder, le 4 viste in un passaggio, scala dell'ancora) |
| `signworld/loss/worldsign.py` | `invariance` (forma di LeJEPA), `SIGRegLoss.per_step`, `Objective.pose_sigreg`, `Objective.weights` (λ per livello) |
| `signworld/loss/sigreg.py` | `SIGReg.per_group`: più gruppi di dimensioni diverse in un solo calcolo |
| `signworld/metrics/readings.py` | `stepwise_sigreg_ratio`, la lettura `s_sigreg` |
| `signworld/models/sjepa/` | la replica di S-JEPA |

Configurazione: `pose_encoder` (con `views`) e `losses.pose_sigreg_weight` in `parameters/model/worldsign.yaml`.

---

## 9. Aperto

- Ampiezze della pipeline delle viste (§4.1): fissate da letteratura e misure, da rivedere dopo le prime run.
- La quadratura di SIGReg, da decidere con il livello semantico (§4.4).
- Soglia di R² della fermata F1 (0,9) [PC7].
- ESP-8: peso dell'ancora ed espansore.
