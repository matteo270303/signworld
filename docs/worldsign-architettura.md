# WorldSign — Architettura

Descrizione completa del modello **com'è implementato** in `signworld/models/worldsign/`, componente per componente e layer per layer, con le forme dei tensori e il numero di parametri. I conteggi sono misurati sul codice (moduli istanziati), non stimati.

Documenti collegati:
- `worldsign-progetto.md`: motivazioni, letteratura, piano; i riferimenti [Lett. N] rimandano alla sua bibliografia;
- `worldsign-loss.md`: energie, loss e regolarizzazioni;
- `worldsign-ablation.md`: ablation concordate e possibili.

Stato: 30/9/2026. Le voci **[Aperto]** sono valori non ancora fissati dal documento di progetto, da calibrare in PC7.

---

## 1. Panoramica

Tre encoder, due predictor, due passaggi per passo di addestramento.

```
 ─── PASSAGGIO FISICO ── due maschere a tubi per passo (brevi e lunghe) ─────────────────────────
  video 64×256×256 ─► token visibili ─► ENCODER VIDEO (V-JEPA 2.1 ViT-L, congelato + LoRA r=16)
                                          │ uscite grezze dei blocchi 6/12/18/24
                                          ▼
                              FUSIONE MULTILIVELLO  4 LayerNorm · 4×1.024 → 1.024 → 384
                                          ▼
                 PREDICTOR FISICO (V-JEPA 2.1 rilasciato, 12 blocchi d=384, congelato + LoRA r=16)
                   ingresso: token visibili fusi + mask token nelle posizioni nascoste
                   uscita: un token per ogni posizione, visibile o nascosta
                                          ▼
                 LETTURA PER ARTICOLATORE  media nel riquadro (mascherati | visibili) → Linear 384→256
                                          ▼
                           ŝ^pred_{t,a}, ŝ^ctx_{t,a} ∈ ℝ²⁵⁶ ── E_fis ──► target LN(s_{t,a})
  posa 32×69×6 ─► ENCODER DI POSA (S-JEPA, congelato + LoRA r=4) ─► per articolatore
                  ─► LAYER FINALE per articolatore 256→256 ─► s_{t,a}   ─► D_pose ─► L_anchor

 ─── PASSAGGIO SEMANTICO ── clip intera, nessuna maschera ────────────────────────────────────────
  video ─► ENCODER VIDEO (lo stesso) ─► 8.192 token normalizzati del blocco 24
        ─► PREDICTOR SEMANTICO (4 blocchi d=384, 8 query, da zero) ─► ŷ ∈ ℝ⁵¹²  ── E_sem ──► ẽ
  didascalia ─► EmbeddingGemma (precalcolato, 768) ─► centratura per lingua ─► testa 768→512→512 ─► ẽ
```

**Notazione delle forme.**

| Simbolo | Valore | Significato |
|---|---|---|
| `B` | 64 | clip per GPU; il **batch effettivo è sempre 128** (64 × 2 GPU) |
| `T`, `T'` | 64, 32 | frame per clip; passi temporali dopo i tubelet da 2 frame |
| `H' = W'` | 16 | patch per lato a 256 pixel, patch da 16 |
| `N` | 8.192 | token per clip, `32 × 16 × 16` |
| `D` | 1.024 | larghezza dell'encoder video |
| `d_p` | 384 | larghezza dei predictor |
| `C` | 256 | larghezza del latente di posa `s_{t,a}` |
| `d` | 512 | spazio semantico di `ŷ` ed `ẽ` |

---

## 2. Ingressi

### 2.1 Video

| Passo | Dettaglio |
|---|---|
| Ritaglio | quadrato attorno al segnante (unione dei riquadri del rilevatore, +15 %), ridimensionato a **256×256**; nero fuori dal frame sorgente |
| Frame | **64 frame** dall'intera frase: 32 guidati dal moto locale (densi dove le mani si muovono), 32 uniformi nel tempo (`data/pose/frames.py`) |
| Aumentazione (solo addestramento) | jitter del riquadro: scala e spostamento fino al **±10 %** del lato, **la stessa trasformazione affine su frame, keypoint e riquadri** (P12); luminosità, contrasto e saturazione ±0,2, uguali per tutti i frame della clip **[Aperto]**; **nessun flip** orizzontale, che scambierebbe la mano dominante. La vista si estrae dal seme, dall'epoca e dalla clip: una run ripresa rivede le stesse viste |
| Normalizzazione | `uint8 → float / 255`, poi media `(0,485; 0,456; 0,406)` e deviazione `(0,229; 0,224; 0,225)` di ImageNet, per canale |
| Forma in ingresso | `(B, 3, 64, 256, 256)` |

### 2.2 Posa

| Passo | Dettaglio |
|---|---|
| Stima | RTMW, 133 keypoint per frame, sui 64 frame selezionati |
| Giunti usati | **69**: corpo 9, mano sinistra 21, mano destra 21, volto 18 |
| Coordinate | unità di spalla: origine fra le spalle, unità la loro distanza. Toglie posizione e scala, non l'orientamento, che è fonologico |
| Validità | un keypoint conta se lo score RTMW supera 1,0, è dentro il frame e sta entro 5 unità di spalla; altrimenti vale zero ed è segnato mancante |
| Token di S-JEPA | per passo e giunto, `(x, y, presenza)` dei due frame del tubelet: **6 canali**, forma `(B, 32, 69, 6)` |
| Keypoint dell'ancora `p̂_{t,j}` | media delle posizioni presenti nei due frame del passo, `(B, 32, 69, 2)` |
| Pesi `c_{t,j}` | 1 se il giunto è presente in almeno un frame del passo, altrimenti 0, `(B, 32, 69)` |
| Riquadri | per passo e articolatore: riquadro dei keypoint con score > 0,3 **[Aperto: 0,3 o 1,0]**, allargato del 10 % per lato, unione dei due frame del passo; almeno 3 keypoint affidabili, altrimenti non visibile. Forma `(B, 32, 4, 4)`, frazioni del frame |

### 2.3 Testo

La riga precalcolata di EmbeddingGemma (768 valori) della didascalia della clip, più l'indice della lingua della didascalia per la centratura (§7).

---

## 3. Encoder video — V-JEPA 2.1 ViT-L distillato

Il checkpoint è `vjepa2_1_vit_large_384`, distillato da un teacher ViT-G [Lett. 32, 72]. È **congelato**: si addestrano solo la LoRA e le LayerNorm dei blocchi. Serve entrambi i passaggi con gli stessi pesi.

| # | Layer | Forma | Parametri | Stato |
|---|---|---|---|---|
| 1 | **PatchEmbed3D**: `Conv3d(3 → 1.024, kernel = stride = (2, 16, 16))` | `(B, 3, 64, 256, 256) → (B, 8.192, 1.024)` | 1.573.888 | congelato |
| 2 | **Embedding di modalità video**: vettore `(1, 1, 1.024)` sommato a ogni token | — | 1.024 | congelato |
| 3 | **Rimozione dei token nascosti** (solo passaggio fisico): i token mascherati si **tolgono** dalla sequenza (`apply_masks`), non si azzerano | `(B, n_vis, 1.024)` | — | — |
| 4 | **24 blocchi transformer**, pre-norm (dettaglio sotto) | invariata | 24 × 12.596.224 | congelati + LoRA + LN |
| 5 | **`norms_block`**: 4 `LayerNorm(1.024, ε = 1e-6)`, una per i blocchi 6/12/18/24 | — | 4 × 2.048 | congelate (la fusione ne usa copie addestrabili) |

Nel checkpoint c'è anche `patch_embed_img` (`Conv3d` con tubelet 1, 787.456 parametri), usato solo per le immagini, e un embedding di modalità immagine (1.024): da noi non servono. **Totale congelato: 304.680.960 parametri.**

**Un blocco dell'encoder** (`dim = 1.024`, 16 teste da 64):

```
x ─┬─► LayerNorm(1.024, ε=1e-6)  [γ, β addestrabili]
   │     ─► qkv: Linear(1.024 → 3.072, bias)  + LoRA su q, k, v separati
   │     ─► RoPE 3D su q e k (sotto)  ─► attenzione (SDPA, scala 1/√64)
   │     ─► proj: Linear(1.024 → 1.024, bias)  + LoRA
   └──── + ◄┘                                         residuo (drop path 0)
x ─┬─► LayerNorm(1.024, ε=1e-6)  [γ, β addestrabili]
   │     ─► fc1: Linear(1.024 → 4.096)  + LoRA  ─► GELU (esatta)  ─► fc2: Linear(4.096 → 1.024)  + LoRA
   └──── + ◄┘                                         residuo; dropout 0
```

| Sottolayer | Parametri congelati |
|---|---|
| `norm1`, `norm2` | 2 × 2.048 (addestrabili) |
| `qkv` | 3.148.800 |
| `proj` | 1.049.600 |
| `fc1` | 4.198.400 |
| `fc2` | 4.195.328 |

**RoPE 3D di V-JEPA 2.1.** In ogni testa (64 dimensioni) le prime 20 dimensioni ruotano con l'indice del passo, le 20 successive con la riga, altre 20 con la colonna; **le ultime 4 non ruotano**. Le frequenze sono `ω_i = 10000^(−i/10)`, `i = 0 … 9`, su coppie di dimensioni. Con `interpolate_rope` le posizioni di riga e colonna si riscalano sulla griglia di pre-addestramento (256/16 = 16): a 256 pixel il fattore è 1.

**LoRA** [Lett. 67]. `W' = W + (α/r)·B·A`, con `r = 16`, `α = 16` (scala 1) **[Aperto: α]**. `A` è gaussiana con deviazione `1/r` e `B` è **zero**, quindi al passo 0 ogni layer calcola esattamente quello pre-addestrato. `qkv` ha un adattatore per ciascuna delle tre fette q, k, v.

| LoRA per blocco | Parametri |
|---|---|
| q, k, v: `3 × 16 × (1.024 + 1.024)` | 98.304 |
| proj: `16 × (1.024 + 1.024)` | 32.768 |
| fc1, fc2: `2 × 16 × (1.024 + 4.096)` | 163.840 |
| **Totale su 24 blocchi** | **7.077.888** |
| LayerNorm addestrabili (`norm1`, `norm2` × 24) | **98.304** |

**Uscite.**
- **Passaggio fisico:** le uscite *grezze* (prima di ogni norma) dei blocchi 6, 12, 18 e 24, sui soli token visibili, lette con hook senza modificare il forward di Meta.
- **Passaggio semantico:** su tutti gli 8.192 token, l'uscita del blocco 24 normalizzata da `norms_block[3]`. È quello che l'encoder restituisce fuori dall'addestramento di Meta.

Il ricalcolo delle attivazioni (activation checkpointing) è per blocco, non rientrante, quindi compatibile con DDP.

---

## 4. Livello fisico

### 4.1 Maschera

**Maschera multi-blocco di V-JEPA** [Lett. 30, 31], a **tubi**: ogni blocco copre **tutti i 32 passi**, quindi la maschera è **identica in ogni frame della clip**.

| Tipo | Blocchi | Area di ciascun blocco | Quota nascosta media |
|---|---|---|---|
| Brevi | 8 | 15 % del frame | 62 % |
| Lunghe | 2 | 70 % del frame | 81 % |

- Rapporto d'aspetto casuale fra 0,75 e 1,5; la dimensione del blocco si estrae una volta per maschera, le posizioni a caso; è nascosta l'unione dei blocchi.
- **Entrambi i tipi a ogni passo**, come V-JEPA: il passaggio fisico gira due volte.
- **Una maschera per tipo è condivisa dalle 64 clip del batch di una GPU** e cambia a ogni passo, su ogni GPU e fra i due tipi. Ogni token è visibile oppure nascosto; nessuno è scartato. Una maschera tutta nascosta si ri-estrae.

**Perché condivisa [Nostra scelta, 30/9].** V-JEPA estrae posizioni diverse per ogni clip e poi taglia le liste di ogni clip alla più corta del batch. I token tolti sono gli ultimi in ordine temporale. Con 64 clip per GPU questo toglierebbe il 43 % (brevi) e il 74 % (lunghe) dei token visibili, e la clip con più contesto lo vedrebbe solo nei primi 13 (o 6) passi su 32: la maschera non sarebbe più un tubo. Condividere la maschera nel batch conserva tubi esatti e i rapporti di V-JEPA.

### 4.2 Fusione multilivello

Sostituisce il `predictor_embed` del predictor rilasciato (§4.4.5 del progetto; `fusion.py`). **Default `mlp`**, lo schema di V-JEPA 2.1 per quattro livelli:

| # | Layer | Forma | Parametri | Stato |
|---|---|---|---|---|
| 1 | 4 `LayerNorm(1.024, ε = 1e-6)`, copie di `norms_block`, una per livello | 4 × `(B, n_vis, 1.024)` | 8.192 | addestrabili |
| 2 | concatenazione sui canali | `(B, n_vis, 4.096)` | — | — |
| 3 | `Linear(4.096 → 1.024)` | `(B, n_vis, 1.024)` | 4.195.328 | da zero |
| 4 | GELU | — | — | — |
| 5 | `Linear(1.024 → 384)` | `(B, n_vis, 384)` | 393.600 | da zero |

**Totale: 4.597.120.** Varianti **[Aperto]**:
- `linear`: un solo `Linear(4.096 → 384)` inizializzato come il layer rilasciato sull'ultimo livello e a zero sugli altri. Al passo 0 riproduce esattamente il predictor rilasciato (1,57 M).
- `residual`: il layer rilasciato, congelato, sull'ultimo livello più l'MLP sui quattro con l'ultimo layer a zero. Esatto al passo 0, con la capacità dell'MLP.

### 4.3 Predictor fisico — il predictor di V-JEPA 2.1 riusato

| # | Layer | Forma | Parametri | Stato |
|---|---|---|---|---|
| 1 | token visibili dalla fusione | `(B, n_vis, 384)` | — | — |
| 2 | **mask token**: 8 vettori da 384 inizializzati a zero nel pre-addestramento; si usa **l'indice 0**, l'unico addestrato nel checkpoint distillato (PC6). Ogni posizione nascosta riceve lo stesso vettore; la posizione entra solo con la RoPE | `(B, n_nasc, 384)` | 3.072 | congelati |
| 3 | concatenazione `[visibili ‖ nascosti]`, poi **riordino per posizione** nella clip | `(B, 8.192, 384)` | — | — |
| 4 | embedding di modalità video del predictor, sommato | — | 384 (+384 immagine, inutilizzato) | congelato |
| 5 | **12 blocchi** pre-norm, come quelli dell'encoder a `dim = 384`, 12 teste da 32 (sotto) | invariata | 12 × 1.774.464 | congelati + LoRA |
| 6 | `predictor_norm`: `LayerNorm(384, ε = 1e-6)` | invariata | 768 | congelata |
| 7 | riordino inverso; **predict_all**: un'uscita per ogni token, visibile o nascosto | `(B, 8.192, 384)` | — | — |
| 8 | `predictor_proj` e `predictor_proj_context` (384 → 1.664, verso il teacher ViT-G) **sostituiti dall'identità** | — | — | tolti |

**Un blocco del predictor.** `LayerNorm(384, ε=1e-6)` → `qkv: Linear(384 → 1.152)` + LoRA q, k, v → RoPE 3D → attenzione → `proj: Linear(384 → 384)` + LoRA → residuo → `LayerNorm(384)` → `fc1: Linear(384 → 1.536)` + LoRA → GELU → `fc2: Linear(1.536 → 384)` + LoRA → residuo.

- **RoPE 3D del predictor:** per testa (32 dimensioni), 10 dimensioni per passo, 10 per riga, 10 per colonna, 2 non ruotate. **La griglia è impostata a 16×16**: il modulo rilasciato la fissa a 24×24 (384 pixel) e non interpola, quindi a 256 pixel ogni token sarebbe decodificato nella riga e colonna sbagliate.
- **LoRA r = 16 sui 12 blocchi:** per blocco `3·16·768 + 16·768 + 2·16·1.920 = 110.592`; **totale 1.327.104**.
- **Parti congelate: 21.298.176** (blocchi 21.293.568, mask token, embedding di modalità, norma). Il modulo rilasciato completo ne ha 22.973.056: la differenza sono l'embedding e le due proiezioni sostituiti.

### 4.4 Lettura per articolatore

Il target è per passo e articolatore, non per token (`readout.py`).

1. **Appartenenza:** ogni riquadro visibile `(x0, y0, x1, y1)` si proietta sulla griglia 16×16 (floor dell'angolo in alto a sinistra, ceil di quello in basso a destra); il riquadro copre ogni patch che tocca.
2. **Due letture** per passo `t` e articolatore `a`, perché V-JEPA 2.1 tiene separati i due termini della loss:
   - `ŝ^pred_{t,a}`: media dei token **nascosti** predetti nel riquadro;
   - `ŝ^ctx_{t,a}`: media dei token **visibili** predetti nel riquadro.
3. **Testa:** `Linear(384 → 256)`, **una sola, condivisa** dalle due letture; 98.560 parametri, da zero.
4. Per la loss si contano anche, per riquadro, i token nascosti `n^mask`, i visibili `n^vis` e la somma dei pesi dei visibili (1 ciascuno nel cooldown di V-JEPA 2.1, che seguiamo).

---

## 5. Encoder di posa — S-JEPA

S-JEPA [Lett. 99] pre-addestrato da noi in PC5; si usa **l'encoder EMA** (`target_embed`, `target_encoder` del checkpoint `pose-teachers/sjepa.pt`). **Nessuna RoPE**: la posizione entra con embedding appresi.

| # | Layer | Forma | Parametri | Stato |
|---|---|---|---|---|
| 1 | `Linear(6 → 256)` su ogni token giunto × passo | `(B, 32, 69, 6) → (B, 2.208, 256)` | 1.792 | congelato |
| 2 | embedding del giunto `(69, 256)` + embedding del passo `(32, 256)`, sommati al token | — | 17.664 + 8.192 | congelati |
| 3 | **8 `TransformerEncoderLayer`** di PyTorch, `norm_first = True` (sotto) | invariata | 8 × 789.760 | congelati + LoRA |
| 4 | `LayerNorm(256, ε = 1e-5)` finale | invariata | 512 | congelata |
| 5 | **media per articolatore** dei token dei suoi giunti (9, 21, 21, 18) | `(B, 32, 4, 256)` | — | — |
| 6 | **layer finale per articolatore**: 4 × `Linear(256 → 256)` (`ArticulatorLinear`), **inizializzato all'identità** | `(B, 32, 4, 256)` = `s_{t,a}` | 263.168 | addestrabile |

**Un layer di S-JEPA** (`d = 256`, 8 teste da 32, dropout 0):

```
x ─┬─► LayerNorm(256, ε=1e-5)
   │     ─► MultiheadAttention: in_proj (256 → 768, bias) + LoRA su q, k, v separati
   │                            out_proj (256 → 256, bias) + LoRA
   └──── + ◄┘
x ─┬─► LayerNorm(256, ε=1e-5)
   │     ─► linear1 (256 → 1.024) + LoRA ─► GELU ─► linear2 (1.024 → 256) + LoRA
   └──── + ◄┘
```

- **LoRA r = 4, α = 4 (scala 1) [Aperto: α].** `MultiheadAttention` legge `in_proj_weight` e `out_proj.weight` direttamente, senza chiamare i Linear, quindi la LoRA è una **parametrizzazione del peso** (`W ↦ W + B·A`) invece di uno strato avvolto. Per layer `3·4·512 + 4·512 + 2·4·1.280 = 18.432`; **totale 147.456**, learning rate **×0,05** sullo schedule principale, come l'encoder del target di VL-JEPA [Lett. 34].
- **Totale congelato: 6.346.240.** Al passo 0, con B = 0 e il layer finale identità, l'uscita coincide con quella di S-JEPA (test dedicato).
- **Layer finale, schedule proprio [Nostra scelta, 30/9]:** addestrabile dal passo 0. Il learning rate ha un picco pari a 0,5 volte quello base **[Aperto]**, sale linearmente nel primo **10 %** della run, poi scende con un **coseno fino a 0 al 50 %** della run **[Aperto]** e da lì il layer resta fermo. Il target si assesta, come un teacher il cui momento tende a 1 (DINO, BYOL, V-JEPA) o un layer ricotto a zero prima degli altri (FreezeOut).
- **ESP-3:** encoder tutto congelato, senza LoRA e senza layer finale.

### 5.1 Teste dell'ancora `D_pose`

Una `Linear` per articolatore, da `s_{t,a}` alle coordinate `(x, y)` dei suoi giunti:

| Articolatore | Layer | Parametri |
|---|---|---|
| corpo | `Linear(256 → 18)` | 4.626 |
| mano sinistra | `Linear(256 → 42)` | 10.794 |
| mano destra | `Linear(256 → 42)` | 10.794 |
| volto | `Linear(256 → 36)` | 9.252 |
| **Totale** | | **35.466** |

Le uscite, concatenate nell'ordine dei giunti, danno `D_pose(s)` di forma `(B, 32, 69, 2)`.

---

## 6. Livello semantico — predictor semantico

Lo schema di VL-JEPA [Lett. 34], da zero (`semantic.py`).

| # | Layer | Forma | Parametri |
|---|---|---|---|
| 1 | ingresso: gli 8.192 token normalizzati del blocco 24 (§3) | `(B, 8.192, 1.024)` | — |
| 2 | `inputs: Linear(1.024 → 384)` | `(B, 8.192, 384)` | 393.600 |
| 3 | **8 query apprese**, costanti (non dipendono dalla clip), inizializzate `N(0, 0,02²)`, accodate ai token video | `(B, 8.200, 384)` | 3.072 |
| 4 | **4 blocchi** pre-norm (sotto) | invariata | 4 × 1.775.232 |
| 5 | `LayerNorm(384, ε = 1e-5)` sulle 8 uscite delle query | `(B, 8, 384)` | 768 |
| 6 | **media delle 8 query** (in ESP-4: 4 gruppi da 2, una media per gruppo) | `(B, K, 384)` | — |
| 7 | `outputs: Linear(384 → 512)` | `(B, K, 512)` = `ŷ` | 197.120 |

**Un blocco del predictor semantico** (12 teste da 32):

```
x ─┬─► LayerNorm(384, ε=1e-5)
   │     ─► qkv: Linear(384 → 1.152) ─► RoPE 3D sui soli token video ─► attenzione bidirezionale
   │        (SDPA, dropout 0,1 sulle probabilità)  ─► proj: Linear(384 → 384)
   │     ─► Dropout(0,1) ─► LayerScale γ₁ (384, init 1e-4) ─► DropPath(0,1)
   └──── + ◄┘
x ─┬─► LayerNorm(384, ε=1e-5)
   │     ─► fc1: Linear(384 → 1.536) ─► GELU ─► Dropout(0,1) ─► fc2: Linear(1.536 → 384) ─► Dropout(0,1)
   │     ─► LayerScale γ₂ (384, init 1e-4) ─► DropPath(0,1)
   └──── + ◄┘
```

- **RoPE 3D nostra** (`rope.py`): per testa 10 dimensioni per il passo, 10 per la riga, 12 per la colonna, base 10.000. **Le query non ruotano**: guardano il video per contenuto, mentre i token video conservano il loro ordine nel tempo. Senza posizione, attenzione e media sarebbero cieche all'ordine.
- **DropPath** (stochastic depth): scarta l'intero ramo residuo per un campione, con probabilità 0,1, solo in addestramento. **LayerScale**: un fattore per canale, inizializzato a 1e-4, che parte vicino all'identità. Dropout, stochastic depth e LayerScale **[Aperto: PC7]**.
- **Totale: 7.695.488**, tutti addestrabili.

---

## 7. Ramo testuale

### 7.1 EmbeddingGemma-300M — congelato, precalcolato fuori dall'addestramento

| Passo | Dettaglio |
|---|---|
| Tokenizzatore | Gemma 3, vocabolario di 262.144 token, al massimo 2.048 |
| Encoder | 24 blocchi transformer, `d = 768`, 3 teste, MLP interno 1.152, **attenzione bidirezionale** |
| Pooling | media dei token |
| Dense | `768 → 3.072` e `3.072 → 768`, senza bias né attivazione |
| Normalizzazione | L2: vettori unitari |
| Identità | modello, commit e prompt (vuoto) in un'impronta verificata a ogni uso; il vettore si usa **intero, a 768** (nessun troncamento MRL) |

### 7.2 Centratura per lingua

`e° = e − μ_ℓ`, con `μ_ℓ` la media delle **didascalie distinte delle clip di training** nella lingua `ℓ` della didascalia, come l'ha misurata PC1. È un buffer, non un parametro, e viaggia con i checkpoint. Una lingua mai vista in training è un errore esplicito (§4.6 del progetto: niente dalla validazione).

### 7.3 Testa

| # | Layer | Forma | Parametri |
|---|---|---|---|
| 1 | `Linear(768 → 512)` | `(B, 512)` | 393.728 |
| 2 | GELU | — | — |
| 3 | `Dropout(0,1)` **[Aperto]** | — | — |
| 4 | `Linear(512 → 512)` | `(B, 512)` = `ẽ` | 262.656 |

Inizializzazione standard di PyTorch. **Totale: 656.384.** È l'unica parte addestrabile del ramo testuale.

---

## 8. Come si usano le uscite

| Uscita | Uso |
|---|---|
| `ŝ^pred`, `ŝ^ctx` contro `LN(s)` | energia fisica `E_fis`; mediata su più maschere, plausibilità `Ē_fis` |
| `D_pose(s)` contro `p̂` | ancora `L_anchor` |
| `s` per articolatore | `SIGReg_posa` |
| `ŷ` contro `ẽ` | energia semantica `E_sem = 1 − cos` (o InfoNCE nel braccio C); retrieval, riconoscimento e profilo temporale come `argmin` dell'energia |
| `ŷ` ed `ẽ` separatamente | `SIGReg_sem` |

Le formule sono in `worldsign-loss.md`.

---

## 9. Parametri — riepilogo misurato

| Componente | Congelati | Addestrabili | Tipo |
|---|---|---|---|
| Encoder video ViT-L | 304.680.960 | — | — |
| ├ LoRA r = 16, 24 blocchi | — | 7.077.888 | vincolato |
| └ LayerNorm dei blocchi | — | 98.304 | vincolato |
| Predictor fisico (12 blocchi, d = 384) | 21.298.176 | — | — |
| ├ LoRA r = 16 | — | 1.327.104 | vincolato |
| ├ fusione multilivello (4 LN + MLP) | — | 4.597.120 | libero |
| └ testa di lettura 384 → 256 | — | 98.560 | libero |
| Predictor semantico | — | 7.695.488 | libero |
| Encoder di posa S-JEPA | 6.346.240 | — | — |
| ├ LoRA r = 4 | — | 147.456 | vincolato |
| └ layer finale per articolatore | — | 263.168 | libero |
| Teste `D_pose` | — | 35.466 | libero |
| Testa testuale | — | 656.384 | libero |
| InfoNCE (solo braccio C) | — | 1 (temperatura) | libero |
| EmbeddingGemma-300M | 0 in GPU (precalcolato) | — | — |
| **Totale** | **≈ 332,3 M** | **21.996.938** | 8,65 M vincolati · 13,35 M liberi |

- **Margine sul tetto di 22 M: 3.062 parametri.** L'asserzione P3 del progetto (21,6 M ± 1 %) va aggiornata.
- **ESP-3** (posa congelata): 21.586.314.
- **ESP-2** (senza livello fisico): restano encoder video con LoRA e norme, predictor semantico e testa testuale.

---

## 10. Esecuzione

| Voce | Scelta |
|---|---|
| Parallelismo | **DDP** su 2 H100, un processo per GPU (`torchrun`). FSDP e ZeRO dividono lo stato del modello, qui circa 1 GB, mentre la memoria va alle attivazioni |
| Batch | **effettivo 128**, 64 clip per GPU, **nessun accumulo di gradienti** |
| Termini sul batch intero | SIGReg (funzione caratteristica e N di tutte le GPU), InfoNCE e L_unif (tutte le coppie) si calcolano sulle 128 clip con collettive dotate di backward. Un test a due processi verifica gradienti identici a quelli di un processo sull'intero batch |
| Casualità | maschere diverse su ogni GPU; direzioni di SIGReg identiche su tutte; tutto riproducibile da seme, passo e rank |
| Precisione | bf16 (autocast), pesi in fp32 |
| Curriculum | con `requires_grad`: una parte ferma non costa backward. Nello stadio 2a il backward non entra nell'encoder video |
