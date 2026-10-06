# WorldSign — Ramo generativo: dal testo al video

Questo documento descrive il **ramo generativo** di WorldSign: dato un testo, genera il video di un segnante che lo esprime. Si appoggia alla gerarchia già addestrata e ne riusa tre elementi, tutti congelati: lo spazio dei token dell'encoder video, la rappresentazione del testo e il livello semantico. Ha due reti proprie:

- un **generatore** dal testo ai latenti video;
- un **autoencoder** che porta quei latenti in pixel.

**Non fa parte del piano attuale.** In `worldsign-progetto.md` §1.2 la produzione è un compito secondario, da ottenere dopo la gerarchia.

Documenti collegati:
- `worldsign-architettura.md`: l'encoder video, la griglia dei token, la testa testuale;
- `worldsign-gerarchia.md`: i livelli, il flusso del gradiente, la LoRA video;
- `worldsign-progetto.md` §3.6: il campionamento dei 64 frame; §4.4.4: EmbeddingGemma e la centratura per lingua.

Stato: bozza del 4/10/2026 **[Nostra proposta]**. Le decisioni del §1 sono prese; i valori segnati **[Aperto]** si fissano nelle prime prove. Nessun codice.

---

## 1. Decisioni

| Punto | Decisione |
|---|---|
| Compito | **solo testo → video**: l'ingresso è il testo, l'uscita il video |
| Spazio dei latenti | i token dell'encoder video **con la LoRA della gerarchia**, così che portino anche la fisica della posa |
| Rappresentazione del testo | **EmbeddingGemma**: gli stati per token e il vettore `ẽ` della gerarchia |
| Campionamento dei frame | **resta quello della gerarchia** (64 frame, metà guidati dal moto); il tempo si gestisce adattando **VFRTok** (§6) |
| Compressione temporale | **4 frame per passo latente** |
| Posa | **esclusa**: nessuna posa come condizione o come bersaglio. La posa ristimata con RTMW resta solo una metrica (§10) |
| Aspetto del segnante | **un segnante fisso**, che ha dato il consenso; entra come riferimento costante (§5.7) |
| Dimensioni | **6 teste da 64 dimensioni** (larghezza 384) in tutti i transformer; parametri contenuti, il generatore ancora di più (§11) |
| Allineamento REPA | **mantenuto**, solo in addestramento (§5.5) |

---

## 2. Collocazione rispetto alla gerarchia

- **Cosa si riusa, congelato:**
  - l'encoder V-JEPA 2.1-L con la **LoRA video** del checkpoint scelto;
  - EmbeddingGemma, con lo **stesso prefisso fissato** della gerarchia;
  - la centratura per lingua e la testa testuale, che danno `ẽ`;
  - il livello semantico, ma solo per valutare (§10).
- **Nessun gradiente torna verso la gerarchia.** Il ramo si addestra in run separate, con ottimizzatori propri.
- **Dipendenza dal checkpoint.** Se la gerarchia si riaddestra, cambiano lo spazio dei token e `ẽ`: il ramo va riaddestrato.
- **Budget separato.** Il tetto di 30 M parametri riguarda la gerarchia, non questo ramo.

### 2.1 Perché generare nello spazio dei token

| Ragione | Dettaglio | Fonte |
|---|---|---|
| **Lo schema a due stadi funziona** | Dal testo si genera prima una rappresentazione, poi un decoder la porta in pixel. Per entrambi gli stadi, i modelli di diffusione sono stati la scelta migliore | unCLIP [R13] |
| **Gli spazi auto-supervisionati sono buoni spazi generativi** | Generare nello spazio di un encoder congelato (DINOv2, SigLIP, MAE) e decodificare con un decoder addestrato a parte batte i VAE | RAE [R1] |
| **…e scalano al testo** | Dal testo, nello spazio di SigLIP-2: converge circa 4 volte più in fretta dei VAE e non va in overfitting in fine-tuning (i VAE sì, dopo 64 epoche) | Scale-RAE [R2] |
| **…e al video con V-JEPA** | Encoder V-JEPA 2 e 2.1 congelati, latenti compressi, generazione anche dal testo: convergenza circa 5 volte più rapida degli autoencoder | VideoRAE [R3] |
| **La gerarchia legge ciò che si genera** | I latenti nascono dallo spazio della gerarchia: il video generato si valuta con il livello semantico, cioè con lo stesso modello che fa il retrieval | [Nostra argomentazione] |
| **La fisica della posa è già nei token** | La LoRA video è addestrata dal livello fisico a rendere la posa predicibile dal video; le mani si leggono linearmente già da V-JEPA 2.1-L congelato (R² 0,706, PC3) | `worldsign-progetto.md` §4.4.1 |

---

## 3. Schema

```
A · AUTOENCODER  (solo video, tutte le clip)

64 frame del campionamento della gerarchia, con i loro istanti t_i
   │
   ▼
V-JEPA 2.1-L + LoRA ❄ ── uscite dei blocchi 8–24, sommate ──► F: 32 × 16 × 16 × 1.024  (+ istanti τ_k)
   │
   ▼
Proiettore P ──► z: 16 × 16 × 16 × 64, a istanti uniformi u_j
   │
   ▼
Decoder R (query agli istanti chiesti, frequenza f) ──► frame 256 × 256
                                           ▲
             supervisione: frame veri della clip agli stessi istanti


B · GENERATORE  (coppie testo–video)

didascalia ──► EmbeddingGemma ❄ ──► token del testo  L × 768 ──────────── cross-attention ─┐
          └──► centratura + testa ❄ ──► ẽ (256) ──┐                                         │
durata T ─────────────────────────────────────────┼──► adaLN-single ───────────────────────┤
lingua dei segni ─────────────────────────────────┤                                         │
tempo del flusso ─────────────────────────────────┘                                         │
riferimento: 16 frame dello stesso video ─► encoder ❄ ─► P ❄ ─► prefisso (256 token) ───────┼─► G ─► velocità
bersaglio: z = P(clip vera) ❄                                                               │
                                                                                            ┘

INFERENZA  (solo testo)

testo ─► EmbeddingGemma ─► token + ẽ ─► durata T̂ (regressore)
      ─► G: 30 passi da rumore, con il prefisso del segnante fisso (calcolato una volta) e la guida
      ─► ẑ: 16 × 16 × 16 × 64 ─► R a 25 fps su [0, T̂], a finestre ─► video 256²
```

---

## 4. Forme e notazione

| Simbolo | Forma | Cos'è |
|---|---|---|
| `x` | 64 × 256 × 256 × 3 | i 64 frame del campionamento della gerarchia |
| `t_i` | 64 | istante di ogni frame selezionato, in secondi dall'inizio della clip |
| `T` | scalare | durata della clip, in secondi |
| `F` | 32 × 16 × 16 × 1.024 | somma delle uscite dei blocchi 8–24 dell'encoder: 8.192 token |
| `τ_k` | 32 | istante di ogni passo di token: `τ_k = (t_{2k} + t_{2k+1}) / 2` |
| `z` | 16 × 16 × 16 × 64 | latente: 4.096 posizioni da 64 canali, 4 frame d'ingresso per passo |
| `u_j` | 16 | istanti uniformi dei passi latenti: `u_j = (j + ½) · T / 16` |
| `f` | scalare | frequenza dei frame in uscita da R, in fps |
| `s_m` | n | istanti dei frame in uscita: `s_m = (m + ½) / f` |
| `e_tok` | L × 768 | stati per token di EmbeddingGemma (prima della media) |
| `ẽ` | 256 | vettore del testo della gerarchia (centratura + testa; 512 fino al 6/10) |
| `ℓ` | indice | lingua dei segni da produrre |

---

## 5. Componenti

**Convenzioni comuni ai transformer (P, R, G):**
- larghezza 384, **6 teste da 64 dimensioni**, pre-norm;
- in P e R il FFN è un **GEGLU** con dimensione interna 4 × 384, come VideoRAE [R3]; in G è un MLP con GELU, come DiT [R8];
- attenzione con la scaled-dot-product di PyTorch, in bf16.

### 5.1 Encoder video (congelato)

- **Modello.** V-JEPA 2.1-L (24 blocchi, larghezza 1.024) con la LoRA video della gerarchia. Si usa senza maschera, sulla clip intera, come il livello semantico.
- **Feature.** Si **sommano elemento per elemento** le uscite dei blocchi 8–24.
  - In VideoRAE, con il solo strato 24: PSNR 26,59 e gFVD 51. Con gli strati 8–24: 29,39 e 40 [R3].
  - Sulla somma si applica una LayerNorm senza parametri, perché i blocchi non dominino per scala **[Nostra scelta]**.
- **Tempo.** L'encoder conserva la sua RoPE per indice; non lo tocchiamo. Ogni passo di token riceve `τ_k`, la media dei due frame selezionati del suo tubelet. Col nostro campionamento i due frame possono essere lontani: l'encoder è già addestrato così.
- **Costo.** Nello stadio A1 l'encoder gira al volo: le feature di una clip occupano 16 MB in fp16, e precalcolarle per tutte le clip non è praticabile.

### 5.2 Proiettore P

**Compito.** Comprimere `F` (8.192 × 1.024) in `z` (4.096 × 64) e **ricampionare il tempo** da `τ_k`, irregolari, a `u_j`, uniformi.

**Struttura** (dal tokenizer di VideoRAE e da VFRTok [R3, R4]):
1. **Proiezione d'ingresso.** `F` → LayerNorm → Linear 1.024 → 384.
2. **Query latenti.** 16 × 16 × 16 = 4.096 query, tutte uguali a un unico token appreso, come in VideoRAE: la posizione arriva solo dalla RoPE.
3. **Sequenza.** 8.192 token d'ingresso + 4.096 query = 12.288 posizioni.
4. **Corpo.** 8 blocchi transformer.
5. **RoPE.**
   - Ogni posizione ha una coordinata temporale **in secondi**, `τ_k` per i token e `u_j` per le query, e una spaziale (riga, colonna da 0 a 15).
   - **RoPE parziale**, come VFRTok: 3 teste su 6 ruotano, 3 no.
6. **Uscita.** Si tengono le 4.096 query, che passano per LayerNorm e Linear 384 → 64, e danno `z`.

**Ingressi da 16 frame.** P deve gestire anche clip da 16 frame, per il riferimento (§5.7). In addestramento il 25 % degli esempi è una finestra di 16 frame, con 4 query temporali **[Nostra proposta]**.

### 5.3 Il latente `z`

- **Forma.** 16 passi × 16 × 16 × 64. Ogni passo copre `T / 16` secondi, cioè 4 frame d'ingresso, la compressione decisa.
- **Tempo uniforme.** Il ricampionamento lo ha già fatto P: il generatore non deve produrre istanti.
- **Normalizzazione per G.** Ogni canale è standardizzato con media e deviazione calcolate sulle clip di addestramento **[Nostra scelta]**.
- **Rumore per R.** In addestramento, R riceve `z + σ·ε` con `σ` uniforme in [0, σ_max]: impara così a tollerare i latenti imperfetti del generatore (RAE [R1]). σ_max è **[Aperto]**.

### 5.4 Decoder R

**Compito.** Rendere i frame a **istanti qualsiasi**, a partire da `z`.

**Struttura** (dal decoder di VideoRAE e da VFRTok [R3, R4]):
1. **Proiezione d'ingresso.** `z` → Linear 64 → 384.
2. **Query d'uscita.**
   - Un **tubelet d'uscita** è un gruppo di 4 frame consecutivi alla frequenza `f`; per ognuno c'è una griglia di 16 × 16 query.
   - Tutte le query sono un unico token appreso. Ricevono la RoPE al **centro temporale** del loro tubelet, in secondi, e un embedding della frequenza `f`, perché R conosca la spaziatura dei frame **[Nostra proposta]**.
3. **Corpo.** 12 blocchi transformer, con la stessa RoPE parziale in secondi di P. Attenzione sui 4.096 latenti più le query della finestra.
4. **Uscita.** Le query passano per LayerNorm, poi per una deconvoluzione 3D (384 → 3 canali, kernel e passo 4 × 16 × 16): ogni query diventa un blocco di 4 frame × 16 × 16 pixel. I pixel sono in [−1, 1].

**Finestre.**
- R rende **finestre di 32 frame**: 8 tubelet × 256 = 2.048 query, più i 4.096 latenti, cioè 6.144 posizioni.
- In addestramento si sceglie una finestra a caso per clip.
- All'inferenza le finestre si susseguono con 8 frame di sovrapposizione, uniti con una dissolvenza lineare **[Nostra proposta]**.

**Frequenza in addestramento.** `f` si estrae a caso da {8, 12, 16, 20, 25} fps; i frame bersaglio sono letti dal video originale ai rispettivi `s_m` (§6.3).

### 5.5 Testa REPA (solo in addestramento)

- **Cosa fa.** Allinea le feature interne di R alle feature dell'**ultimo blocco dell'encoder con la LoRA**, lo stesso strato letto dal livello semantico.
- **Perché.**
  - Il gradiente attraversa R e raggiunge P, così `z` conserva la struttura semantica dell'encoder invece di diventare un codice buono solo per i pixel.
  - In VideoRAE sostituisce la regolarizzazione KL dei VAE e migliora la generazione: gFVD da 67 a 40 nel latente discreto, da 105 a 93 nel continuo [R3].
  - Con un generatore piccolo come il nostro, un latente facile da generare conta di più **[Nostra argomentazione]**.
  - L'idea viene da REPA, che allinea gli stati interni di un generatore alle feature di un encoder pre-addestrato e accelera l'addestramento di oltre 17,5× [R12].
- **Struttura.**
  - MLP 384 → 768 → 1.024 con GELU, come il `FeatureAligner` di VideoRAE.
  - Si applica alle query in uscita dal **blocco 4 di R**, uno strato precoce come in VideoRAE. Il numero esatto è **[Aperto]**.
- **Allineamento nel tempo.** Le query di R stanno agli istanti `s_m`, mentre i bersagli stanno agli istanti `τ_k`. Le feature di R si interpolano linearmente sugli istanti `τ_k` prima del confronto **[Nostra scelta]**.
- **Perdita.** `L_REPA = L_local + λ_g · L_global`:
  - `L_local` è il meno coseno medio token per token;
  - `L_global` è il meno coseno fra le medie [R3].
- **Uso.** Si scarta dopo l'addestramento.

### 5.6 Discriminatore (solo in addestramento)

- **Struttura [Nostra proposta].** PatchGAN 3D su finestre di 16 frame a 256²: quattro convoluzioni 3D (64, 128, 256 e 256 canali; kernel 3 × 4 × 4; passo 1 × 2 × 2 in tempo × spazio) più una convoluzione d'uscita a un canale. Circa **5,1 M** parametri.
- **Perdita.** Hinge.
- **Peso adattivo.** Il peso della perdita avversaria segue il rapporto fra i gradienti della ricostruzione e della perdita avversaria sull'ultimo strato di R, come VQGAN [R9].
- **Avvio ritardato.** Entra solo dopo una fase iniziale di sola ricostruzione, come in VQGAN. La durata è **[Aperto]**.
- VideoRAE non riporta il suo discriminatore.

### 5.7 Generatore G

**Compito.** Generare `z` dal testo, partendo da rumore.

**Ingresso.**
1. `z` normalizzato (§5.3) viene diviso in patch 1 × 2 × 2: 16 × 8 × 8 = **1.024 token da 256 valori**.
2. Linear 256 → 384.
3. RoPE 3D piena: tempo in secondi `u_j`, righe e colonne da 0 a 7.

La larghezza rispetta il vincolo di RAE: almeno pari alla dimensione dei token che si generano, 384 ≥ 256 [R1].

**Il riferimento del segnante.**

| | In addestramento | All'inferenza |
|---|---|---|
| Da dove viene | 16 frame dello stesso video, presi a passo uniforme in un **altro momento** e passati per encoder e P | 16 frame del **segnante fisso**, calcolati una volta |
| Cosa insegna o fa | G impara a copiare l'aspetto dal prefisso | fissa l'identità del video generato |

- **Forma.** Il latente di 16 frame (4 × 16 × 16 × 64) diventa 4 × 8 × 8 = **256 token di prefisso**, con la stessa proiezione d'ingresso.
- **Tempo e segmento.** Il prefisso sta a istanti negativi `[−T_ref, 0)` e riceve un embedding di segmento appreso.
- **Rumore.** Si aggiunge rumore al prefisso, come in Cosmos Video2World [R7]. In addestramento la sua ampiezza è casuale; all'inferenza è piccola e fissa. I valori sono **[Aperto]**.
- **Sequenza.** 1.024 + 256 = 1.280 token. La perdita si calcola solo sui 1.024 token bersaglio.

**Condizioni.**

| Condizione | Come entra | Spenta a caso in addestramento |
|---|---|---|
| Token del testo `e_tok` (L × 768, maschera sul padding) | cross-attention in ogni blocco: chiavi e valori proiettati da 768 a 384 | 10 %, sostituiti da un token nullo appreso |
| `ẽ` | adaLN-single | insieme ai token, sostituito da un vettore nullo appreso |
| Durata `T` (embedding sinusoidale del logaritmo) | adaLN-single | mai |
| Lingua dei segni `ℓ` (embedding appreso) | adaLN-single | mai. **Con solo OpenASL è costante**; serve sul corpus multilingue, dove la stessa frase scritta si può segnare in lingue diverse **[Nostra proposta]** |
| Riferimento | prefisso | 30 %, sostituito da un prefisso nullo appreso |

**Il tempo del flusso.** Embedding sinusoidale → MLP 256 → 384 → 384.

**adaLN-single** (PixArt-α [R5]).
- Un **solo MLP globale** prende la somma di tempo del flusso, `ẽ` proiettato, durata e lingua. Produce sei vettori, cioè spostamento, scala e gate per l'attenzione e per l'MLP, condivisi da tutti i blocchi.
- Ogni blocco aggiunge un proprio vettore appreso.
- In PixArt-α le proiezioni adaLN pesavano il 27 % del modello, e questa condivisione ha ridotto i parametri da 833 M a 611 M.

**Blocco** (× 12):
1. auto-attenzione modulata, sulle 1.280 posizioni;
2. cross-attention sui token del testo, con LayerNorm propria;
3. MLP modulato (GELU, 4 × 384).

I gate partono da zero e anche lo strato d'uscita parte da zero (adaLN-Zero di DiT [R8]).

**Uscita.** adaLN finale → Linear 384 → 256 → si ricompongono le patch: la **velocità** sulla griglia 16 × 16 × 16 × 64.

### 5.8 Regressore di durata

- **Ingresso.** `ẽ` concatenato a `log(1 + numero di parole)`.
- **Struttura.** MLP 257 → 256 → 2, che dà media e log-deviazione del **logaritmo della durata**.
- **Perdita.** NLL gaussiana.
- **All'inferenza.** `T̂ = exp(μ)`, limitato all'intervallo delle durate di addestramento **[Nostra scelta]**.

---

## 6. Il tempo: VFRTok adattato al nostro campionamento

### 6.1 Il problema
- La gerarchia vede 64 frame per clip, **metà scelti dove c'è più movimento**: gli istanti `t_i` non sono equidistanti.
- I token portano quindi un tempo deformato. Decodificati così come sono, darebbero un video con il ritmo sbagliato.

### 6.2 Cosa fa VFRTok [R4]
- **Tempo in secondi nella RoPE.** La coordinata temporale della RoPE è il **tempo in secondi**, non l'indice del frame: l'angolo di rotazione per secondo è fisso, `θ = C · (t / fs) · 10000^(−6c/n)`.
- **Addestramento asimmetrico.** Encoder e decoder vedono la stessa clip a frequenze diverse (12–30 fps in pre-addestramento, fino a 120 fps in fine-tuning). Il decoder impara così a rendere a **istanti arbitrari**.
- **RoPE parziale.** Metà delle teste ha la RoPE 3D, metà nessuna RoPE, per non legare troppo i latenti alla posizione.

### 6.3 Come lo adattiamo
1. **Il campionamento della gerarchia non cambia.**
2. **Istanti in secondi.** `t_i = (indice del frame nel video − indice d'inizio della clip) / fps`. Gli indici dei frame selezionati sono già salvati dalla pipeline; il tubelet prende `τ_k = (t_{2k} + t_{2k+1}) / 2`.
3. **Angolo per secondo fisso.**
   - La coordinata temporale della RoPE è `25 · tempo`: un secondo vale 25 unità, come frame a 25 fps.
   - Nel sotto-spazio temporale di una testa (20 dimensioni delle 64; le spaziali hanno 20 e 24), la frequenza c-esima è `ω_c = 10.000^(−2c/20)` e l'angolo è `25 · tempo · ω_c` **[Nostra scelta]**.
4. **P ricampiona.** Ingresso a `τ_k` irregolari, query latenti a `u_j` uniformi. È l'asimmetria di VFRTok applicata fra encoder e latente.
5. **R rende a istanti arbitrari.** Le query d'uscita stanno agli istanti `s_m` di una frequenza `f` scelta a caso in addestramento, e i bersagli sono i frame veri a quegli istanti. È l'asimmetria di VFRTok fra latente e decoder.
6. **G** genera a tempo uniforme, e riceve solo la durata `T`.

### 6.4 Una differenza voluta
- **Il principio di VFRTok.** Assume che l'informazione cresca con la durata, e allunga il latente di conseguenza.
- **La nostra scelta.** Noi teniamo **16 passi latenti per ogni clip**, coerenti con i 64 frame della gerarchia.
- **La conseguenza.** Nelle frasi lunghe la risoluzione temporale scende: con 12 s, ogni passo latente copre 0,75 s. È lo stesso limite della gerarchia.
- **[Aperto]** Va misurato sulla distribuzione delle durate. Se pesa, la variante è un numero di passi latenti proporzionale alla durata, come in VFRTok.

### 6.5 Bersagli e annotazioni agli istanti d'uscita
- I frame bersaglio di R si leggono dal video originale a 256² agli istanti `s_m`.
- I riquadri di mani e volto, usati solo per pesare la perdita (§7.1), esistono solo sui frame selezionati. Agli istanti `s_m` si interpolano linearmente dai frame selezionati più vicini **[Nostra scelta]**.

---

## 7. Perdite

### 7.1 Autoencoder (P, R, testa REPA, discriminatore)

```
L_AE = L1_w(x̂, x) + λ_P · LPIPS(x̂, x) + λ_G · g · L_adv(x̂) + λ_R · L_REPA
```

- **`L1_w` pesato:**
  - errore assoluto per pixel, con peso `1 + w_h` dentro i riquadri delle mani e `1 + w_f` dentro quello del volto **[Nostra proposta]**;
  - le mani portano il contenuto del segnato, e i generatori generici le deformano [R10];
  - è solo una pesatura della perdita, non un condizionamento.
- **`LPIPS`.** Perdita percettiva fotogramma per fotogramma, con VGG-16 congelata.
- **`L_adv` e `g`.** La perdita avversaria del generatore (hinge, §5.6) e il suo peso adattivo alla VQGAN.
- **`L_REPA`.** Vedi §5.5.
- **Pesi** `λ_P`, `λ_G`, `λ_R`, `w_h`, `w_f`, `λ_g`: **[Aperto]**. VideoRAE non li riporta, e vanno tarati nelle prime prove.

### 7.2 Generatore G: flow matching

Formulazione rettificata, come Stable Diffusion 3 [R6]:

```
x_t = (1 − t) · z̃ + t · ε,      ε ~ N(0, I),  t ∈ [0, 1]
v   = ε − z̃
L_G = E ‖ v_θ(x_t, t, condizioni) − v ‖²      sui 1.024 token bersaglio
```

- **Campionamento di `t`.** Logit-normale, come SD3 [R6].
- **Spostamento dello schedule** (RAE [R1]): `t' = α t / (1 + (α − 1) t)`, con α = √(m / 4.096).
  - Da noi m = 16 · 16 · 16 · 64 = 262.144, quindi **α = 8**.
  - Sui token grezzi dell'encoder sarebbe stato circa 45: è un'altra ragione per generare nello spazio compresso.

### 7.3 Durata

NLL gaussiana sul logaritmo della durata (§5.8).

---

## 8. Addestramento

| Stadio | Cosa si addestra | Dati | Congelato |
|---|---|---|---|
| **A0** | la gerarchia, nel piano attuale | — | — |
| **A1** | P, R, testa REPA, discriminatore | **tutte le clip, senza testo**: OpenASL (98.417 coppie, 288 h) per la fattibilità, poi il corpus (circa 6.650 h, circa 4 M clip) | encoder + LoRA |
| **Test d'ingresso** | nessuno: si **ricostruiscono** clip vere di validazione | validazione | tutto |
| **B1** | nessuno: precalcolo di `z` e dei token di testo | tutte le clip | tutto |
| **B2** | G e regressore di durata | coppie testo–video | encoder, P, R, testo |
| **B3**, consigliato | adattamento al segnante fisso: R e il percorso del prefisso in G, rifiniti brevemente | registrazioni del segnante fisso, con consenso | il resto |

**Perché B3.** Se il segnante fisso non compare nei dati, la fedeltà del suo aspetto dipende solo dalla copia dal prefisso.

**Il test d'ingresso decide** (§10.1): se le mani non sopravvivono alla ricostruzione, non si passa a B.

**Ottimizzazione.** I valori sono di partenza, da tarare **[Aperto]**:
- AdamW in bf16, con warm-up lineare;
- G: learning rate 10⁻⁴ e una copia EMA dei pesi per l'inferenza, come DiT [R8];
- batch e passi: per confronto, VideoRAE ha addestrato il suo generatore da 2 B con batch 256 per 800 k passi su 1,09 M clip [R3]. I nostri si fissano in base alla convergenza su OpenASL.
- Le run vanno su Condor.

**Memoria per il precalcolo (B1)** [Nostro calcolo]:

| Dato | Per clip | OpenASL | Corpus |
|---|---|---|---|
| `z` | 4.096 × 64 in fp16 = 0,5 MB | circa 50 GB | circa 2 TB |
| Token del testo | circa 24 × 768 in fp16 ≈ 37 KB | circa 3,6 GB | circa 150 GB |

---

## 9. Inferenza

1. **Testo.** Il testo, con il prefisso della gerarchia, passa per EmbeddingGemma: si ottengono `e_tok` (L × 768) e il vettore medio. Dal vettore medio, centratura per lingua e testa testuale danno `ẽ`.
2. **Durata.** `T̂` dal regressore, oppure scelta a mano.
3. **Riferimento.** Il prefisso del segnante fisso, calcolato una volta e conservato.
4. **Generazione.** G parte da rumore e fa **30 passi di Euler** sullo schedule spostato, da `t = 1` a `t = 0`. A ogni passo la velocità combina le condizioni con due scale di guida, come InstructPix2Pix [R11]:

   ```
   v = v(∅, ∅) + s_r · [v(∅, rif) − v(∅, ∅)] + s_c · [v(testo, rif) − v(∅, rif)]
   ```

   Le scale `s_r` e `s_c` sono **[Aperto]**.
5. **Uscita di G.** `ẑ` si de-normalizza: 16 × 16 × 16 × 64.
6. **Rendering.** R rende a **25 fps** su [0, T̂], a finestre di 32 frame con 8 di sovrapposizione: un video a 256².

---

## 10. Valutazione

### 10.1 Test d'ingresso (dopo A1)
- **Su clip vere di validazione ricostruite da P e R, si misurano:**
  - **mani:** keypoint ristimati con RTMW sul ricostruito e sull'originale, con l'errore espresso in larghezze di spalle;
  - **pixel:** PSNR, LPIPS, rFVD;
  - **contenuto:** R@1 della gerarchia sulle clip ricostruite, contro quello sulle originali.
- **Soglie [Aperto].** Si fissano prima di guardare i risultati, come per il gate (`worldsign-progetto.md` §4.12.2).

### 10.2 Generazione (dopo B2)

| Livello | Misura |
|---|---|
| **Semantico** | Il video generato passa per il campionamento e la gerarchia. Si misurano `E_sem` contro la frase di partenza e la posizione della frase fra tutte le didascalie: una retro-traduzione come retrieval, nelle due direzioni |
| **Movimento** | Sulle frasi di test, posa ristimata con RTMW sul video generato contro quella della clip vera (DTW-MJE). Solo metrica: la posa non entra nel modello |
| **Pixel e tempo** | FVD contro clip vere; coerenza temporale; errore sulla durata generata |
| **Identità** | Somiglianza del volto generato con quello del segnante fisso, con un modello di riconoscimento facciale **[Nostra proposta]** |
| **Umano** | Comprensione e naturalezza giudicate da segnanti sordi |

**Cautela.** La gerarchia non può essere l'unico giudice di ciò che il ramo genera: le misure indipendenti (RTMW, FVD, persone) sono obbligatorie.

---

## 11. Parametri

**Conto per blocco** [Nostro calcolo]:
- P e R: attenzione 4 · 384² più FFN GEGLU 12 · 384², circa **2,37 M**;
- G: auto-attenzione 0,59 M, cross-attention 0,89 M, MLP 1,18 M, vettore adaLN, circa **2,66 M**.

| Modulo | Composizione | Parametri |
|---|---|---|
| **Proiettore P** | proiezione d'ingresso 1.024 → 384 (0,39 M) + 8 blocchi (18,93 M) + uscita 384 → 64 (0,02 M) | **19,3 M** |
| **Decoder R** | ingresso 64 → 384 (0,02 M) + embedding di `f` (0,10 M) + 12 blocchi (28,39 M) + deconvoluzione 384 → 3 × 4 × 16 × 16 (1,18 M) | **29,7 M** |
| **Generatore G** | 12 blocchi (31,95 M) + componenti globali (1,53 M: embedding del tempo, proiezione di `ẽ` 256 → 384, embedding di durata e lingua, MLP di adaLN-single, ingresso e uscita, segmento, token nulli) | **33,5 M** |
| **Regressore di durata** | MLP 257 → 256 → 2 | **0,07 M** |
| **Totale usato all'inferenza** | | **≈ 82,5 M** |
| Testa REPA (solo addestramento) | MLP 384 → 768 → 1.024 | 1,1 M |
| Discriminatore (solo addestramento) | PatchGAN 3D | ≈ 5,1 M |
| **Totale in addestramento** | | **≈ 88,9 M** |

**Congelati, riusati:**
- encoder V-JEPA 2.1-L, circa 300 M, più la LoRA della gerarchia;
- EmbeddingGemma, 308 M;
- testa testuale, 0,66 M;
- VGG-16 di LPIPS, solo in addestramento.

**Confronto con VideoRAE [R3]:**

| | VideoRAE | Noi |
|---|---|---|
| Proiettore | «small»: 512, 8 strati, ≈ 34 M | 384, 8 strati, 19,3 M |
| Decoder | «base»: 768, 12 strati, ≈ 113 M | 384, 12 strati, 29,7 M |
| Generatore | 2 B | 33,5 M |
| Dati del generatore | 1,09 M clip, circa 3.800 h | OpenASL 98 k coppie (288 h); corpus circa 4 M (circa 6.650 h) |

**Lunghezze delle sequenze:**

| Modulo | Posizioni |
|---|---|
| P | 12.288 |
| R | 6.144 per finestra |
| G | 1.280, più L token di testo in cross-attention |

---

## 12. Rischi e punti aperti

| Punto | Stato |
|---|---|
| **Fedeltà delle mani** dopo la ricostruzione | il rischio principale; lo misura il test d'ingresso (§10.1). VideoRAE ha ricostruito bene UCF-101 (rFVD 13, PSNR 29,4), ma senza segnato |
| **Decoder piccolo** (29,7 M contro i circa 113 M di VideoRAE) | se il test d'ingresso fallisce, la prima leva è R: più strati, poi più teste da 64 |
| **Generatore piccolo** (33,5 M) | taglia da prova di fattibilità; se non basta, si cresce prima in profondità |
| **Risoluzione temporale delle frasi lunghe** (§6.4) | **[Aperto]**: da misurare sulle durate |
| **Segnante fisso fuori dai dati** | B3 consigliato |
| **Lingua dei segni** sul corpus multilingue | condizione `ℓ` (§5.7); con solo OpenASL è costante |
| **Iperparametri** (pesi delle perdite, σ, scale di guida, ottimizzazione) | **[Aperto]**, da tarare nelle prime prove |
| **Identità e consenso** | il segnante fisso deve aver dato il consenso all'uso della sua immagine |

---

## 13. Riferimenti

| | Lavoro | Link |
|---|---|---|
| R1 | Zheng et al., *Diffusion Transformers with Representation Autoencoders* (RAE) | https://arxiv.org/abs/2510.11690 |
| R2 | *Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders* (Scale-RAE) | https://arxiv.org/abs/2601.16208 |
| R3 | *VideoRAE: Taming Video Foundation Models for Generative Modeling via Representation Autoencoders* | https://arxiv.org/abs/2607.14088 — codice: https://github.com/zhxie0117/VideoRAE |
| R4 | *VFRTok: Variable Frame Rates Video Tokenizer with Duration-Proportional Information Assumption* (NeurIPS 2025) | https://arxiv.org/abs/2505.12053 |
| R5 | Chen et al., *PixArt-α* | https://arxiv.org/abs/2310.00426 |
| R6 | Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (Stable Diffusion 3) | https://arxiv.org/abs/2403.03206 |
| R7 | NVIDIA, *Cosmos World Foundation Model Platform* | https://arxiv.org/abs/2501.03575 |
| R8 | Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT) | https://arxiv.org/abs/2212.09748 |
| R9 | Esser et al., *Taming Transformers for High-Resolution Image Synthesis* (VQGAN) | https://arxiv.org/abs/2012.09841 |
| R10 | *Pose-Guided Fine-Grained Sign Language Video Generation* (ECCV 2024) | https://arxiv.org/abs/2409.16709 |
| R11 | Brooks et al., *InstructPix2Pix* | https://arxiv.org/abs/2211.09800 |
| R12 | Yu et al., *Representation Alignment for Generation* (REPA) | https://arxiv.org/abs/2410.06940 |
| R13 | Ramesh et al., *Hierarchical Text-Conditional Image Generation with CLIP Latents* (unCLIP) | https://arxiv.org/abs/2204.06125 |
