# WorldSign — Gerarchia e aggiornamento dei pesi

Come i tre livelli di WorldSign (posa, fisico, semantico) sono collegati, quale loss aggiorna quali pesi, in che ordine entrano nell'addestramento e come si sorveglia ogni componente prima e dopo il suo ingresso.

Documenti collegati:
- `worldsign-posa.md`: il livello 0, l'encoder di posa;
- `worldsign-architettura.md` e `worldsign-loss.md`: i componenti e i termini;
- `worldsign-ablation.md`: le ablation che cambiano la gerarchia (globale, ESP-2, ESP-6, ESP-8).

Stato: 6/10/2026. Decisioni della revisione del 2–3/10 **[Nostra scelta]**; stadi, warm-up della posa e SIGReg al livello fisico rivisti il 6/10 sul confronto con LeJEPA e LeWorldModel (§4.1).

---

## 1. Decisioni

| Punto | Decisione |
|---|---|
| Struttura | **H-JEPA addestrato per livello**: il livello alto legge la rappresentazione del livello basso, senza modificarla |
| Livello semantico | legge `sg(Enc_θ(x))`, l'uscita dell'ultimo blocco dell'encoder video sulla clip intera; **nessun condizionamento** sulla predizione fisica |
| Livello fisico | unico a modificare l'encoder video (LoRA); bersaglio `sg(s)` |
| Lettura fisica | per passo: le medie dei 4 riquadri concatenate, poi Linear(1536 → 192) |
| Calendario | 15 epoche; stadi **P** (epoche 1–2), **F₀** (epoca 3), **F** (dall'epoca 4) |
| Learning rate | warm-up di **2 epoche** per ogni gruppo dalla sua entrata; posa: warm-up di 2 epoche (lo stadio P), poi coseno |
| SIGReg al livello fisico | **nessuna**: agisce solo attraverso il bersaglio `s` (§4.1) |
| Early stopping | pazienza **3** epoche, contata solo nello stadio F |
| Monitoraggio | ogni componente letto prima e dopo il suo ingresso; una fermata a ogni confine di stadio |

---

## 2. I livelli

**La fonte.** LeCun, *A Path Towards Autonomous Machine Intelligence*, §4.6 (H-JEPA): *«JEPA-2 takes the representations extracted by JEPA-1 as inputs […] Training can be performed level-wise or globally, using any non-contrastive method for JEPA»* [Lett. 27]. La frase non dice come; la traduzione operativa qui sotto è nostra.

**Notazione:**
- x è la clip intera (32 × 16 × 16 = 8.192 token); (C, T) sono i token visibili e nascosti di una maschera;
- `Enc_θ` è V-JEPA 2.1-L con la LoRA; θ sono LoRA e norme dell'encoder;
- η è la fusione, φ la LoRA del predictor fisico, ρ la lettura;
- ψ è il predictor semantico, χ la testa testuale, ω il modello di posa.

```
Livello 0 (posa)       s  = G_ω(p)                                                  min_ω          L_0
Livello 1 (fisico)     ŝ  = R_ρ( P_φ( F_η( Enc_θ^{6,12,18,24}(x_C) ), C, T ) )      min_{θ,η,φ,ρ}  E_fis(ŝ, sg(s))
Livello 2 (semantico)  h  = sg( Enc_θ^{24}(x) )      ŷ = Q_ψ(h)                     min_{ψ,χ}      L_2(ŷ, ẽ_χ)
```

**I due livelli video non ricevono lo stesso tensore, e non serve.**
- La maschera è il modo in cui il livello 1 costruisce il suo compito: contesto = blocchi visibili, bersaglio = la posa della clip.
- Al livello 2 passa la **stessa funzione** `Enc_θ`, con gli stessi parametri, applicata all'osservazione intera.
- È lo schema di VL-JEPA [Lett. 34]: V-JEPA 2, addestrato con le maschere, letto sul video intero da un predictor che non lo modifica.
- In HWM e Hi-LeWM il compito del livello basso (predire il passo successivo) non richiede maschere, quindi i due livelli leggono lo stesso latente (https://arxiv.org/abs/2604.03208, https://arxiv.org/abs/2607.12547).

**La fusione appartiene al predictor fisico,** di cui sostituisce `predictor_embed`: non fa parte di ciò che il livello 1 «estrae». Per questo il livello 2 legge l'encoder, non la fusione.

---

## 3. Flusso del gradiente

In pratica la loss è **una sola**, `L = L_0 + E_fis + L_2`, con un solo backward e un solo passo di AdamW. I due stop-gradient rendono il gradiente diagonale a blocchi:

| Parametri | Gradiente da | Backward nel ViT-L |
|---|---|---|
| ω (posa) | solo `L_0` | — |
| θ, η, φ, ρ (encoder e livello fisico) | solo `E_fis` | sì, nel passaggio fisico |
| ψ, χ (livello semantico) | solo `L_2` | **no**: il passaggio semantico fa il forward dell'encoder senza gradiente |

**I pesi fra i livelli non contano.** Con Adam, moltiplicare per una costante la loss che un blocco di parametri riceve lascia l'aggiornamento invariato, a meno di ε. Contano solo i rapporti **dentro** un livello:
- in `L_0`: invarianza, ancora e SIGReg sulla posa;
- in `L_2`: `E_sem` e SIGReg semantico.

La composizione resta quella del progetto, `(1 − λ)·(termini predittivi) + λ·(SIGReg)`, livello per livello: λ = 0,04 sulla posa (`worldsign-posa.md` §4.2), λ = 0,05 sul semantico.

**Verifica prima di lanciare (P17).** Su un batch piccolo:
- `E_fis` non deve dare gradiente al modello di posa;
- `L_2` non deve darne alla LoRA video, nello schema per livello;
- i termini della posa non devono darne al ramo video.

Se una di queste condizioni fallisce, la run non parte.

**Inferenza:** video → `Enc_θ` → predictor semantico → ŷ; testo → EmbeddingGemma → testa → ẽ; retrieval = argmin `E_sem`. Il modello di posa, la fusione, il predictor fisico e la lettura servono solo in addestramento.

---

## 4. Il livello fisico per passo

**Lettura.** Il predictor fisico predice un token in ogni posizione (`predict_all`). Al passo t:

```
m_{t,a} = media dei token predetti (visibili e nascosti) nel riquadro dell'articolatore a     (0 se il riquadro è vuoto)
ŝ_t     = Linear(4·384 → 192)( [ m_{t,corpo}, m_{t,sx}, m_{t,dx}, m_{t,volto} ] )
```

Rispecchia il lato posa: una media per articolatore, poi la concatenazione.

**Energia.** Nella media del riquadro entrano sia i token nascosti sia quelli visibili. Due letture separate (solo nascosti, solo visibili) non funzionerebbero più: un riquadro senza token nascosti darebbe una parte vuota, mentre il bersaglio contiene comunque quell'articolatore, quindi resterebbe un errore irriducibile. La pesatura per token di V-JEPA 2.1 [Lett. 32] si porta allora al passo:

```
n^m_{b,t} = Σ_a |riquadro a ∩ nascosti al passo t|          w^v_{b,t} = Σ_a Σ_{visibili nel riquadro} peso del token
ω_{b,t}   = c_{b,t} · ( n^m_{b,t} + λ_ctx · w^v_{b,t} )       λ_ctx = 0,5;  peso del token visibile = 1 (o 1/√d)
c_{b,t}   = media della presenza dei 69 giunti al passo t

E_fis^k = Σ_{b,t} ω_{b,t} · (1/192)‖ ŝ^k_{b,t} − LN(sg(s_{b,t})) ‖₁  /  Σ_{b,t} ω_{b,t}
E_fis   = media sui 2 tipi di maschera k
```

Ogni token nascosto dentro i riquadri conta 1, ogni token visibile λ_ctx = 0,5, come in V-JEPA 2.1. Per la plausibilità e le energie per clip, la stessa somma si fa clip per clip.

### 4.1 SIGReg al livello fisico [revisione del 6/10]

**Al livello fisico non c'è SIGReg, come nei due paper.**
- LeJEPA e LeWorldModel la applicano alla rappresentazione appresa che definisce lo stato, quella che può collassare: lo `z` di LeWM, che per noi è `s` (livello 0, per passo).
- LeWM non regolarizza la predizione (`pred_emb`), e neanche noi `ŝ`.
- I token dell'encoder video non ne hanno bisogno. Il bersaglio è fisso e non collassato, quindi il collasso non conviene. SIGReg cancellerebbe la geometria pre-addestrata di V-JEPA 2.1. E gli 8.192 token di una clip sono campioni ancora più correlati dei 32 passi.
- Il collasso dimensionale dell'encoder si sorveglia (deriva dell'encoder, rango, R@1), non si regolarizza.

**Come agisce attraverso il bersaglio.**
- **Varianza uguale.** `s` ha la stessa varianza nelle 192 dimensioni, quindi nessuna domina la L1 di `E_fis`.
- **Scala già standardizzata.** `LN(sg(s))` è quasi una no-op: toglie circa 2 gradi di libertà su 192, la media sui canali e la norma, che per una gaussiana in 192 dimensioni oscilla del ±5 %. La si tiene come protezione della scala all'ingresso di F₀.
- **Base non fissata.** SIGReg, `L_inv` e l'ancora sono invarianti per rotazione di `s`: il decoder dell'ancora può ruotare insieme. Niente fissa la base del bersaglio, che può derivare mentre la lettura fisica lo insegue.
  - La CKA con il passo 0 è cieca a questa deriva. La misura invece `s_rotation` (`worldsign-posa.md` §6): sul batch sonda, la correlazione canale per canale di `s` con la lettura precedente, prima e dopo la rotazione migliore (Procrustes, stimata su metà delle clip e letta sull'altra metà).
  - L'allarme scatta oltre 0,1. Se la rotazione risultasse grande, il rimedio di riserva è congelare il decoder dell'ancora dopo lo stadio P, che fissa la base nelle 138 direzioni che decodifica.
- Per questo il warm-up della posa coincide con lo stadio P (§6.2): il bersaglio rallenta già quando entra il livello fisico, come il momentum dell'EMA che cresce in V-JEPA e BYOL.

**Scelte del 6/10.**
- **Forma della loss.** Resta quella di V-JEPA 2.1: L1 su `LN(sg(s))`. La MSE senza LayerNorm di LeWM è un'ablation facoltativa (F8, `worldsign-ablation.md` §3.2).
- **Nessuna maschera «futuro».** Le maschere restano tubi su tutti i 32 passi, come V-JEPA: il livello fisico ricostruisce lo stato nascosto anche dai passi futuri e non prevede in avanti, come farebbe invece un world model alla LeWM. Una maschera che nasconda gli ultimi passi è stata valutata e scartata.
- **Dropout 0,1 nel predictor fisico**, come il predictor di LeWM, dove porta il successo da 78 a 96 %. Si alzano i moduli `nn.Dropout` dei blocchi rilasciati, senza toccare la LoRA; il dropout sulle probabilità dell'attenzione resta a 0 (`worldsign-architettura.md` §4.3).

---

## 5. Il livello semantico

- **Input:** `h = sg(Enc_θ^{24}(x))`, l'uscita normalizzata dell'ultimo blocco sulla clip intera, come VL-JEPA.
- **Forward senza gradiente:** il passaggio semantico non conserva le attivazioni del ViT-L.
- **Cosa addestra la predizione video → testo:** il predictor semantico (7,67 M) e la testa testuale. L'encoder cambia solo per opera della fisica: è la scommessa di H3.
- **Configurazione:** `semantic.trains_encoder: false`. Con `true` si ha l'ablation «globale» (§8).

---

## 6. Calendario

### 6.1 Stadi

Le epoche sono **15**. Gli stadi seguono i confini di epoca.

| Stadio | Epoche | Passaggi | Famiglie addestrate |
|---|---|---|---|
| **P** | 1–2 | posa, semantico | `pose`, `semantic_new` |
| **F₀** | 3 | + fisico | in più `physical_new` (fusione e lettura); LoRA dell'encoder e del predictor ferme |
| **F** | 4 → fine | tutti | in più `physical_lora` e `video_lora` |

Senza livello fisico (ESP-2) c'è un solo stadio, **S**: solo il semantico, e la LoRA non si addestra mai.

**Perché questi stadi.**
- **P:** la LoRA non deve inseguire un bersaglio di posa ancora casuale. Il semantico parte subito, su V-JEPA 2.1 pre-addestrato.
- **P dura 2 epoche [revisione del 6/10]:** coincide con il warm-up della posa. Il suo learning rate arriva al picco a fine P e da lì scende: il bersaglio rallenta quando il livello fisico comincia a inseguirlo (§4.1). Prima il warm-up durava 3 epoche e il bersaglio accelerava ancora durante F₀ e la prima epoca di F. Un controllo della configurazione impedisce un warm-up più lungo di P.
- **F₀:** è il «prima la testa» di LP-FT [Lett. 95]: le teste fisiche nuove imparano su un encoder fermo, così non ne distorcono le feature quando la LoRA si sblocca.
- **Precedenti per gli stadi di un'epoca:**
  - ULMFiT sblocca un gruppo di strati per epoca (https://arxiv.org/abs/1801.06146);
  - LLaVA-1.5 addestra per 1 epoca il solo proiettore, poi tutto (https://arxiv.org/abs/2310.03744);
  - MC-JEPA introduce il secondo obiettivo dopo il 10 % della run (https://arxiv.org/abs/2307.12698).

### 6.2 Learning rate

Ogni famiglia ha il suo gruppo di AdamW (con e senza weight decay):

```
fattore(gruppo, passo) = schedule della famiglia(passo) × cooldown(passo)

famiglie tranne la posa:  0 prima della loro entrata; warm-up lineare di 2 epoche dall'entrata; poi costante
posa:                     warm-up lineare di 2 epoche (lo stadio P); poi coseno fino a 0 alla fine pianificata
cooldown (V-JEPA 2):      1 fino all'inizio del cooldown; poi lineare fino a 0 in 5 % dei passi della run
```

| Famiglia | Entra | Picco | Warm-up |
|---|---|---|---|
| `pose` | passo 0 | 3e-4 | 2 epoche (lo stadio P), poi coseno |
| `semantic_new` | passo 0 | LR base (2e-4) | 2 epoche |
| `physical_new` | inizio di F₀ | LR base | 2 epoche |
| `physical_lora`, `video_lora` | inizio di F | LR base | 2 epoche |

Un warm-up per ogni gruppo che entra è la pratica di LLaVA (warm-up a ogni stadio) e di SigLIP 2 (stato dell'ottimizzatore nuovo per i parametri aggiunti, https://arxiv.org/abs/2502.14786).

### 6.3 Early stopping e cooldown

- **Validazione:** a ogni fine epoca (e alle cadenze di §4.13 del progetto).
- **Il miglior checkpoint e la pazienza** (3 epoche senza miglioramento della metrica decisionale) **contano solo nello stadio F.** Durante P e F₀ la R@1 misura il semantico su un encoder non ancora adattato.
- **Cooldown:** alla fine pianificata o all'early stopping la run riparte dal miglior checkpoint e scende linearmente a 0, come V-JEPA 2.

---

## 7. Monitoraggio per componente

Ogni componente si legge **prima** di entrare (le letture alla fine dello stadio precedente) e **dopo** (le letture alla fine del suo primo stadio). Con le diagnostiche a fine epoca i due momenti coincidono con i confini di stadio.

| Componente | Entra | Prima | Dopo |
|---|---|---|---|
| Modello di posa | P | passo 0: riferimento di rango, deviazione, SIGReg di s | fine P: fermata F1 |
| Predictor semantico, testa testuale | P | passo 0: R@1 a caso, γ_sem | fine P: fermata F1 |
| Fusione, lettura fisica | F₀ | fine P: `E_fis` con le teste casuali (letto, non addestrato) | fine F₀: fermata F2 |
| LoRA video e del predictor | F | fine F₀: R@1, drift dell'encoder = 0, rapporto LoRA = 0 | fine dell'epoca 4: fermata F3 |

**Letture fisse a ogni cadenza:**
- i termini di ogni livello e il learning rate di ogni famiglia;
- quote e coseni del gradiente per livello: posa sull'encoder di posa; fisico sulla LoRA video;
- collasso di s e di ŷ;
- letture del livello fisico per passo: R² complessivo, per copertura della maschera, sui passi soprattutto visibili e nascosti; la dinamica contro la baseline; gli errori in keypoint.

**Fermate programmate** (§4.13.5 del progetto, riancorate agli stadi):

| Fermata | Quando | Criteri |
|---|---|---|
| **F1** | fine di P (epoca 2) | **posa:** ancora e SIGReg_posa in calo · rango di s > 0,5 × passo 0 · IsoScore di s ≥ 0,8 · R² di posizione delle mani da s ≥ 0,9. **Semantico:** γ_sem > 0,3 · SIGReg_sem in calo · R@1 > 5× il caso · test col rumore superato · query non collassate |
| **F2** | fine di F₀ (epoca 3) | `E_fis` in calo · R² dei passi soprattutto visibili > 0,9 · la dinamica batte la baseline · nessuna fuga (test del leak) |
| **F3** | fine della prima epoca di F (epoca 4) | `E_fis` in calo · nessuna LoRA sposta il suo strato oltre il 10 % · drift dell'encoder ≥ 0,5 · R@1 > baseline ridge · ω < 0,95 · hubness stabile · curva di R@1 estrapolata compatibile con X · nessun conflitto stabile fra gradienti |
| **F4** | fine | tabella del gate su OpenASL |

Nella run di gate una fermata fallita ferma la run; nelle altre run si registra e basta.

---

## 8. Ablation collegate

| Ablation | Cosa cambia | Domanda |
|---|---|---|
| **Globale** | `semantic.trains_encoder: true`: `E_sem` raggiunge la LoRA video (lo schema prima della revisione) | il semantico deve plasmare l'encoder? |
| **ESP-2** | nessun livello fisico: la LoRA non si addestra, il semantico legge V-JEPA 2.1 congelato (lo schema di VL-JEPA) | la fisica migliora la semantica? (H3) |
| **ESP-6** | livello 0 = target encoder di V-JEPA 2.1 invece della posa | serve la posa? La lettura per token va formalizzata a parte |
| **ESP-8** | VICReg al posto di SIGReg | rimandata |

---

## 9. Codice

| Modulo | Contenuto |
|---|---|
| `signworld/experiment/train/curriculum.py` | famiglie, stadi P/F₀/F (e S), quando entra ogni famiglia |
| `signworld/experiment/train/schedules.py` | schedule per gruppo, cooldown, gruppi di AdamW |
| `signworld/experiment/train/trainer.py` | il passo, gli stadi, early stopping nello stadio F, fermate ai confini |
| `signworld/models/worldsign/model.py` | i tre livelli e i loro stop-gradient |
| `signworld/models/worldsign/video_branch.py` | il passaggio semantico senza gradiente nell'encoder |
| `signworld/models/worldsign/readout.py`, `physical.py` | la lettura per passo |
| `signworld/loss/worldsign.py` | `E_fis` per passo |
| `signworld/experiment/train/monitor.py`, `preflight.py` | letture, fermate, P17 |

Configurazione: `training.stages`, `training.activation_warmup_epochs`, `training.epochs`, `training.patience`, `semantic.trains_encoder` in `parameters/model/worldsign.yaml`.
