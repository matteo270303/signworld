# Un world model a energia per il retrieval multilingua della lingua dei segni

**Documento di progetto** · proposta pre-implementazione · settembre 2026 · piano sperimentale rivisto il 29/9 per CVPR 2027 (§4.14) · metodologia della posa e della gerarchia rivista il 3/10

> **Come leggere le etichette**
> - **[Lett. n]** — affermazione sostenuta dal riferimento *n* (elenco in fondo al documento).
> - **[Nostra ipotesi]** — congettura nostra, non ancora verificata; indichiamo come verrà testata.
> - **[Nostra scelta]** — decisione di progetto motivata, ma non derivata direttamente dalla letteratura.
> - **[Nostra argomentazione]** — ragionamento nostro a sostegno di una scelta.
> - **[Nostra stima]** — valore quantitativo approssimato, da ricalibrare sui dati reali.
> - **[Aperto]** — punto non ancora deciso; indichiamo l'esperimento preliminare che lo chiuderà.
>
> Teoremi matematici standard e nozioni generali di deep learning compaiono senza etichetta.

> **Revisione del 3/10/2026.** L'encoder di posa **non è pre-addestrato**: si addestra da zero insieme al resto del modello, con invarianza fra viste, SIGReg e ancora (`worldsign-posa.md`). La gerarchia è un **H-JEPA addestrato per livello**: il livello fisico legge la posa con lo stop-gradient ed è l'unico ad adattare l'encoder video; il semantico legge l'encoder senza modificarlo. Gli stadi sono P, F₀ e F, a confini di epoca (`worldsign-gerarchia.md`). Le sezioni che descrivevano l'S-JEPA pre-addestrato, il bersaglio per articolatore e gli stadi 1a/1/2a/2 sono aggiornate; in caso di dubbio valgono i due documenti.
>
> **Revisione del 4/10/2026.** Il retrieval si misura e si giudica **in entrambe le direzioni** (T2V e V2T). Tutte le metriche si riportano nei due versi: R@k, Precision@k, Recall@k, MRR, MedR, R@1 tollerante, ripartizioni, intervalli bootstrap, test col rumore, hubness delle due gallerie. Le baseline di PC2 e il gate F4 hanno una soglia per direzione. F3 confronta la media con la media, e `X` passa da 46,7 a 46,5 (§4.12.2). La metrica che decide resta la media di R@1 T2V e V2T (§4.10).

---

## 1. Introduzione

### 1.1 Contesto

Le lingue dei segni sono lingue naturali che usano il canale visivo-gestuale. Hanno una fonologia (le unità minime che compongono la forma dei segni), un lessico e una grammatica propri. Non esiste una lingua dei segni universale: ogni comunità sorda ha la sua — ASL negli Stati Uniti, BSL nel Regno Unito, DGS in Germania, CSL in Cina, LIS in Italia — e lingue dei segni diverse, in generale, non sono mutuamente comprensibili, anche quando nei rispettivi paesi si parla la stessa lingua (ASL e BSL sono l'esempio classico).

In ambito computazionale i compiti principali sono:

| Compito | Cosa fa |
|---|---|
| **CSLR** — riconoscimento continuo | trascrive il segnato continuo in una sequenza di *gloss*, cioè etichette scritte convenzionali per ogni segno |
| **SLT** — traduzione | dal video segnato al testo nella lingua parlata |
| **SLP** — produzione | dal testo a segnato generato (avatar o video) |
| **Retrieval** | dato un testo, trova i video che lo esprimono (text-to-video, T2V), o viceversa (V2T) |

Le gloss richiedono annotazione esperta e sono costose. Per questo la ricerca recente preferisce approcci *gloss-free*, che imparano direttamente da coppie video–testo.

### 1.2 Obiettivo

Costruire un **modello fondazionale per la lingua dei segni continua e multilingua** con:

- il **retrieval testo↔video** come compito primario, ottimizzato direttamente durante l'addestramento;
- **traduzione, riconoscimento e produzione** come compiti secondari, da ottenere in un secondo momento con *attentive probing* o fine-tuning leggero **[Nostra scelta]**.

Vogliamo una rappresentazione con tre proprietà: (i) geometria ben condizionata, cioè utilizzabile con una semplice similarità coseno; (ii) una cognizione minima della **fisica del corpo che segna** (inerzia, continuità del moto, vincoli articolari); (iii) la **semantica** del messaggio.

### 1.3 L'idea in breve

Tre ingredienti:

1. **Predire invece di ricostruire.** Seguiamo le Joint-Embedding Predictive Architectures (JEPA) [Lett. 27, 29, 30]: il modello predice la *rappresentazione astratta* di ciò che non vede, non i pixel.
2. **Predire a due livelli di astrazione, con un solo encoder video condiviso.**
   - **Livello fisico:** dal video in gran parte mascherato, il modello predice lo **stato del corpo** (le rappresentazioni della posa) anche nelle parti nascoste. La ricetta di mascheramento e di loss è quella di V-JEPA 2.1 [Lett. 30, 32]; cambia solo il bersaglio.
   - **Livello semantico:** dal video intero, il modello predice l'**embedding della didascalia**, come in VL-JEPA [Lett. 34].
   - Il livello semantico legge l'encoder video adattato dal livello fisico, senza modificarlo: la gerarchia è un H-JEPA addestrato per livello [Lett. 27] (`worldsign-gerarchia.md`).
3. **Nessun negativo contrastivo.** L'allineamento testo–video si impara senza confronti con esempi negativi. La dispersione dello spazio (*uniformity*) che serve al retrieval la impone un regolarizzatore distribuzionale, **SIGReg** [Lett. 35], che spinge gli embedding verso una distribuzione gaussiana isotropa.

Il tutto è formulato come **energy-based model** [Lett. 28]: l'energia è l'errore di predizione; il retrieval è la ricerca dell'energia minima; una sequenza plausibile è una sequenza a energia bassa. Una **variabile latente** viene provata solo in un'ablation facoltativa (§4.5.9, §4.14).

### 1.4 Contributi attesi

- **Una verifica controllata** del fatto che, nel retrieval cross-modale, l'uniformity possa venire da un vincolo sulla distribuzione degli embedding invece che da negativi contrastivi, a parità di dati visti **[Nostra ipotesi, §5.1 H1]**.
- **Un world model a due livelli di astrazione** (fisico e semantico) per la lingua dei segni continua: tre encoder, due predictor, circa 29,8 M di parametri addestrabili **[Nostra proposta]**.
- **Una batteria di diagnostiche e di test di plausibilità basati sull'energia**, costruibili senza annotazione **[Nostra proposta]**.
- **Un'analisi della condivisione fra lingue dei segni** nelle diverse rappresentazioni del modello (ipotesi «a clessidra») **[Nostra ipotesi, §5.1 H5]**.

### 1.5 Cosa è fuori scope in questo primo lavoro

- **Allineamento lessicale** (quale porzione di video corrisponde a quale parola): è un problema aperto del settore (§4.5.5).
- **Tracciamento dello spazio segnico** (i punti dello spazio in cui il segnante colloca i referenti) come risultato rivendicato.
- **Zero-shot su lingue dei segni mai viste** (§2.7).
- **Coerenza del discorso fra enunciati** e incompatibilità semantiche dentro la frase **[Nostra scelta]**.
- **Traduzione e produzione come risultato principale**: rientrano come compiti a valle.

### 1.6 Struttura del documento

§2 motivazioni teoriche e concettuali · §3 dati · §4 architettura e metodologia · §5 aspettative, ipotesi e pericoli · §6 conclusione.

### 1.7 Glossario essenziale

| Termine | Significato |
|---|---|
| **Embedding** | vettore che rappresenta un input (video, testo) in uno spazio continuo |
| **Encoder** | rete che produce embedding da un input |
| **Predictor** | rete che, in una JEPA, predice l'embedding di un target a partire dall'embedding del contesto |
| **Target encoder** | l'encoder che produce il bersaglio della predizione (qui: posa per il livello fisico, testo per il semantico) |
| **Collasso** | soluzione degenere in cui l'encoder produce lo stesso vettore (o vettori in un sottospazio minuscolo) per ogni input |
| **Contrastivo / InfoNCE** | obiettivo che avvicina coppie corrette e allontana coppie sbagliate («negativi») nel batch |
| **Uniformity** | quanto gli embedding sono sparsi nello spazio [Lett. 40] |
| **Isotropia** | varianza uguale in tutte le direzioni dello spazio degli embedding |
| **Articolatore** | mano sinistra, mano destra, busto, volto |
| **Tubelet** | blocco spazio-temporale di pixel (qui 2 frame × 16×16 pixel) trasformato in un token |
| **Maschera multi-blocco** | insieme di grandi blocchi spaziali nascosti su tutti i frame della clip, come in V-JEPA [Lett. 30] |
| **Query** | vettori appresi che il predictor usa per sapere *cosa* predire |
| **LoRA** | adattamento a basso rango di pesi congelati [Lett. 67] (§4.4.2) |
| **MRL** | Matryoshka Representation Learning: embedding troncabili a dimensioni minori [Lett. 56]; EmbeddingGemma lo supporta, noi non tronchiamo (§4.4.4) |
| **Energia** | punteggio scalare di incompatibilità fra due input: bassa = compatibili [Lett. 28] |
| **Variabile latente** | variabile non osservata che assorbe l'incertezza di una predizione |
| **Effective rank** | numero «effettivo» di direzioni usate dagli embedding [Lett. 43] |
| **Attentive probing** | valutazione con un piccolo modulo di attenzione addestrato sopra feature congelate |

---

## 2. Motivazioni teoriche e concettuali

### 2.1 Nozioni di base per un lettore esterno al dominio

#### 2.1.1 La struttura del segnato

- **Parametri fonologici.** Un segno si descrive con configurazione della mano, luogo di articolazione, movimento e orientamento [Lett. 1], più componenti **non manuali**: espressione facciale, movimenti del busto, *mouthing* (movimenti labiali che richiamano parole della lingua parlata).
- **Coarticolazione ed epentesi.** Nel segnato continuo i confini fra segni non sono marcati. Fra un segno e il successivo ci sono movimenti di transizione (epentesi) che non appartengono a nessuno dei due.
- **Vincoli sui segni a due mani** [Lett. 2]. *Symmetry Condition*: se si muovono entrambe le mani, tendono ad avere la stessa configurazione e movimenti simmetrici. *Dominance Condition*: se si muove solo la dominante, la non dominante ha un repertorio ristretto di configurazioni. **Non sono però universali**: esistono segni che le violano e, in ASL e nella lingua dei segni svizzero-tedesca, solo circa il **67 %** delle costruzioni con classificatori a due mani rispetta la Dominance Condition [Lett. 77]. Per questo **non le usiamo come vincolo di progetto** **[Nostra scelta]**.
- **Iconicità.** Una parte del lessico ha motivazione iconica. In un'analisi di 1.944 segni della LIS, il 50 % delle configurazioni e il 67 % dei luoghi di articolazione risultano motivati iconicamente [Lett. 3]. L'iconicità però **gonfia** le misure di somiglianza lessicale fra lingue dei segni non imparentate [Lett. 4].
- **Interferenza fra lingue negli esseri umani.** Nei segnanti multilingui si osservano trasferimenti negativi proprio su mouthing, lessico e configurazione della mano [Lett. 5].

#### 2.1.2 Retrieval, bi-encoder e benchmark

Nel retrieval **a bi-encoder** video e testo si codificano separatamente, e il punteggio di una coppia è una similarità, tipicamente il coseno. Gli embedding della collezione si calcolano una volta sola, così ogni query costa un prodotto interno per candidato. Un *cross-encoder*, che processa ogni coppia congiuntamente, richiederebbe invece un passaggio della rete per ogni coppia query–candidato. **Per questo usiamo il coseno: è un vincolo di costo del compito, non una semplificazione teorica [Nostra argomentazione].**

Benchmark standard:

| Benchmark | Lingua dei segni / lingua parlata | Nota |
|---|---|---|
| **OpenASL** [Lett. 86] | ASL / inglese | benchmark principale di retrieval; notiziari e video della NAD da YouTube |
| **PHOENIX-2014T** [Lett. 12] | DGS / tedesco | dominio ristretto (meteo) |
| **CSL-Daily** [Lett. 13] | CSL / cinese | |

**OpenASL** conta 288 ore di ASL da oltre 200 segnanti e 98.417 coppie video–frase. Validazione (966) e test (975) sono coppie estratte **a caso**, quindi gli stessi video e segnanti compaiono anche in addestramento [Lett. 86]. Stato dell'arte nel retrieval: **C²RL, R@1 T2V = 62,2, V2T = 61,6**, con fine-tuning su OpenASL [Lett. 87].

How2Sign [Lett. 11], su cui SEDS riporta R@1 T2V = 62,5 [Lett. 15] e CiCo aveva migliorato di +22,4 (T2V) e +28,0 (V2T) punti di R@1 il metodo precedente [Lett. 14], **per noi non è accessibile [Nostra premessa]**.

#### 2.1.3 Apprendimento a embedding congiunti e collasso

Due encoder mappano due input correlati (per esempio un video e la sua didascalia) in vettori, e l'obiettivo li avvicina. **La soluzione banale è mappare tutto su una costante (collasso).** La letteratura distingue tre famiglie di meccanismi anti-collasso [Lett. 28]:

| Famiglia | Idea | Esempi |
|---|---|---|
| **Contrastiva** | allontana esplicitamente i negativi | InfoNCE, CLIP |
| **Asimmetrica** | un ramo target «lento» (stop-gradient, media mobile esponenziale dei pesi, EMA) rompe la simmetria | BYOL, V-JEPA |
| **Regolarizzata** | vincola la distribuzione degli embedding perché resti sparsa | VICReg, SIGReg |

La loss **InfoNCE**, su un batch di N coppie:

```
L_InfoNCE  =  − (1/N) · Σ_i  log  [  exp(⟨z_i, t_i⟩ / τ)  /  Σ_j exp(⟨z_i, t_j⟩ / τ)  ]
```

- `z_i` embedding del video *i*, `t_i` embedding del suo testo, `⟨·,·⟩` prodotto interno fra vettori normalizzati
- `τ` temperatura; il denominatore somma sulla coppia corretta e sugli **N−1 negativi** del batch

Wang & Isola [Lett. 40] mostrano che, al limite, InfoNCE ottimizza due proprietà:

```
L_align  =  E_(coppie positive)  ‖ f(x) − f(y) ‖²                  vicinanza delle coppie corrette
L_unif   =  log  E_(coppie casuali)  exp( −2 · ‖ f(x_i) − f(x_j) ‖² )   dispersione sulla sfera
```

#### 2.1.4 JEPA: predire rappresentazioni, non pixel

Una JEPA predice l'embedding di un segnale `y` a partire da un segnale compatibile `x`, con un predictor `P` condizionato da una variabile aggiuntiva `a` [Lett. 29]:

```
ŝ_y  =  P( Enc_x(x) , a )          L  =  D( ŝ_y , Enc_y(y) )

   Enc_x, Enc_y   encoder del contesto e del target (possono essere reti diverse)
   a              condizionamento: DOVE o COSA predire
   D              distanza nello spazio delle rappresentazioni
```

L'errore si calcola nello spazio delle rappresentazioni. La motivazione [Lett. 30]: predire pixel costringe a modellare dettagli irrilevanti e impredicibili (texture, rumore); predire feature concentra la capacità su ciò che è predicibile e semanticamente rilevante. **V-JEPA** estende l'idea al video, mascherando regioni spazio-temporali [Lett. 30]; **V-JEPA 2** la porta a oltre un milione di ore di video [Lett. 31]; **V-JEPA 2.1** migliora le feature dense [Lett. 32].

**La maschera è solo un modo di costruire la coppia (x, y).**
- In I-JEPA `x` e `y` vengono dalla stessa immagine: la maschera crea il bersaglio nascosto, e il condizionamento `a` sono i token di maschera con la posizione del blocco da predire, cioè **dove** predire [Lett. 29].
- **VL-JEPA** si addestra invece su terne ⟨X_V, X_Q, Y⟩: un input visivo, una query testuale e il testo bersaglio. Ha quattro componenti: un X-Encoder che comprime il video in «token visivi», un **Predictor** che mappa ⟨token visivi, query⟩ nella predizione dell'embedding del bersaglio, un Y-Encoder che *«astrae dall'informazione irrilevante per il compito»*, e un Y-Decoder usato solo quando serve testo leggibile. La query dice **cosa** predire, e **non c'è maschera**: il bersaglio è già un'altra modalità [Lett. 34].

Il nostro modello usa entrambe le forme: la prima al livello fisico, la seconda al livello semantico (§2.6).

#### 2.1.5 Energy-based models e variabili latenti

Un **EBM** definisce un'energia scalare `E(x, y)`: bassa se `x` e `y` sono compatibili, alta altrimenti. L'inferenza cerca la `y` di energia minima [Lett. 28]:

```
ŷ  =  argmin_y  E(x, y)
```

Addestrare un EBM vuol dire abbassare l'energia sui dati veri e **impedire che resti bassa ovunque** (collasso = paesaggio energetico piatto). Le due strategie [Lett. 28]:

- **contrastiva**: alzare esplicitamente l'energia su esempi negativi;
- **regolarizzata**: limitare il **volume** dello spazio che può avere energia bassa, così che abbassarla sui dati la alzi automaticamente altrove.

Quando la relazione è **uno-a-molti** (un contesto ammette più esiti validi) si introduce una **variabile latente** `z` e si usa l'**energia libera** [Lett. 28]:

```
F(x, y)    =  min_z  E(x, y, z)

F_β(x, y)  =  −(1/β) · log  ∫ exp( −β · E(x, y, z) ) dz          versione morbida; per β → ∞ torna il minimo
```

La capacità di `z` va limitata — rendendola discreta, sparsa, stocastica o a bassa dimensione — altrimenti `z` può «spiegare» qualunque `y` e l'energia diventa nulla ovunque [Lett. 28].

### 2.2 Motivazione 1 — i costi dei metodi contrastivi

- **Limite informativo.** InfoNCE è un limite inferiore sulla mutua informazione fra le due viste, e **non può superare `log N`**, con N la dimensione del batch [Lett. 38, 39]. Per stimare informazione alta servono batch molto grandi.
- **Costo concreto.** VL-JEPA, l'architettura più vicina alla nostra, fa il pretraining contrastivo con **batch da 24.000**, per **4 settimane su 24 nodi × 8 GPU H200** [Lett. 34].
- **Argomento di principio.** Secondo LeCun i metodi regolarizzati sono **meno soggetti alla maledizione della dimensionalità** di quelli contrastivi, e quindi più promettenti per gli EBM [Lett. 28].
- **Il nostro vincolo.** Lavoriamo su un cluster accademico: un batch contrastivo da decine di migliaia di esempi non è alla nostra portata **[Nostra premessa]**.

### 2.3 Motivazione 2 — l'uniformity serve comunque, e questo delimita la domanda

**Il problema esiste nel segnico.** SignCL [Lett. 17] documenta il **representation density problem**: nei modelli gloss-free le rappresentazioni di segni **semanticamente diversi** finiscono troppo vicine, con perdite di prestazioni consistenti fra metodi di estrazione diversi. La loro soluzione è un termine contrastivo.

**VL-JEPA misura il costo di togliere il termine contrastivo** [Lett. 34], su classificazione, retrieval e VQA:

| Loss | Classificazione | **Retrieval** | VQA |
|---|---|---|---|
| **InfoNCE** | 23,3 | **30,3** | 44,3 |
| Coseno | −6,8 | **−10,1** | +2,3 |
| L1 | −8,5 | **−14,8** | — |
| L2 | −9,8 | **−18,6** | — |

**La nostra lettura [Nostra argomentazione].** InfoNCE = allineamento **+** uniformity [Lett. 40]. Coseno, L1 e L2, come testati, **non hanno alcun meccanismo di dispersione**. La tabella misura quindi il costo di **togliere l'uniformity**, non la necessità dei **negativi**. La configurazione «allineamento + regolarizzatore distribuzionale» **non è stata testata**: è la domanda di questo progetto.

**Il segnale cross-modale non contrastivo funziona già per la comprensione.** Uni-Sign (pretraining generativo) [Lett. 8], Scaling SLT [Lett. 19], i baseline di YouTube-SL-25 [Lett. 6] e SONAR-SLT [Lett. 20] non usano negativi e ottengono buoni risultati in traduzione e riconoscimento. **Nessuno di questi lavori valuta però lo spazio di embedding come spazio metrico per il retrieval [Nostra ricognizione della letteratura].**

**Informativo non vuol dire usabile con il coseno.** Su Kinetics-400 con feature congelate, V-JEPA ViT-L/16 ottiene **56,7 %** con probe lineare e **80,8 %** con attentive probe; DINOv2 ViT-g/14 ottiene 78,4 % e 83,4 % [Lett. 30]. Le feature predittive contengono l'informazione, ma non in forma linearmente accessibile. **Ne ricaviamo una regola: il vettore usato per il retrieval deve essere ottimizzato direttamente dalle loss [Nostra scelta].**

### 2.4 Motivazione 3 — SIGReg: cosa garantisce e cosa no

**Cosa dice la teoria** [Lett. 35]. LeJEPA dimostra che la **gaussiana isotropa** è la distribuzione degli embedding che minimizza il rischio nel caso peggiore su compiti a valle sconosciuti: per probe lineari (le deviazioni dall'isotropia amplificano bias e varianza) e per probe non lineari come k-NN e kernel (unico minimizzatore del bias quadratico integrato).

**Come funziona SIGReg** [Lett. 35]. Proietta gli embedding su molte direzioni casuali monodimensionali e, per ciascuna, confronta con un test statistico di Epps–Pulley la funzione caratteristica empirica con quella di una normale standard. Costo lineare nel batch, **un solo iperparametro**, **nessuno stop-gradient, nessun teacher-student, nessuna EMA**. Validato su oltre 10 dataset e oltre 60 architetture; ViT-H/14 raggiunge 79 % su ImageNet-1K in linear probe.

**Limiti dichiarati dagli autori** [Lett. 35]: un bias di minibatch `O(1/N)`; resta aperto se compiti specifici traggano vantaggio da embedding **anisotropi**. La validazione è **unimodale e in-domain**.

**LeVJEPA** estende SIGReg al video con **5,6–20,8×** meno compute di V-JEPA 2 a parità di epoche, ma **elimina il predictor**: l'obiettivo diventa l'invarianza fra una vista globale e viste locali [Lett. 36].

**Quattro osservazioni formali**, che guidano il design:

1. **SIGReg non può allineare [Nostra osservazione, elementare].** Se `z ~ N(0, I)`, allora `R·z ~ N(0, I)` per ogni rotazione `R`. La distribuzione marginale non contiene informazione su *quale* video corrisponda a *quale* testo: l'allineamento deve venire tutto dai termini predittivi.
2. **Indipendenza dei blocchi di coordinate** (teorema di Cramér–Wold). Una distribuzione è determinata dalle sue proiezioni monodimensionali. Se tutte sono normali standard, la congiunta è `N(0, I)`, e i blocchi disgiunti di coordinate sono indipendenti.
3. **SIGReg su ciascuna modalità penalizza il modality gap [Nostra osservazione].** Se `ŷ` ed `ẽ` sono entrambe ≈ `N(0, I)`, hanno la stessa distribuzione marginale: le medie coincidono in 0 e nessun classificatore le distingue. Il gap non si chiude da solo nell'addestramento contrastivo [Lett. 41].
4. **Legame con gli EBM [Nostra connessione].** A covarianza fissata, la gaussiana massimizza l'entropia. SIGReg massimizza quindi il contenuto informativo degli embedding a varianza fissata, cioè riduce il volume a bassa energia per punto dati: è il ramo regolarizzato della tassonomia di §2.1.5.

**SIGReg sul target di posa [Nostra scelta, 3/10].** L'encoder di posa si addestra da zero (§4.4.3, `worldsign-posa.md`). Il collasso del bersaglio è escluso da due difese diverse. **Lo stop-gradient:** il livello fisico legge `sg(s)`, quindi `E_fis` non può spingere il bersaglio verso la costante. **I termini della posa:** invarianza fra due viste, SIGReg su entrambe e un'ancora di ricostruzione dei keypoint, nella forma di LeJEPA, senza EMA [Lett. 35]. Una costante annulla l'invarianza ma non l'ancora né SIGReg.

### 2.5 Motivazione 4 — perché un world model predittivo

V-JEPA acquisisce una **fisica intuitiva** da video naturali: con il paradigma della *violation-of-expectation* distingue scene plausibili da implausibili, mentre i modelli a predizione di pixel e i modelli multimodali generativi restano al livello del caso [Lett. 33]. I risultati per proprietà:

| Riuscite | Non significative |
|---|---|
| object permanence, continuity, shape constancy, support, **inertia** | color constancy, **solidity**, **collisions**; gravity con risultati misti |

Due indicazioni dallo stesso studio [Lett. 33]: **addestrando solo su un dataset di movimenti (SSv2), la shape constancy si apprende male**, quindi la composizione dei dati conta; dimensione del modello e strategia di mascheramento incidono poco.

**Rilevanza per il segnato [Nostra osservazione].** Inerzia e continuità sono centrali per la cinematica del segnante. Solidity e collisions corrispondono ai **contatti** (mano–mano, mano–volto), che nel segnato hanno valore fonologico: proprio le proprietà su cui V-JEPA fallisce.

### 2.6 Motivazione 5 — perché due livelli di astrazione

**Letteratura.** In H-JEPA più JEPA sono impilati: i livelli bassi fanno predizioni dettagliate e a breve termine, quelli alti predizioni astratte, e **la predizione avviene a tutti i livelli** [Lett. 27]. Anche il segnato ha una struttura gerarchica (fonologia → lessico → enunciato) [Lett. 1].

**Il nostro argomento specifico [Nostra argomentazione]: la qualità della supervisione cambia da livello a livello.**

| Livello | Cosa si predice | Da dove viene il target | Qualità |
|---|---|---|---|
| **fisico** | lo stato del corpo nelle parti nascoste | posa estratta automaticamente dalla **stessa clip**, densa | rumore dello stimatore di posa; **nessun** rumore di allineamento |
| **semantico** | il significato della clip | didascalie | rumorose: in BOBSL il segnato arriva in media **~2,7 s dopo** il sottotitolo [Lett. 9] |

Il livello fisico è **indipendente dal testo**: è immune al disallineamento delle didascalie e usa anche i video senza didascalia affidabile.

**Perché una gerarchia nell'encoder condiviso, e non moduli impilati [Nostra argomentazione].** L'encoder video è adattato con LoRA **da entrambi i predictor**. Il gradiente fisico lo spinge a codificare lo stato e la dinamica del corpo. Quello semantico lo spinge a codificare il significato, incluse le componenti che la posa cattura male (espressione facciale, mouthing). Non servono moduli intermedi da dimensionare; le sole uscite intermedie usate sono quelle della fusione multi-livello di V-JEPA 2.1, con i blocchi fissati dal suo codice (§4.4.5).

**Perché la maschera al livello fisico e non a quello semantico [Nostra argomentazione].**

| | Livello fisico | Livello semantico |
|---|---|---|
| **Bersaglio** | rappresentazioni della posa della stessa clip | embedding della didascalia |
| **Senza maschera il compito sarebbe** | quasi una **stima di posa**: i keypoint vengono dagli stessi pixel, basterebbe una mappa locale | **già difficile**: il significato non sta in nessun pixel, serve l'intera clip |
| **Con la maschera** | inferire dal contesto le parti nascoste → dinamica, coarticolazione | un segno nascosto è una **parola mancante**: il bersaglio non è più determinato dall'input, e il predictor viene spinto verso la media |
| **Scelta** | maschera multi-blocco di V-JEPA [Lett. 30] | nessuna maschera, come VL-JEPA [Lett. 34] |

### 2.7 Motivazione 6 — perché multilingua, e perché con cautela

**A favore.** In YouTube-SL-25 l'addestramento multilingue **migliora sia le lingue con molti dati sia quelle con pochi** [Lett. 6].

**Lo zero-shot non funziona.**

| Modello | Zero-shot | Dopo fine-tuning |
|---|---|---|
| YouTube-SL-25 (BLEURT su SGS / SFS / SIS) [Lett. 6] | 9,5 / 9,3 / 6,8 | 37,7 / 25,2 / 18,8 |
| SignCLIP (R@1 fuori dominio, asl-signs) [Lett. 16] | 0,01 | 0,74 |

**Interferenza.** L'addestramento congiunto su lingue diverse può degradare le prestazioni per singola lingua [Lett. 65], e a capacità fissa troppe lingue riducono le prestazioni di ciascuna [Lett. 64].

**Come la letteratura segnica gestisce la condivisione.** MLSLT introduce un *routing* dinamico per controllare quanto condividere fra lingue [Lett. 21]; il riconoscimento continuo migliora mappando segni da un'altra lingua dei segni [Lett. 23]; un modello gloss-free multilingue ha avuto bisogno di identificare la lingua token per token [Lett. 22]. **La nostra sintesi: si condivide la percezione, si specializza l'uscita [Nostra sintesi].**

**L'ipotesi a clessidra [Nostra ipotesi, §5.1 H5]:** l'informazione sulla lingua dei segni è **bassa** nelle rappresentazioni della posa (corpo e iconicità comuni [Lett. 3, 4]), **più alta** nell'uscita dell'encoder video (lessici diversi, interferenza [Lett. 5]), di nuovo **bassa** nell'embedding semantico, allineato a testi centrati per lingua.

### 2.8 La tesi del progetto

> **Tesi.** In un world model predittivo a due livelli di astrazione per la lingua dei segni, l'uniformity necessaria al retrieval cross-modale può essere fornita da un vincolo distribuzionale sugli embedding (SIGReg), senza negativi contrastivi, raggiungendo prestazioni confrontabili con InfoNCE a parità di dati visti e di batch **[Nostra ipotesi]**.

Le ipotesi derivate (H1–H7) e le congetture di scala (S1–S3) sono in §5.

### 2.9 Posizionamento rispetto ai lavori esistenti

| Lavoro | Supervisione cross-modale | Negativi | Valuta il retrieval | Struttura predittiva |
|---|---|---|---|---|
| CiCo [Lett. 14], SEDS [Lett. 15] | contrastiva | sì | sì | no |
| SignCLIP [Lett. 16] | contrastiva | sì | sì | no |
| Uni-Sign [Lett. 8] | generativa (testo) | no | no | no |
| Scaling SLT [Lett. 19], YouTube-SL-25 [Lett. 6] | generativa | no | no | no |
| SONAR-SLT [Lett. 20] | embedding di frase | no | no | no |
| SHuBERT [Lett. 18] | nessuna (auto-supervisione sulla posa) | no | no | predizione mascherata di cluster |
| VL-JEPA [Lett. 34] | predizione di embedding testuali | **sì (InfoNCE)** | sì | un livello |
| LeJEPA / LeVJEPA [Lett. 35, 36] | nessuna | no (SIGReg) | no | LeVJEPA senza predictor |
| **Questo progetto** | predizione dell'embedding testuale + SIGReg | **no** | **sì** | **due livelli (fisico e semantico); variabile latente solo in un'ablation facoltativa** |

---

## 3. Dati

### 3.1 Scelta di fondo: solo segnato continuo

Il progetto usa **solo segnato continuo** e nessun dizionario di segni isolati **[Nostra scelta]**. Conseguenze:

- Spreadthesign, il più grande dizionario multilingue (≈500 ore, fino a 44 lingue dei segni), è escluso. È anche non ridistribuibile per licenza [Lett. 16].
- Perdiamo **SP-10**, l'unico benchmark multilingue standard, che deriva da Spreadthesign [Lett. 21].
- Il modello non ha un'ancora supervisionata a livello lessicale (§4.5.5).
- Dataset di segni isolati (ASL Citizen [Lett. 74], Sem-Lex [Lett. 75], WLASL [Lett. 76]) si potranno usare **solo per la valutazione diagnostica**, mai per l'addestramento **[Nostra proposta]**.

### 3.2 I corpora

| Corpus | Lingua | Ore | Coppie clip–testo | Unità video | Canali ≈ segnanti |
|---|---|---|---|---|---|
| **YouTube-SL-25** [Lett. 6] | 25+ lingue dei segni | 3.207 (2.980 con didascalie) | **2.160.000** | 39.197 video | 3.072 canali |
| **CSL-News** [Lett. 8] | CSL | 1.985 | **722.711** clip sotto i 512 frame | programmi TV (numero non riportato) | pochi |
| **BOBSL** [Lett. 7, 9] | BSL | 1.447–1.467 ¹ | ~1,2 M frasi | ~1.960 episodi, 426 programmi | decine di interpreti |
| **Totale** | | **≈ 6.650** | **≈ 4,08 M** (stima conservativa 3,03 M ²) | **≈ 41.200 +** CSL-News | **≈ 3.100** |

¹ Le fonti riportano valori leggermente diversi a seconda della versione del dataset.
² Contando per BOBSL solo le clip con allineamento affidabile (§3.3).

**Distribuzione per lingua in YouTube-SL-25** [Lett. 6]:

| ASL | International Sign | Indian SL | Polish SL | German SL | Brazilian SL | altre ~19 |
|---|---|---|---|---|---|---|
| 1.394 h (~43 %) | 285 h | 209 h | 137 h | 108 h | 101 h | ≥ 15 h ciascuna |

**Qualità delle didascalie** [Lett. 6]: ~0,76 % di contenuto parlato anziché segnato, ~0,96 % di interstiziali, alcune traduzioni incomplete.

**Limiti di rappresentatività da dichiarare** [Lett. 6]: sbilanciamento verso i paesi del Nord globale; solo l'**1,9 % dei dati (60 ore)** con tonalità di pelle scure.

### 3.3 Uso dei dati per livello

- **Livello fisico** (bersaglio = posa, nessun testo): si usa tutto il video. 6.650 ore ≈ 23,9 milioni di secondi → **4,8 M finestre da 5 s** senza sovrapposizione, **19,1 M** con sovrapposizione del 75 % **[Nostra stima]**.
- **Livello semantico**: servono coppie clip–testo. Per BOBSL, dove i sottotitoli hanno un ritardo medio di ~2,7 s [Lett. 9], si usano **solo le clip ad allineamento affidabile** (negli split usati in [Lett. 9]: 113.826 allineate automaticamente e 34.046 manualmente); BOBSL completo resta nel solo livello fisico **[Nostra scelta]**.

### 3.4 L'unità statistica effettiva e gli split

```
≈ 4,08 M clip  /  ≈ 41.000 video   ≈  100 clip per video
≈ 4,08 M clip  /  ≈  3.100 canali  ≈  1.300 clip per canale
```

**[Nostra argomentazione]** Il rischio di overfitting principale non è memorizzare le clip: è imparare **scorciatoie di canale** (sfondo, illuminazione, inquadratura, identità del segnante, argomento del programma). Per quel rischio la dimensione del campione rilevante sono i **video e i segnanti**, non le clip.

**Conseguenza vincolante.** Gli split si fanno **per canale** (YouTube-SL-25) e **per programma** (BOBSL, CSL-News), **mai per clip**. Con ~100 clip per video, uno split casuale mette lo stesso segnante e lo stesso sfondo in addestramento e in validazione, e nasconde l'overfitting. Teniamo due split di validazione: **held-out channel** (generalizzazione a nuovi segnanti) e **held-out language** (trasferimento fra lingue) **[Nostra scelta]**.

### 3.5 Sbilanciamento fra lingue: nessun ribilanciamento

I dati si usano **nelle proporzioni in cui esistono**, senza ribilanciare le lingue **[Nostra scelta]**. ASL e CSL coprono gran parte del corpus (§3.2).

Ribilanciare ha un costo. Il campionamento con temperatura (`p_i ∝ n_i^α`) [Lett. 64] riduce lo sbilanciamento, ma fa ripetere molte volte le lingue con pochi dati:

| α | Rapporto di campionamento ASL / lingua da 15 h |
|---|---|
| 1,0 | 93 × (nessun bilanciamento) |
| 0,5 | 9,6 × |
| 0,3 | 3,9 × |
| 0,0 | 1 ×, cioè ~40 epoche sulle lingue da 15 h |

- **Perché nessun ribilanciamento [Nostra argomentazione].** Valutazione e fine-tuning si fanno su lingue per cui i dati ci sono (OpenASL, PHOENIX-2014T, CSL-Daily). Ribilanciare vorrebbe dire ripetere molte volte le lingue piccole, con rischio di overfitting, oppure sottocampionare quelle grandi, sprecando dati.
- **Come si controlla.** R@1 ripartito per lingua e scarto train/val per lingua (§4.13).
- **Se lo sbilanciamento pesa troppo sulle prestazioni**, si cerca una soluzione alternativa in quel momento **[Aperto]**.

### 3.6 Preprocessing

| Flusso | Trattamento |
|---|---|
| **Video** | crop sul riquadro del segnante (+15 %), ridimensionato a **256×256**. **64 frame dall'intera frase**, senza troncamenti, scelti con un campionamento guidato dal movimento locale (sotto) **[Nostra scelta, §4.4.1]** |
| **Posa** | stimatore whole-body RTMW a 133 keypoint (formato COCO-WholeBody), **estratta sugli stessi 64 frame del video**, da cui si selezionano gli stessi **69 keypoint** di Uni-Sign: mano sinistra 21, mano destra 21, corpo 9, volto 18 [Lett. 8]. Normalizzazione per frame: centro nel punto medio fra le spalle, scala uguale alla distanza fra le spalle **[Nostra scelta]**. Un keypoint conta solo se ha punteggio RTMW > 1,0 (i punteggi non sono probabilità: mediana 7,6), cade dentro l'immagine e dista al più **5 larghezze di spalle** dall'origine; altrimenti è mancante. Un frame le cui spalle distano meno di **metà della mediana della clip** non fa da riferimento e prende quello dell'ultimo frame valido **[Nostra scelta, dopo PC5]**: senza queste due guardie l'1 % delle clip arrivava a centinaia di unità di spalla (massimo 628) e dominava ogni errore quadratico. La presenza di ogni keypoint si conserva come peso. Il formato è quello dell'encoder di posa (§4.4.3): 32 passi di due frame × 69 keypoint |
| **Testo** | embedding precalcolati una volta con EmbeddingGemma (§4.4.4) |

**Selezione dei 64 frame [Nostra scelta, ispirata a Lett. 93, 94].** Ogni frase si prende intera; dai suoi T frame nativi se ne scelgono 64:

```
S[t]        movimento locale: media dei blocchi dell'immagine con la maggiore differenza fra frame consecutivi
32 frame    distribuiti uniformemente sulla CUMULATA di S      →  densi dove c'è movimento
32 frame    distribuiti uniformemente nel TEMPO                 →  nessun tratto della frase resta scoperto
indici fusi e ordinati (se coincidono, si completa con altri frame uniformi); la posa si estrae sugli stessi frame
se T ≤ 64:  campionamento uniforme, con frame ripetuti
```

- **Base in letteratura.** Campionare sulla cumulata del movimento migliora il campionamento uniforme soprattutto dove conta il movimento: +1,5–2,2 punti di top-1 su Something-Something e Diving48, +0,4–1,2 su dataset centrati sull'aspetto [Lett. 93]. Con soggetti piccoli nell'inquadratura, la variante a patch guadagna fino al 13,8 % relativo [Lett. 94]. Il movimento *locale* serve perché le dita si muovono anche quando le braccia sono ferme **[Nostra argomentazione]**.
- **Nei concorrenti.** CiCo usa finestre dense su tutto il video [Lett. 14], C²RL segmenti uniformi [Lett. 87], VL-JEPA frame uniformi a 2 fps [Lett. 34]; Uni-Sign sceglie dinamicamente i frame RGB in base alla confidenza delle mani [Lett. 8].
- **Spaziatura.** Grazie alla metà uniforme, due frame consecutivi distano al massimo circa T/32 frame nativi **[Nostra osservazione]**. L'effetto sull'encoder, pre-addestrato su frame a passo fisso [Lett. 30], si misura nel collaudo (§4.13.1).

### 3.7 Composizione del batch

- **Ogni campione è una clip singola** con la sua posa e, se disponibile, la sua didascalia.
- Le clip **con didascalia affidabile** partecipano a **entrambi** i passaggi (fisico e semantico, §4.5.1); i video **senza didascalia affidabile** (BOBSL non allineato) solo al passaggio fisico **[Nostra scelta]**.
- Nel braccio InfoNCE le clip dello stesso video sono **falsi negativi quasi duplicati** e vanno escluse dal denominatore (§4.6).

### 3.8 Augmentation

**Principio.** L'augmentation produce **viste, non campioni**: non aggiunge informazione, agisce come regolarizzatore. Soprattutto, **non moltiplica né i video né i segnanti [Nostra argomentazione]**.

**Vincolo nuovo imposto dall'architettura [Nostra argomentazione].** Al livello fisico il video è l'input e la posa della **stessa clip** è il bersaglio. Ogni trasformazione geometrica va quindi applicata **in modo identico a video e keypoint**; una trasformazione applicabile solo alla posa renderebbe il bersaglio incoerente con ciò che il video mostra.

| Ammessa | Motivazione |
|---|---|
| **Jitter del riquadro** (±10 %) | robustezza standard; la stessa trasformazione affine si applica ai keypoint |
| **Colore e luminosità** | solo sul video; non tocca la geometria |

Combinazioni: `5 × 5 = 25` viste per clip **[Nostra stima]**, che **non** equivalgono a 25 campioni indipendenti.

| Vietata | Motivazione |
|---|---|
| **Flip orizzontale** | scambia la mano dominante e la lateralità dello spazio segnico, dove i referenti sono collocati a destra o a sinistra **[Nostra argomentazione]** |
| **Ritaglio temporale** della frase | la frase si prende intera: un ritaglio toglierebbe segni e renderebbe il bersaglio semantico non determinato (§2.6) |
| **Rotazione 3D con riproiezione 2D della sola posa** | utile nel riconoscimento basato su posa [Lett. 25], ma senza equivalente sul video: il bersaglio non corrisponderebbe più all'input **[Nostra argomentazione]** |
| **Riscalatura delle proporzioni degli arti** | stesso motivo: si applica solo alla posa |
| Time-warp aggressivo | velocità e ripetizione del movimento sono distintive |
| Crop che taglia mani o volto | le componenti manuali e non manuali portano informazione |
| Mixup / CutMix | mescolano frame di segni diversi |

**Target della posa in 2D, non in 3D.** Le stime 3D monoculari sono più rumorose delle 2D e portano ad accuratezze inferiori nel riconoscimento [Lett. 24]; l'ambiguità prospettica è strutturale e il sollevamento 2D→3D fallisce proprio nei contatti mano–mano e mano–volto [Lett. 26] **[Nostra scelta]**. **[Aperto]** Un canale binario che indichi quale articolatore è davanti (profondità relativa, molto più robusta di quella assoluta), **non nel piano attuale** (§4.14).

**Esclusa: l'augmentation delle didascalie** in più lingue parlate. Associare a un video ASL traduzioni in altre lingue introduce nel target varianza linguistica estranea al segnato **[Nostra scelta]**.

### 3.9 Benchmark e comparabilità

Nessun lavoro del settore usa lo stesso corpus di pretraining: Uni-Sign ha costruito CSL-News [Lett. 8], YouTube-SL-25 è un corpus nuovo [Lett. 6], SignCLIP usa dati non ridistribuibili [Lett. 16]. **Ciò che deve essere comparabile è il protocollo [Nostra argomentazione].** Cinque accorgimenti **[Nostra scelta]**:

1. pretraining sul nostro corpus;
2. sul modello finale, fine-tuning e valutazione sugli **split standard**: OpenASL con il protocollo di C²RL [Lett. 87], PHOENIX-2014T e CSL-Daily con il protocollo di CiCo/SEDS [Lett. 14, 15], riportando R@1/5/10;
3. ore di pretraining dichiarate nella tabella di confronto;
4. **controllo a dati appaiati**: il nostro metodo addestrato sui soli dati di addestramento di OpenASL, confrontato con i baseline sugli stessi dati;
5. **audit di contaminazione, vincolante.** OpenASL è in larga parte un sottoinsieme di YouTube-ASL, con cui condivide i due canali più grandi [Lett. 88], e YouTube-SL-25 contiene YouTube-ASL [Lett. 6]; nessuno dei due dichiara di escludere i video di valutazione. Si rimuove quindi dal pretraining ogni segmento di YouTube-SL-25 che, nello stesso video, si sovrappone a una clip di validazione o di test di OpenASL (margine ±2 s), più le didascalie identiche o quasi identiche a quelle di validazione e test nello stesso video **[Nostra scelta]**.

**Perché per intervalli e non per video interi [Nostra argomentazione].** Lo split di OpenASL è per coppie casuali: il suo addestramento contiene già il resto di quei video, quindi rimuovere gli intervalli riproduce le condizioni del benchmark, mentre rimuovere i video interi toglierebbe i canali ASL più grandi.

**Conseguenze da dichiarare.** OpenASL è un benchmark **in dominio** (stessi video e segnanti in addestramento e test): misura la prestazione, non la generalizzazione, che resta affidata allo split held-out channel. I video di YouTube possono sparire nel tempo [Lett. 88]: l'elenco degli identificativi usati si congela e si pubblica **[Nostra scelta]**.

**[Aperto]** Non esiste un benchmark standard di retrieval multilingue continuo: ne definiremo uno da YouTube-SL-25, senza fondarci la tesi principale.

---

## 4. Architettura e metodologia

### 4.1 Principi di progetto

1. **Replicare ricette validate invece di inventarne.** Maschera e loss di V-JEPA 2.1 al livello fisico [Lett. 30, 32]; schema di VL-JEPA al livello semantico [Lett. 34] **[Nostra scelta]**.
2. **Riusare tutti i pesi pre-addestrati utili**, adattandoli con LoRA **su tutti i blocchi**: encoder video e predictor fisico. L'encoder di posa non ha pesi pre-addestrati e si addestra da zero (§4.4.3) **[Nostra scelta]**.
3. **Tetto di 30 M di parametri addestrabili**, per il rischio di overfitting di §3.4. Era 20 M, poi 22 M per la fusione multi-livello di V-JEPA 2.1 (§4.4.5), poi 30 M il 3/10 per l'encoder di posa da zero (`worldsign-posa.md` §3), che si addestra sulla posa e non sulle didascalie **[Nostra scelta]**.
4. **Parametri vincolati contro parametri liberi [Nostra argomentazione].** Le matrici LoRA su pesi congelati restano in un intorno a basso rango di una funzione già sensata e sono a basso rischio. I moduli addestrati da zero non hanno un prior e sono a rischio più alto. **Se c'è da tagliare, si tagliano i liberi.**
5. **Un encoder congelato più grande è più sicuro, non meno [Nostra argomentazione].** I suoi pesi congelati non possono memorizzare il corpus, e spostano lavoro dai parametri appresi a quelli pre-addestrati. Il suo costo è computazionale, non statistico.

### 4.2 Panoramica

```
 ═══════════ RAMO TESTUALE (precalcolato, fuori dal grafo di addestramento) ═══════════
  didascalia ─► EmbeddingGemma-300M (congelato) ─► 768-d, intero (nessun troncamento)
            ─► centratura per lingua parlata ─► testa MLP  768 → 512 → 512  ─► ẽ ∈ ℝ⁵¹²
 ═══════════════════════════════════════════════════════════════════════════════════════

 VIDEO  crop 256² · 64 frame · tubelet 2×16×16  →  8.192 token

 ─── PASSAGGIO FISICO ── maschera multi-blocco di V-JEPA (~90 %) ──────────────────────────
  video ⊙ m ─► ┌─────────────────────────────────────────────────────┐
               │ ENCODER VIDEO  V-JEPA 2.1 ViT-L  (PC3, 29/9)         │ ─► ~820 token visibili
               │ 24 blocchi · d = 1024 · CONGELATO + LoRA r = 16      │
               └─────────────────────────────────────────────────────┘
                              ▼
               FUSIONE MULTI-LIVELLO (V-JEPA 2.1): blocchi 6·12·18·24 ─► LayerNorm per livello
               ─► concatenazione 4.096 ─► MLP 4.096 → 1.024 → 384
                              ▼
               ┌─────────────────────────────────────────────────────┐
               │ PREDICTOR FISICO  (predictor di V-JEPA 2.1)          │ ─► 8.192 token predetti
               │ 12 blocchi · d = 384 · LoRA r = 16 + testa nuova     │
               │ [Aperto: PC6]                                        │
               └─────────────────────────────────────────────────────┘
                              ▼
               lettura R per passo t (4 riquadri concatenati) ─► ŝ_t ── L1 pesata ──► E_fis
                                                                              ▲ sg
  POSA  69 keypoint, presenza ─► ENCODER POSA (da zero, §4.4.3) ─► s_t ───────┘
                                 L_inv (vista) · L_anchor (keypoint) · SIGReg

 ─── PASSAGGIO SEMANTICO ── nessuna maschera, come VL-JEPA ─────────────────────────────────
  video ─► ENCODER VIDEO (lo stesso, senza gradiente) ─► sg(8.192 token dell'ultimo blocco)  ┐
                                          8 query apprese (costanti) ┤
                                                                     ▼
               ┌─────────────────────────────────────────────────────┐
               │ PREDICTOR SEMANTICO  4 blocchi · d = 384 · da zero   │
               └─────────────────────────────────────────────────────┘
                              ▼
                  media delle 8 query ─► ŷ ∈ ℝ⁵¹² ──── D ────► E_sem ◄──── ẽ

 ═══════════════════════════════════════════════════════════════════════════════════════
   SIGReg sulla posa per vista e per modalità su { ŷ } e { ẽ }  ·  un solo λ  ·  addestramento per livello
   retrieval = argmin di E_sem  ·  plausibilità = E_fis media su più maschere
 ═══════════════════════════════════════════════════════════════════════════════════════
```

| | Quanti | Quali |
|---|---|---|
| **Encoder** | 3 | video (congelato + LoRA, adattato dal solo livello fisico) · posa (da zero, `worldsign-posa.md`) · testo (congelato, precalcolato) |
| **Predictor** | 2 | fisico (riuso di V-JEPA 2.1 + LoRA, con fusione multi-livello) · semantico (da zero, schema VL-JEPA) |
| **Vettore di retrieval** | — | `ŷ`, l'uscita del predictor semantico |
| **Parametri addestrabili** | ≈ 29,8 M | dettaglio in §4.8 |

### 4.3 Notazione

| Simbolo | Significato |
|---|---|
| `T = 64`, `T' = 32` | frame per clip; posizioni temporali dopo la tokenizzazione |
| `N = 8.192` | token per clip (`32 × 16 × 16`) |
| `m`, `𝓜` | maschera multi-blocco; insieme dei token mascherati |
| `a ∈ {LH, RH, corpo, volto}` | articolatore |
| `p̂_{t,j}`, `c_{t,j}` | keypoint `j` normalizzato al tempo `t`, e sua confidenza |
| `s_t ∈ ℝ^C` | bersaglio di posa per passo, `C = 192` (`worldsign-posa.md`) |
| `ŝ_t ∈ ℝ^C` | la stessa quantità, predetta dal video mascherato |
| `ŷ ∈ ℝ⁵¹²` | embedding semantico predetto dal video (vettore di retrieval) |
| `ẽ ∈ ℝ⁵¹²` | target testuale della didascalia |
| `q_1 … q_8` | query apprese del predictor semantico |
| `sg(·)` | stop-gradient: il valore passa, il gradiente no (`worldsign-gerarchia.md` §2) |
| `z`, `K` | variabile latente discreta e sua cardinalità (**solo nell'ablation facoltativa ESP-4**, §4.5.9) |

### 4.4 Componenti

#### 4.4.1 Encoder video: V-JEPA 2.1 ViT-L distillato

**Fatti** [Lett. 32, 72]:
- V-JEPA 2.1 rilascia varianti distillate **ViT-B (80 M)** e **ViT-L (300 M)**, ottenute da un teacher ViT-G.
- I checkpoint hanno risoluzione nativa **384×384**, **64 frame**, tubelet da 2, e usano la **3D-RoPE** (posizione codificata come rotazione nell'attenzione, separatamente per tempo, altezza e larghezza) [Lett. 31, 72].
- V-JEPA 2 viene addestrato in modo progressivo: 16 frame a 256², poi *cool-down* fino a 64 frame a 384² [Lett. 31].

**Scelte:**

- **Congelato + LoRA r = 16 su tutti i 24 blocchi**, attenzione e MLP **[Nostra scelta]**.
- **Attenzione bidirezionale**, come nei pesi pre-addestrati **[Nostra scelta]**.
- **256×256 con crop sul segnante**, invece di 384×384 a inquadratura intera: 8.192 token contro 18.432. La 3D-RoPE lo rende possibile senza interpolare posizioni, e il crop dà alle mani più pixel **[Nostra scelta, confermata da PC4 il 29/9: a 384² il probe delle mani guadagna +0,005 di R², sotto la soglia di +0,05, e il retrieval peggiora (T2V R@1 0,056 contro 0,067)]**.
- **Uscite verso i predictor.**
  - Il predictor fisico riceve la **fusione multi-livello di V-JEPA 2.1**: blocchi 6, 12, 18 e 24, ciascuno con la propria LayerNorm, concatenati e fusi da un MLP (§4.4.5) [Lett. 32, 72]. Sono gli indici che il codice di V-JEPA 2.1 usa per un encoder a 24 blocchi, non una scelta nostra.
  - Il predictor semantico riceve l'uscita dell'ultimo blocco, come in VL-JEPA [Lett. 34], **senza gradiente**: l'encoder lo adatta solo il livello fisico (`worldsign-gerarchia.md`) **[Nostra scelta]**.

**Perché V-JEPA 2.1, e con quale riserva.** La 2.1 introduce una loss predittiva **densa** su tutti i token e una **supervisione profonda** su più layer intermedi [Lett. 32], e l'avevamo scelta per questo. **Però** le varianti distillate sono state addestrate con una loss **solo sull'ultimo layer del teacher, senza supervisione profonda** [Lett. 32]. Nei checkpoint distillati la predizione multi-livello non c'è (`n_output_distillation = 1`), mentre ViT-g e ViT-G ne hanno 4 [Lett. 72]: la fusione la aggiungiamo noi (§4.4.5). **PC3, chiuso il 29/9: V-JEPA 2.1-L.** Sulle feature congelate (8.000 clip di YouTube-SL-25, test su 1.394) i due encoder sono alla pari: R² delle mani 0,706 contro 0,700 di V-JEPA 2-L, l'encoder usato da VL-JEPA [Lett. 34]; T2V R@1 0,067 per entrambi; V2T R@1 0,039 contro 0,067, a favore di V-JEPA 2-L. Il probe fonologico non è calcolabile. A parità vale la regola di §4.12.1: la 2.1-L. **Con la regola del 4/10** il probe testuale si legge sulla media delle due direzioni (0,053 contro 0,067) e va a V-JEPA 2-L, mentre le mani vanno alla 2.1-L. Una vittoria a testa non basta a nessuno dei due, quindi **l'esito non cambia**: 2.1-L.

#### 4.4.2 LoRA

Ogni matrice lineare congelata `W` diventa [Lett. 67]:

```
W'  =  W  +  (α / r) · B · A

   W ∈ ℝ^{d_out × d_in}   congelata
   A ∈ ℝ^{r × d_in}       addestrabile, inizializzazione gaussiana
   B ∈ ℝ^{d_out × r}      addestrabile, inizializzata a ZERO  →  W' = W all'inizio
   r                      rango;  α fattore di scala
   parametri addestrabili per matrice:  r · (d_in + d_out)
```

| Dove | Rango | Learning rate |
|---|---|---|
| Encoder video, 24 blocchi | 16 | base |
| Predictor fisico, 12 blocchi | 16 | base |

**Avvertenza** [Lett. 68]: LoRA **non** protegge dal *catastrophic forgetting* più del fine-tuning completo. Limita il sottospazio dei parametri, non la deriva della funzione.

#### 4.4.3 Encoder della posa (target del livello fisico)

**Scelta [Nostra scelta, 3/10]: l'encoder di posa di worldSign, adattato al tubelet di V-JEPA e addestrato da zero insieme al resto.** Tutti i dettagli sono in `worldsign-posa.md`; qui la sintesi.

- **Ingresso:** i token di posa (69 giunti × 32 passi, x, y e presenza dei due frame), ricanonicalizzati frame per frame fra le spalle; 9 canali per giunto e frame (globale, locale rispetto alla radice della parte, osso, velocità, valido), i due frame del passo concatenati.
- **Architettura:** un transformer spaziale per articolatore (corpo, mani, volto; pesi non condivisi), media sui giunti; i 4 token **concatenati** (512); un transformer temporale a 512 sui 32 passi; `LayerNorm` e `Linear(512 → 192)`, senza dropout. **8,03 M parametri.** Il bersaglio è **un vettore per passo**, `s_t ∈ ℝ¹⁹²`.
- **Addestramento:** invarianza fra 4 viste (la sequenza pulita e tre estrazioni di disturbi: camera, rumore del rilevatore, giunti nascosti) nella forma di LeJEPA [Lett. 35]; SIGReg su ogni vista passo per passo, come LeWorldModel; ancora di ricostruzione dei keypoint; λ = 0,04; senza EMA. Learning rate 3e-4, warm-up sul 20 % della run e coseno (`worldsign-posa.md` §4).
- **Rapporto con il livello fisico:** `E_fis` legge `LN(sg(s))`. Il video non può spostare il bersaglio.
- **S-JEPA** [Lett. 99] è replicato a parte, sui dati del paper (`signworld/models/sjepa/`), per un adattamento futuro; non è collegato al modello. Le misure di PC5 riguardavano un S-JEPA pre-addestrato che non fa più parte del piano.

#### 4.4.4 Ramo testuale (l'Y-Encoder)

**Encoder: EmbeddingGemma-300M, congelato e precalcolato.** Uscita a 768 dimensioni, troncabile con MRL a 512/256/128, oltre 100 lingue, contesto fino a 2.048 token [Lett. 57]. È l'encoder testuale di default di VL-JEPA [Lett. 34], dove svolge il ruolo di Y-Encoder. Precalcolare vuol dire che in addestramento il ramo testuale costa quanto una lettura da tabella.

**Alternative valutate.** L'ablazione di VL-JEPA mostra che l'encoder testuale pesa molto: rispetto a EmbeddingGemma, Qwen3-Embedding-8B guadagna +5,4 in retrieval e PE-Core-G +7,9 [Lett. 34]. Abbiamo scartato PE-Core-G perché addestrato su *alt-text* di immagini [Lett. 60], che **valutiamo prevalentemente inglese** e quindi poco adatto a didascalie in 25+ lingue **[Nostra valutazione, non verificata quantitativamente]**. SONAR [Lett. 58] e BGE-M3 [Lett. 59] sono multilingui; abbiamo preferito allinearci a VL-JEPA **[Nostra scelta]**. Non scegliere SONAR significa **rinunciare al suo decoder**: la traduzione richiederà un decoder addestrato a valle **[Nostra osservazione]**.

**Nessun troncamento [Nostra scelta, 29/9].** Il vettore di EmbeddingGemma si usa intero, a 768 dimensioni. PC1 ha mostrato che troncare non riduce l'anisotropia che SIGReg vede (≈ 14 volte il valore di una gaussiana sia a 768 sia a 512), e che sulle nostre didascalie, a 512 e 256 dimensioni, il prefisso MRL non conserva i vicini meglio di coordinate prese a caso; l'IsoScore più alto delle dimensioni minori è un effetto della dimensione stessa [Nostra misura].

**Centratura per lingua parlata:**

```
μ_ℓ  =  media degli embedding delle didascalie in lingua parlata ℓ        (calcolata una volta)
e°   =  e  −  μ_ℓ
```

Gli embedding multilingue tendono a raggrupparsi **per lingua anziché per significato**, e la centratura per lingua è un rimedio semplice ed efficace [Lett. 61]. È la stessa logica di sottrarre la media comune [Lett. 53].

**Niente whitening fisso.** Il whitening (rendere la covarianza identità) aiuta embedding BERT fortemente anisotropi [Lett. 47, 48], ma:

- su encoder addestrati in modo contrastivo il post-processing **peggiora le prestazioni**: aumenta l'uniformity ma **degrada l'allineamento** [Lett. 46];
- i modelli di embedding contrastivi moderni sono **già fortemente isotropi**, con dimensionalità effettiva molto sotto quella nominale [Lett. 51];
- non c'è una relazione stabilita fra anisotropia e prestazioni [Lett. 50];
- il whitening è sconsigliato per la classificazione [Lett. 52].

**Al suo posto, una testa MLP a due layer** (`768 → 512 → 512`, GELU, **inizializzazione standard**, che entra nello stadio P con il suo warm-up, §4.10), addestrata insieme alle altre loss. È un *whitening durante l'addestramento*: WhitenedCSE mostra che integrarlo nel training evita il degrado dell'allineamento del post-processing [Lett. 49]; un whitening appreso è preferibile a quello PCA nel retrieval di immagini [Lett. 55]; e in LLaVA-1.5 un proiettore MLP su encoder congelati supera quello lineare [Lett. 70] **[Nostra scelta]**.

**Nota di parsimonia.** Le varianti «senza testa», «testa lineare» e «whitening fisso» **non saranno testate**: la decisione di non usare il whitening fisso si basa sulla letteratura citata, non su un nostro esperimento **[Nostra scelta]**.

#### 4.4.5 Predictor fisico: riuso del predictor di V-JEPA 2.1

**Perché il riuso ha senso [Nostra argomentazione].** Il predictor di V-JEPA 2.1 è addestrato a fare **esattamente** il compito del nostro livello fisico — predizione densa da video mascherato con maschere multi-blocco. Cambia solo lo spazio del bersaglio.

| Aspetto | Dettaglio |
|---|---|
| Struttura | per `vjepa2_1_vit_large_384`: **12 blocchi, dimensione 384** [Lett. 72] |
| Ingresso | **fusione multi-livello di V-JEPA 2.1** (sotto), più token di maschera con la posizione del target (il condizionamento «dove» di §2.1.4) [Lett. 30] |
| Uscita | **da sostituire**: nella distillazione il layer finale produce la dimensione del teacher ViT-G [Lett. 32]; si aggiunge una **testa nuova** verso lo spazio della posa |
| Adattamento | LoRA r = 16 sui 12 blocchi + fusione multi-livello e testa d'uscita addestrate da zero **[Nostra scelta]** |
| ⚠️ Condizione | le fonti sono **discordanti**: il paper descrive i modelli distillati come solo encoder, il codice di caricamento restituisce anche un predictor [Lett. 32, 72] **[Aperto: PC6]** |
| Ripiego | se i pesi non ci sono: stessa struttura ma **1 blocco da zero** (~1,8 M), per restare nel tetto di 30 M; oppure il predictor di V-JEPA 2-L, coerente solo se PC3 sceglie V-JEPA 2-L come encoder **[Nostra scelta]** |

**Fusione multi-livello, riprodotta da V-JEPA 2.1** [Lett. 32; codice ufficiale, Lett. 72]:

```
ENCODER ViT-L (24 blocchi)
  uscite dei blocchi 6, 12, 18, 24          (hierarchical_layers = [5, 11, 17, 23] nel codice)
  ciascuna con la propria LayerNorm
  concatenazione lungo i canali:  4 × 1.024 = 4.096

FUSIONE (predictor_embed)
  Linear 4.096 → 1.024  ·  attivazione  ·  Linear 1.024 → 384

PREDICTOR (12 blocchi)  →  norma condivisa  →  testa verso lo spazio della posa
```

**Perché la adottiamo.** Nell'ablazione di V-JEPA 2.1 la sola loss sui token visibili migliora le feature dense ma **perde circa 10 punti sui compiti globali**; la predizione multi-livello li recupera quasi tutti [Lett. 32, Tab. 1]:

| Ricetta | IN1K | SSv2 | NYU RMSE ↓ | ADE20K mIoU |
|---|---|---|---|---|
| V-JEPA 2 | 82,2 | 72,8 | 0,682 | 22,2 |
| + loss sui token visibili | 72,6 | 62,5 | 0,474 | 33,8 |
| + predizione multi-livello | **80,8** | **72,1** | 0,463 | **38,6** |

Senza la fusione, il nostro livello fisico corrisponderebbe alla seconda riga, e il retrieval semantico è un compito globale **[Nostra argomentazione]**. L'ablazione riguarda però un pretraining da zero con teacher EMA, non un adattamento con LoRA verso la posa.

**Cosa è riprodotto esattamente e cosa no:**

| Parte | Esatta? |
|---|---|
| Blocchi 6/12/18/24, LayerNorm per livello, concatenazione | sì |
| Fusione 4.096 → 1.024 → 384 | sì |
| Loss sui mascherati e sui visibili (L1 pesata per distanza) | sì |
| 4 uscite verso 4 livelli dell'encoder del target | **no**: in V-JEPA 2.1 i quattro target sono layer dello stesso encoder video; il nostro target è la posa. Si usa **un'uscita**, verso la posa **[Nostra scelta]** |

L'attivazione della fusione si prende dalla configurazione ufficiale di V-JEPA 2.1 in fase di implementazione **[Aperto]**. Inizializzazione: §4.10.

#### 4.4.6 Predictor semantico: lo schema di VL-JEPA

**Corrispondenza con le quattro componenti di VL-JEPA** [Lett. 34]:

| Componente di VL-JEPA | In VL-JEPA | Da noi |
|---|---|---|
| **Terna di addestramento** | ⟨X_V, X_Q, Y⟩: visivo, query, testo bersaglio | ⟨clip, query appresa, didascalia⟩ |
| **X-Encoder** (X_V → S_V) | V-JEPA 2 ViT-L congelato | encoder video V-JEPA 2.1-L + LoRA, **condiviso col livello fisico** **[Nostra scelta]** |
| **Predictor** (⟨S_V, X_Q⟩ → Ŝ_Y) | ultimi 8 layer di Llama-3.2-1B, 490 M, attenzione bidirezionale | **4 blocchi, d = 384, da zero**, attenzione bidirezionale; 490 M sarebbero fuori budget **[Nostra scelta]** |
| **Query X_Q** | una domanda; in pre-addestramento un prompt di captioning | **8 query apprese costanti**: il compito è sempre la didascalia **[Nostra scelta]** |
| **Y-Encoder** (Y → S_Y) | EmbeddingGemma, addestrato con LR ×0,05 | EmbeddingGemma congelato, intero a 768 + centratura + testa MLP (§4.4.4) |
| **Y-Decoder** (Ŝ_Y → Ŷ) | escluso dalla fase principale di addestramento | nessuno; traduzione e produzione restano compiti a valle (§1.2) |

**Struttura [Nostra scelta, salvo dove indicato]:**
- proiezione d'ingresso `1024 → 384`;
- **3D-RoPE sui token video**: senza informazione di posizione, attenzione e pooling sono invarianti alle permutazioni, cioè ciechi all'ordine temporale **[Nostra argomentazione]**;
- attenzione bidirezionale congiunta su token video e query [Lett. 34];
- uscita: **media dei token d'uscita delle 8 query**, poi proiezione `384 → 512`. VL-JEPA fa la media sui token non di padding [Lett. 34]; noi la limitiamo alle query, che hanno la stessa lunghezza per ogni clip.

**Perché serve un predictor anche senza maschera.**

| Motivo | Dettaglio | Fonte |
|---|---|---|
| **C'è qualcosa da predire** | la didascalia non entra mai nell'encoder video: il bersaglio è interamente nascosto all'input | [Lett. 29, 34] |
| **Forma diversa** | l'encoder produce 8.192 token a 1024 dimensioni; il bersaglio è un solo vettore nello spazio del testo. Aggregare e cambiare spazio è già una funzione da imparare | [Nostra argomentazione] |
| **La capacità conta** | in VL-JEPA più layer nel predictor danno +0,8 nel retrieval e +3,0 in VQA; un'ablazione **senza** predictor non esiste | [Lett. 34, Tab. 7d] |
| **Encoder condiviso** | se l'uscita dell'encoder dovesse stare direttamente nello spazio del testo, entrerebbe in conflitto con il livello fisico; il predictor separa i due compiti | [Nostra argomentazione] |
| **Asimmetria d'informazione** | il video contiene cose assenti dalla didascalia (identità, sfondo, prosodia) e viceversa; il predictor decide cosa tenere | [Nostra argomentazione] |

#### 4.4.7 Capacità specifica per lingua («clessidra») — [Aperto]

- **Nella configurazione di riferimento non c'è alcun modulo specifico per lingua** **[Nostra scelta]**: tutti i parametri sono condivisi.
- **Dove metterlo, se servirà [Nostra ipotesi]:** il posto naturale è il predictor semantico — query condizionate sulla lingua parlata della didascalia, oppure una LoRA per lingua dei segni a rango minimo sul suo primo blocco.
- **All'inferenza** la lingua dei segni del video può non essere nota: si userebbe una **miscela pesata** dei moduli, con pesi dati da un classificatore di lingua, differenziabile e adatta anche a International Sign **[Nostra proposta]**.
- ESP-7 è tolto dal piano attuale (§4.14): il modulo resta un'estensione, da valutare con i probe di identificazione della lingua (§4.12.3).

#### 4.4.8 Una scelta considerata e rimossa: la partizione dell'embedding

Avevamo valutato di dividere l'embedding di retrieval in una parte semantica e una parte «di lingua». **L'abbiamo rimossa [Nostra scelta]**, per quattro ragioni:

1. la centratura per lingua toglie già la componente linguistica **dal target** [Lett. 61];
2. SIGReg rende indipendenti le coordinate (§2.4), quindi un'eventuale informazione di lingua resta separabile anche senza assegnarle coordinate dedicate;
3. i risultati di identificabilità contenuto/stile valgono a meno di una trasformazione invertibile e non richiedono una partizione rigida [Lett. 45];
4. i benchmark principali sono monolingui.

Al suo posto: un **probe di identificazione della lingua su `ŷ`**. Se risultasse alto, si applica in valutazione, a costo zero, una **centratura per lingua dei segni**, `score = cos(ŷ − μ_ℓ, ẽ)` **[Nostra scelta]**.

### 4.5 Formulazione a energia e loss

#### 4.5.1 Due passaggi per passo di addestramento

| Passaggio | Input dell'encoder video | Predictor | Bersaglio | Chi riceve gradiente |
|---|---|---|---|---|
| **Posa** | — | — | sé stessa: una vista, i keypoint | encoder di posa e decoder dell'ancora |
| **Fisico** | video con maschera multi-blocco (~10 % dei token visibili) | fisico | `sg(s_t)` dalla posa **completa** | LoRA video, LoRA e testa del predictor fisico, fusione |
| **Semantico** | video **completo**, letto senza gradiente | semantico | `ẽ`, precalcolato | predictor semantico, testa testuale |

- **Addestramento per livello, nessuna EMA** **[Nostra scelta, 3/10]**: gli stop-gradient `sg(s)` e `sg(Enc(x))` fanno sì che ogni loss aggiorni solo il suo livello (`worldsign-gerarchia.md` §3).
- La posa **non entra mai** nell'input del contesto: serve solo come bersaglio (§4.6).

#### 4.5.2 Livello fisico: maschera, predizione densa, lettura per passo

**Maschera: quella di V-JEPA, senza modifiche** [Lett. 30, 31].

| | Valore |
|---|---|
| Maschere brevi | **8 blocchi**, ciascuno copre il **15 %** del frame |
| Maschere lunghe | **2 blocchi**, ciascuno copre il **70 %** del frame |
| Forma | rapporto d'aspetto casuale fra 0,75 e 1,5 |
| Estensione temporale | ogni blocco spaziale **si ripete su tutti i frame** (tubi) |
| Rapporto medio mascherato | **~90 %** |

V-JEPA 2 dichiara di usare la stessa strategia [Lett. 31]. L'ablazione di V-JEPA [Lett. 30] mostra perché non inventarne un'altra:

| Strategia di maschera | K400 | SSv2 | IN1K |
|---|---|---|---|
| Tubi casuali (90 %) | 51,5 | 46,4 | 55,6 |
| Multi-blocco **causale** (6 frame) | 61,3 | 49,8 | 66,9 |
| Multi-blocco **causale** (12 frame) | 71,9 | 63,6 | 72,2 |
| **Multi-blocco** | **72,9** | **67,4** | **72,8** |

Nascondere il futuro e predirlo dal solo passato va **peggio** che predire grandi blocchi con contesto su tutta la clip [Lett. 30].

**Loss densa: quella di V-JEPA 2.1** [Lett. 32]. La loss totale somma due termini: uno sui **token mascherati** e uno sui **token visibili**, entrambi **L1** contro il target **normalizzato con layer norm**, il secondo **pesato in modo decrescente con la distanza dalla maschera**. Il termine sui token visibili impedisce che diventino aggregatori globali, ed è la chiave della qualità delle feature dense [Lett. 32].

```
L_dense  =  L_pred  +  λ · L_ctx

L_pred   =  media sui token mascherati i  di   (1/C) · ‖ ẑ_i − LN(h_i) ‖₁
L_ctx    =  media sui token visibili i    di   (1/C) · ‖ ẑ_i − LN(h_i) ‖₁ / √( d_min(i) )

   LN(·)      layer norm sui C canali del target, senza parametri
   d_min(i)   distanza spazio-temporale minima da un token mascherato
   λ = 0,5, raggiunto con un warm-up progressivo                                   [Lett. 32]
```

Il codice di V-JEPA 2.1 calcola così entrambi i termini: `F.layer_norm` sui token del teacher, errore `|z − h|` con `loss_exp = 1` mediato su token e canali, peso `1 / √d_min` solo nel termine di contesto; la media si fa poi sulle maschere del passo [Lett. 32].

- **Predizione multi-livello: fusione in ingresso sì, quattro uscite no.** Il predictor fisico riceve la fusione dei blocchi 6/12/18/24 come in V-JEPA 2.1; l'uscita resta una, verso la posa, perché i quattro target di V-JEPA 2.1 sono layer dell'encoder video (§4.4.5) **[Nostra scelta]**.

**Lettura per passo [Nostra scelta, 3/10, adattamento di Lett. 32].** Il bersaglio non è per token ma per passo: per ogni passo si fa la media dei token predetti, visibili e nascosti, nel riquadro di ciascun articolatore; le 4 medie si concatenano e una testa le porta allo spazio della posa, come l'encoder di posa concatena le sue 4 parti (`worldsign-gerarchia.md` §4).

```
ŝ_t  =  Testa( [ media dei token predetti da Pred_fis( Enc_v(v ⊙ m) ) nel riquadro di a al passo t ]_{a = corpo, sx, dx, volto} )
```

- Il riquadro si ricava dai keypoint, proiettato sulla griglia di 16×16 patch; un riquadro non visibile dà zeri.
- Due letture separate (solo nascosti, solo visibili), come teneva V-JEPA 2.1, non funzionano con un bersaglio per passo: un riquadro senza token nascosti lascerebbe vuota la sua parte mentre il bersaglio la contiene. La distinzione fra i due termini passa nei pesi (§4.5.3).

#### 4.5.3 Energia fisica, plausibilità e ancora

```
e_t(m)  =  (1/C) · ‖ ŝ_t − LN( sg(s_t) ) ‖₁             errore di un passo: L1 media sui canali, target con layer norm

ω_t(m)  =  c_t · ( n^m_t(m)  +  λ · w^v_t(m) )              peso del passo

               Σ_t  ω_t(m) · e_t(m)
E_fis(v, p ; m)  =  ──────────────────────                    λ = 0,5
                     Σ_t  ω_t(m)

Ē_fis(v, p)      =  (1/M) · Σ_{i=1..M}  E_fis(v, p ; m_i)          plausibilità: media su M maschere casuali

   c_t        confidenza del passo: media della presenza dei 69 giunti
   n^m_t      token mascherati nei riquadri del passo
   w^v_t      somma dei pesi dei token visibili nei riquadri (1 ciascuno nel cooldown, o 1/√d_min)
```

- **I pesi di V-JEPA 2.1, portati sul passo [Nostra scelta, adattamento di Lett. 32].** Ogni token mascherato nei riquadri conta una volta, ogni token visibile `λ` volte il suo peso: è la media di V-JEPA 2.1 su `L_pred + λ · L_ctx`. La confidenza `c_t` è l'unico peso nostro. Formule complete in `worldsign-loss.md` §3.
- **In addestramento** si minimizza `E_fis` sui dati del training set, mediata sulle due maschere del passo (brevi e lunghe), come in V-JEPA.
- **In valutazione** la plausibilità di una sequenza è la sua energia, **mediata su più maschere**: una sola maschera potrebbe non coprire proprio la parte anomala **[Nostra scelta]**.
- **Cosa misura [Nostra osservazione].** Le maschere a tubi coprono tutta la clip nel tempo, quindi l'energia valuta la coerenza **usando contesto prima e dopo**: quanto una sequenza è coerente dentro la finestra, non quanto il futuro sia prevedibile dal solo passato. È coerente con l'ablazione di §4.5.2.
- **Perché funziona senza negativi** [Lett. 28]. In un EBM regolarizzato l'energia viene abbassata solo sulle configurazioni osservate. Il volume a bassa energia è limitato dalla regolarizzazione (SIGReg, rango della LoRA), quindi una configurazione mai vista resta ad energia alta **senza che il modello abbia mai visto un negativo**. È il meccanismo con cui V-JEPA mostra più «sorpresa» davanti agli eventi implausibili [Lett. 33].

**I termini della posa** (`worldsign-posa.md` §4): l'encoder di posa si addestra con la sua invarianza, l'ancora di ricostruzione e SIGReg (§4.5.6).

```
L_inv     =  media sui passi validi di  ¼ Σ_v (1/C) ‖ μ_t − z_{v,t} ‖²            z_v: le 4 viste (pulita e tre di disturbi); μ_t: il loro centro

               Σ_t Σ_j  c_{t,j} · ‖ D( s_t )_j − p̂_{t,j} ‖²
L_anchor  =  ──────────────────────────────────────
                σ² · Σ_t Σ_j  c_{t,j}
```

`D` è un solo decoder lineare dal vettore del passo ai 69 giunti; `p̂_{t,j}` è il keypoint mediato sui due frame del tubelet; `σ²` la varianza dei keypoint sulle clip di training, così che predire la media valga 1. Il peso `c_{t,j}` impedisce di imparare il rumore dello stimatore di posa **[Nostra scelta]**, ispirato al campionamento guidato dalla confidenza di Uni-Sign [Lett. 8].

#### 4.5.4 Livello semantico: energia video–testo

```
ŷ(v)         =  Proj( media sulle 8 query di  Pred_sem( Enc_v(v) , q_1 … q_8 ) )
ẽ(c)         =  Testa( centratura( EmbeddingGemma(c) ) )

E_sem(v, c)  =  D( ŷ(v) , ẽ(c) )          D = 1 − cos( ŷ , ẽ )  nei bracci A₀, A, B di ESP-1
```

**Uso dell'energia** [Lett. 34: classificazione e retrieval scelgono il candidato più vicino alla predizione]:

```
testo → video      v*          =  argmin su v nella galleria      E_sem(v, c)
video → testo      c*          =  argmin su c fra i candidati     E_sem(v, c)
riconoscimento     etichetta*  =  argmin sulle etichette          E_sem(v, testo dell'etichetta)
```

**Perché le coppie sbagliate restano ad energia alta senza negativi [Nostra argomentazione, su Lett. 35].**

```
se SIGReg rende ŷ ed ẽ circa N(0, I_d):  ogni proiezione ha varianza 1,  quindi  media ‖ŷ‖² ≈ media ‖ẽ‖² ≈ d

coppia NON corrispondente, circa indipendente:     media ‖ŷ_i − ẽ_j‖²  ≈  d + d  =  2d      cos ≈ 0 (± 1/√d)
coppia corrispondente, dopo l'addestramento:       ‖ŷ_i − ẽ_i‖²  =  ε · d,  ε piccolo      cos ≈ 1 − ε/2

divario di energia con D = 1 − cos:   ≈ 1 − ε/2   senza aver mai visto un negativo
```

Il «circa» conta: didascalie dal contenuto simile non sono indipendenti. La scala `d` non cambia il ragionamento, perché `D` usa il coseno: conta che le coppie indipendenti abbiano coseno vicino a 0, cioè l'isotropia.

**Il rischio principale: predire la media [Lett. 81; Nostra argomentazione].** Se la clip non determina del tutto la didascalia, una loss di regressione è minimizzata dalla **media** delle didascalie compatibili, che può non somigliare a nessuna. È coerente con l'ablazione di VL-JEPA, dove le loss di regressione perdono molto nel retrieval (§2.3) [Lett. 34]. Per questo ESP-1 confronta anche un termine di uniformity esplicito (B) e InfoNCE (C), e per questo esiste l'ablazione con variabile latente (§4.5.9).

**Profilo temporale, solo in valutazione e a costo zero in addestramento [Nostra proposta]:**

```
profilo(t)  =  E_sem( v[t, t+L] , c )          finestra scorrevole su un video lungo
```

Il minimo indica **dove** la frase viene segnata. L'allineamento fra segnato e sottotitoli è un compito studiato su BOBSL [Lett. 9]; usarne l'energia è **[Nostra ipotesi]**.

#### 4.5.5 Perché al livello lessicale non c'è allineamento

Per imporre «stesso segno → stesso punto» servirebbero **(a)** una segmentazione dei confini dei segni e **(b)** una mappa da ogni segmento al lessema che rappresenta **[Nostra formalizzazione]**. Il corpus continuo non fornisce nessuna delle due: la didascalia indica il significato di ~5 secondi ma non quale intervallo corrisponda a quale parola, e l'ordine dei segni non coincide con quello delle parole.

Alternative valutate e **non adottate**:
- confini ricavati dalla cinematica [Lett. 71] più clustering alla SHuBERT [Lett. 18] — una softmax su un codebook è di fatto un termine discriminativo, in tensione con la tesi **[Nostra osservazione]**;
- le annotazioni automatiche di BOBSL [Lett. 10] e le gloss dei benchmark serviranno **solo per valutare** se una struttura lessicale emerge **[Nostra scelta]**;
- la segmentazione agglomerativa di VL-JEPA [Lett. 34] resta uno strumento possibile per lavori futuri.

#### 4.5.6 SIGReg

```
per M direzioni casuali  v_m  (‖v_m‖ = 1), ricampionate a ogni passo:

   u_m      =  { ⟨ v_m , x_n ⟩  :  n = 1 … N }                                 proiezioni 1-D del batch
   EP(u_m)  =  N · ∫_{−5}^{5} | φ̂_{u_m}(τ) − e^{−τ²/2} |² · e^{−τ²/2} dτ         statistica di Epps–Pulley

   SIGReg(X)  =  (1/M) · Σ_m  EP(u_m)                                          (nella forma di [Lett. 35])
```

- `φ̂_u(τ) = (1/N) Σ_n e^{iτu_n}` è la funzione caratteristica empirica; `e^{−τ²/2}` è quella della normale standard, ed è anche il peso dell'integrale. L'integrale si calcola con la regola dei trapezi su 17 nodi in `[−5, 5]`; `M = 1.024` direzioni, come raccomanda LeJEPA [Lett. 35].
- **Il fattore N** è quello di LeJEPA: per un campione davvero gaussiano il valore atteso resta ≈ 1,06 (`√(2π) − √(2π/3)`) qualunque sia `N`, mentre per ogni altra distribuzione cresce con `N`. Vale per campioni indipendenti. È la scala a cui si riferisce λ [Lett. 35]. Con più GPU, `N` conta i campioni di tutte le GPU.
- **Applicato con un λ per livello** **[Nostra scelta]**:
  - al bersaglio di posa, **separatamente per vista e per passo**, come LeWorldModel, in **tutti** i bracci: `SIGReg_posa` = media sulle 4 viste e sui 32 passi di SIGReg delle clip presenti a quel passo, con λ = 0,04 (`worldsign-posa.md` §4.2);
  - a **ciascuna modalità separatamente**, con le stesse direzioni: `SIGReg_sem = ½ · [ SIGReg({ŷ}) + SIGReg({ẽ}) ]`, nei bracci **A, B e C** di ESP-1 (non in A₀ e B₀, §4.5.7). È come LeJEPA lo applica, a ogni vista separatamente [Lett. 35].

**Perché per modalità e per vista [Nostra argomentazione].** SIGReg garantisce qualcosa solo sulla distribuzione su cui è calcolato [Lett. 35]: sull'unione, `ŷ` ed `ẽ` potrebbero compensarsi a vicenda; separatamente, ciascuna ha la garanzia piena. Il bersaglio di posa è un vettore per passo che concatena i quattro articolatori (§4.4.3): non ci sono più insiemi per articolatore.
- **Tutto il batch effettivo per valutazione**, raccolto su tutte le GPU: 128 campioni per modalità in `SIGReg_sem`, fino a 128 clip per vista e passo in `SIGReg_posa`. Il bias di minibatch è `O(1/N)`; 128 è il batch più piccolo testato da LeJEPA, ancora competitivo [Lett. 35] **[Nostra scelta]**.

#### 4.5.7 Obiettivo complessivo e bracci di ESP-1

```
L  =  (1 − λ_P) · ( L_inv + L_anchor )  +  λ_P · SIGReg_posa                    λ_P = 0,04
   +  (1 − λ) · ( E_fis  +  L_pred_sem )  +  λ · SIGReg_sem                    λ = 0,05
```

`L_pred_sem` e `SIGReg_sem` dipendono dal braccio di ESP-1:

```
       L_pred_sem              SIGReg_sem
A₀     E_sem                   —                         solo allineamento
A      E_sem                   ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
B₀     E_sem + L_unif          —                                  L_unif come in [Lett. 40]
B      E_sem + L_unif          ½ [SIGReg({ŷ}) + SIGReg({ẽ})]
C      L_InfoNCE(ŷ, ẽ)         ½ [SIGReg({ŷ}) + SIGReg({ẽ})]      batch 128; negativi dello stesso video esclusi
```

Tutti i bracci usano lo stesso batch effettivo di 128 clip, InfoNCE compreso (§4.14).

**Pesi presi dalla letteratura, nessuna calibrazione.**

| Scelta | Evidenza |
|---|---|
| λ = 0,05 fra termini predittivi e SIGReg | è la forma della loss di LeJEPA; λ = 0,05 è un *«default robusto»* e le prestazioni sono *«stabili al variare di λ»* [Lett. 35]; nel paper vale per 8–10 viste, da rivedere con il livello semantico |
| λ_P = 0,04 sulla posa | lo 0,02 di LeJEPA per 4 viste a 256 campioni, raddoppiato per i ≤ 128 campioni per passo; dentro la zona stabile di LeWorldModel (`worldsign-posa.md` §4.2) |
| Pesi dentro `E_fis` | quelli di V-JEPA 2.1 (§4.5.2) [Lett. 32] |
| Pesi uguali fra i termini predittivi | sommare le loss con pesi uguali *«eguaglia o supera gli ottimizzatori multi-task complessi»* [Lett. 96] |

- **Con l'addestramento per livello i pesi fra livelli non contano [Nostra argomentazione, 3/10].** Ogni parametro riceve il gradiente di un solo livello, e con Adam la scala di una loss non cambia l'aggiornamento dei parametri che ricevono solo quella (`worldsign-gerarchia.md` §3). Contano solo i rapporti dentro un livello: invarianza, ancora e SIGReg sulla posa; `E_sem` e SIGReg semantico.
- **Condizione perché i pesi uguali abbiano senso dentro un livello [Nostra argomentazione]:** scale confrontabili per costruzione. `L_inv` è di ordine 1 con `s` vicino a `N(0, I)`; `L_anchor` si divide per la varianza dei keypoint, così che predire la media valga 1; `E_sem = 1 − coseno` sta fra 0 e 2.

Il livello della posa e quello fisico sono **identici in tutti i bracci**: i bracci differiscono solo nella loss semantica.

#### 4.5.8 Perché i termini non si ostacolano

- **Allineamento contro SIGReg [Nostra argomentazione].** `E_sem` riduce la varianza *dentro* le coppie, SIGReg controlla la varianza della distribuzione *marginale*. Sono compatibili perché le coppie semanticamente distinte sono molte più delle dimensioni (~4·10⁶ ≫ 512).
- **L'unico conflitto reale è l'anisotropia del target [Nostra argomentazione].** Se il target non è isotropo, i due termini hanno ottimi diversi. I modelli contrastivi moderni sono già isotropi [Lett. 51]; lo verifica PC1, e se il conflitto c'è lo segnala in addestramento il coseno fra i gradienti (§4.13).
- **Livello fisico contro livello semantico sull'encoder condiviso [Nostra argomentazione].** I due gradienti arrivano alla stessa LoRA da passaggi separati, quindi il loro coseno si misura a costo nullo. Un coseno stabilmente negativo indicherebbe che la posa e il significato chiedono all'encoder cose incompatibili (§4.13).
- Il testo resta **precalcolato e congelato**: il suo contributo passa solo dalla testa MLP.

#### 4.5.9 La variabile latente (solo nell'ablation facoltativa ESP-4)

**Nelle run generiche non c'è variabile latente** **[Nostra scelta]**. Viene provata **una volta**, sulla configurazione migliore di ESP-1 (θ\*), se i tempi lo consentono: ESP-4 è **facoltativa** (§4.14).

**Due cose diverse chiamate «z».**

| Modello | Variabile | Nota o latente | Cosa dice al predictor |
|---|---|---|---|
| I-JEPA | token di maschera + posizione | nota | **dove** predire [Lett. 29] |
| VL-JEPA | query X_Q | nota | **cosa** predire [Lett. 34] |
| EBM a variabile latente | z | **latente** | l'informazione sul bersaglio che non si ricava dall'input [Lett. 28] |
| Nostro livello fisico | posizioni della maschera | nota | dove |
| Nostro livello semantico | query costanti | nota, banale | nulla: sempre la didascalia |

**A cosa servirebbe [Lett. 28, 81].** Un predictor deterministico addestrato in regressione converge alla **media condizionata** e, con più esiti validi, *«media modi distinti»*.

```
esempio [Nostra illustrazione]: stessa clip, due traduzioni valide con embedding ẽ_A ed ẽ_B,  δ = ‖ẽ_A − ẽ_B‖²

SENZA z                ŷ = (ẽ_A + ẽ_B) / 2          →   E(v, A) = E(v, B) = δ / 4
CON z a due valori     ŷ_1 ≈ ẽ_A ,  ŷ_2 ≈ ẽ_B       →   F(v, A) ≈ 0 ,  F(v, B) ≈ 0

legge della varianza totale:
tr Cov(ẽ)  =  tr Cov( E[ẽ | v] )   +   media su v di  tr Cov(ẽ | v)
              parte predicibile         ambiguità: errore minimo di un predictor deterministico
```

**Il pericolo: la capacità** [Lett. 28]. Se `z` può valere qualunque vettore, il predictor impara a copiarla e l'energia diventa nulla per **qualunque** didascalia. Nel retrieval la `z` agisce da entrambi i lati: abbassa l'energia della traduzione giusta ma atipica, e anche quella delle didascalie sbagliate.

```
caso idealizzato [Nostra stima]: ipotesi indipendenti, SIGReg riuscito, d = 256
distanza quadra fra punti indipendenti:  media 2,  deviazione standard √(8 / d) ≈ 0,18

energia media di una coppia sbagliata, minimo su K ipotesi:
   K = 1  →  2,00        K = 2  →  ≈ 1,90        K = 4  →  ≈ 1,82
```

Con K piccolo l'energia delle coppie sbagliate cala poco; fra elementi semanticamente vicini della galleria la perdita di discriminazione può essere maggiore.

**Forma adottata per ESP-4 [Nostra scelta]:**

```
le 8 query del predictor semantico si dividono in K = 4 gruppi da 2  →  4 ipotesi ŷ_1 … ŷ_4

E_k(v, c)     =  D( ŷ_k(v) , ẽ(c) )
F_sem(v, c)   =  min su k di  E_k(v, c)                          energia libera, capacità log₂ 4 = 2 bit

addestramento   minimizzare F_sem, con rilassamento ε: le ipotesi perdenti ricevono una piccola frazione del gradiente
                + SIGReg per modalità:  ½ [ SIGReg({ ŷ_k }, tutte le ipotesi) + SIGReg({ ẽ }) ]
inferenza       retrieval = argmin di F_sem;  nel braccio C la similarità è il massimo sulle ipotesi, come in PVSE
```

- **Con K = 1 il modello coincide esattamente con la configurazione di riferimento**: l'ablazione isola la `z`, a **zero parametri aggiuntivi**.
- **Perché K = 4.** PVSE, con K embedding per istanza e perdita sulla coppia migliore, trova K > 1 migliore di K = 1 (MS-COCO R@1 da 66,7 a 69,2; TGIF da 2,82 a 3,28), con ottimo K = 3 su COCO e TGIF e K = 5 sul dataset più ambiguo [Lett. 80]. K = 4 sta nel mezzo e vale 2 bit. PVSE è però contrastivo: da noi il minimo è **fra le ipotesi della stessa coppia**, non fra campioni diversi, e resta coerente con l'assenza di negativi **[Nostra argomentazione]**.
- **Rilassamento ε.** Senza di esso, se all'inizio tutti i bersagli cadono nella stessa cella di Voronoi, si aggiorna una sola ipotesi [Lett. 81].

**Altre forme di z, escluse [Nostra argomentazione]:**

| Forma | Motivo dell'esclusione |
|---|---|
| continua, trovata ottimizzando `z` per ogni coppia | nel retrieval servirebbero N_video × N_testi ottimizzazioni |
| calcolata dalla didascalia (stile VAE) | la `z` vede la risposta: senza un vincolo forte l'energia si appiattisce [Lett. 28] |
| embedding probabilistici | modellano l'incertezza con distribuzioni [Lett. 85], ma cambiano quadro: niente più distanza fra punti, e SIGReg andrebbe ripensato |

**Al livello fisico nessuna variabile latente**, come in V-JEPA 2.1 **[Nostra scelta]**.

### 4.6 Prevenzione dei leak

| Rischio | Contromisura |
|---|---|
| L'encoder vede i token che deve predire | i token mascherati vengono **rimossi** dall'input dell'encoder, non azzerati, come in V-JEPA [Lett. 30] |
| La posa entra nel contesto | la posa alimenta **solo** l'encoder del target; nessun percorso la collega all'input del predictor (P14) |
| Il riquadro di lettura rivela dove si trova l'articolatore nascosto | il riquadro viene dai keypoint del bersaglio: il predictor **non è costretto a localizzare** l'articolatore mascherato, solo a descriverlo. **[Aperto]** Si misura con l'energia a riquadri spostati (§4.13); se l'allarme scatta, si aggiunge un termine che penalizza le predizioni «da articolatore» fuori dal riquadro |
| Falsi negativi nel braccio InfoNCE | clip dello stesso video escluse dal denominatore (P15) |

**Perché questi controlli [Nostra argomentazione].** Se l'encoder vede i frame di cui deve predire la posa, il compito diventa stima di posa, con contenuto fisico nullo, e **la loss scende comunque**: l'errore è silenzioso.

### 4.7 Rapporto con H-JEPA, LV-EBM, V-JEPA 2.1 e VL-JEPA

| Elemento | H-JEPA [Lett. 27] | Questo progetto |
|---|---|---|
| Livelli | JEPA impilate | **tre livelli**: posa (bersaglio), fisico, semantico; il semantico legge l'encoder adattato dal fisico |
| Predictor per livello | sì | sì: fisico e semantico |
| Predizione a tutti i livelli | sì | sì |
| Livello basso dettagliato, alto astratto | sì | fisico denso per passo; semantico globale per clip |
| Addestramento | per livello o globale [Lett. 27] | **per livello**, con stop-gradient fra i livelli; «globale» come ablation |
| Variabile latente | sì | solo nell'ablation facoltativa ESP-4 |
| Variabile azione e pianificazione | sì | **no**: non agiamo sul mondo |

| Elemento | LV-EBM [Lett. 28] | Questo progetto |
|---|---|---|
| Energia scalare | sì | `E_fis` (video–posa), `E_sem` (video–testo) |
| Inferenza per minimizzazione | sì | sulla collezione (retrieval = `argmin` di `E_sem`); sul latente solo in ESP-4 |
| Anti-collasso | contrastivo o regolarizzato | **regolarizzato** (SIGReg) |
| Energia calcolata da una rete congiunta su (x, y) | possibile | **no**: da noi l'energia è una distanza fissa fra embedding calcolati separatamente. Una rete che guarda video e testo insieme (cross-encoder) richiederebbe un passaggio per ogni coppia della galleria e, per non collassare, dei negativi [Lett. 89] (§2.1.2) |

| Elemento | V-JEPA 2.1 [Lett. 32] | Nostro livello fisico |
|---|---|---|
| Maschera | multi-blocco, ~90 % | **identica** |
| Loss | densa: mascherati + visibili vicini, L1 pesata per distanza | **stessi pesi**, letta per passo |
| Predizione multi-livello | 4 livelli in ingresso e in uscita (non nei modelli distillati) | **fusione dei 4 livelli in ingresso**, un'uscita verso la posa |
| Target encoder | EMA dello stesso encoder video | encoder di **posa** da zero, letto con lo stop-gradient e addestrato con i suoi termini, senza EMA |

| Elemento | VL-JEPA [Lett. 34] | Nostro livello semantico |
|---|---|---|
| X-Encoder | V-JEPA 2 ViT-L congelato | V-JEPA 2.1-L congelato + LoRA (PC3) |
| Predictor | ultimi 8 layer di Llama-3.2-1B (490 M) | 4 blocchi da zero (7,7 M) |
| Query | domanda o prompt | 8 query apprese costanti |
| Y-Encoder | EmbeddingGemma, LR ×0,05 | EmbeddingGemma congelato + testa MLP |
| Maschera | nessuna | nessuna |
| Loss | **InfoNCE** | **allineamento + SIGReg** (InfoNCE solo come braccio C) |


### 4.8 Parametri

Stime con formule standard: un blocco transformer con MLP 4× ha ≈ `12·d²` parametri; una matrice LoRA ne ha `r·(d_in + d_out)` **[Nostra stima]**.

| Componente | Inizializzazione | Congelati | Addestrabili | Tipo |
|---|---|---|---|---|
| Encoder video ViT-L | V-JEPA 2.1 | 300 M | — | — |
| ├ LoRA r = 16, 24 blocchi (attenzione e MLP) | — | — | 7,08 M | vincolato |
| └ LayerNorm e bias | — | — | 0,10 M | vincolato |
| Encoder di posa: 4 parti a 128, temporale a 512, `Linear(512 → 192)` | da zero (§4.4.3) | — | 8,03 M | libero |
| Predictor fisico, 12 blocchi, d = 384 | V-JEPA 2.1 **[Aperto: PC6]** | ~22 M | — | — |
| ├ LoRA r = 16 (12 blocchi) | — | — | 1,33 M | vincolato |
| ├ fusione multi-livello: 4 LayerNorm, Linear 4.096 → 1.024, Linear 1.024 → 384 | da zero, inizializzazione in §4.10 | — | 4,60 M | libero |
| └ testa di lettura per passo (`4·384 → C`, `C = 192`) | da zero | — | 0,30 M | libero |
| Predictor semantico: 4 blocchi d = 384, proiezioni `1024 → 384` e `384 → 512`, 8 query | da zero | — | 7,67 M | libero |
| Decoder dell'ancora `D` (`192 → 138`) | da zero | — | 0,03 M | libero |
| Testa testuale MLP, `768 → 512 → 512` | standard (stadio P) | — | 0,66 M | libero |
| EmbeddingGemma-300M | — | 0 in GPU (precalcolato) | 0 | — |
| **Totale** | | **≈ 326 M** | **≈ 29,8 M** | **8,5 vincolati · 21,3 liberi** |

**Margine di ~0,2 M rispetto al tetto di 30 M** (conteggio misurato in `worldsign-architettura.md` §9: 29.799.434, con `C = 192` dal 6/10). Conseguenze:

| Eventualità | Effetto sul totale |
|---|---|
| Ripiego del predictor fisico (1 blocco da zero, §4.4.5) | ≈ 22,0 M |
| Modulo specifico per lingua (§4.4.7) | a rango minimo, da contenere nel margine **[Aperto]** |
| Variabile latente (ESP-4, facoltativa) | **+0**: riusa le query esistenti |

**Tagli possibili**, se servisse: predictor semantico a 3 blocchi (−1,8 M), oppure a dimensione 256 (≈ −4,1 M). **Mai la LoRA [Nostra scelta].**

### 4.9 Costo computazionale

Unità: **1 = un forward dell'encoder ViT-L su tutti gli 8.192 token.** FLOP per forward ≈ `2 × parametri × token` più il termine d'attenzione `2 × strati × token² × d`; backward ≈ 2 × forward, stima prudente con LoRA, dove i pesi congelati non richiedono gradiente **[Nostra stima]**.

| Voce | Valore **[Nostra stima]** |
|---|---|
| Forward del passaggio fisico | encoder su ~820 token ≈ 0,06 · predictor fisico su 8.192 token ≈ 0,12 → **≈ 0,18** |
| Forward del passaggio semantico | encoder su 8.192 token = 1,00 · predictor semantico ≈ 0,04 → **≈ 1,04** |
| **Passo completo** | il passaggio semantico non fa backward nell'encoder (addestramento per livello): `3 × 0,18 + 1,00 + 3 × 0,04` ≈ **1,7**; l'encoder di posa è trascurabile **[Nostra stima, 3/10]** |
| Quota del passaggio semantico | **≈ 65 %** |
| Stadio P del curriculum (posa e semantico, §4.10) | ≈ 1,1 |
| ViT-L rispetto a ViT-B, a parità di token | ≈ 2,5× per passo (i predictor restano uguali) |
| Braccio C (InfoNCE) | stesso costo degli altri bracci: batch 128, senza GradCache (§4.14) |
| Risoluzione 384² (18.432 token) | ≈ 3,4× l'encoder a 256²: il termine d'attenzione cresce col quadrato |
| Ramo testuale | ~0: embedding precalcolati |
| Fusione multi-livello | trascurabile: un MLP sui ~820 token visibili |
| **Durata di una run su 2 H100 da 80 GB**, 6 epoche senza early stopping (stima del 29/9; con 15 epoche e il passo a ≈ 1,7 i tempi vanno moltiplicati per ≈ 1,15, §4.10) | Con utilizzo del calcolo al 30–40 %: ViT-B ≈ 3,5–6 giorni (stima centrale ≈ 4,6) · ViT-L ≈ 9–15 giorni (≈ 11,6). **Calibrazione su una run precedente** sulle stesse 2 GPU (ViT-L a 384², 64 frame, batch 512, ~312 s per passo): utilizzo effettivo del calcolo ≈ 10 %. Le metriche di sistema di quella run (102 ore) mostrano GPU attive ~90 % del tempo ma memoria occupata al 28 %, micro-batch da 4 clip con GradCache (che aggiunge un passaggio in avanti), ricalcolo delle attivazioni su tutti i blocchi e CPU al 5 %: il limite è il modo in cui la GPU viene usata, non il caricamento dei dati. A quell'efficienza i tempi vanno moltiplicati per ~3,8. Ipotesi: circa 989 TFLOPs di picco in bf16 per GPU (valore usato da torchtitan per H100, da verificare), activation checkpointing, scalatura quasi lineare su 2 GPU perché la LoRA rende minima la sincronizzazione dei gradienti |

**Il passaggio semantico non mascherato domina il costo.** Se la misura nella dry run (PC7) sfora il budget, c'è un **ripiego**: scartare **token casuali al 50 %** nel solo passaggio semantico, per efficienza e non come compito predittivo. Il costo del passo scende a ≈ 1,2 (era ≈ 1,8 su 3,7 prima che il passaggio semantico perdesse il backward nell'encoder).
- Il precedente: in CLIP, scartare il 50 % dei patch dà ~2× di velocità e ~+1 % di accuratezza [Lett. 79]; nella fase video–testo di UMT si maschera l'80 % dei token video [Lett. 78].
- **Cautela [Nostra ipotesi]:** entrambi sono contrastivi, dove conta solo l'ordinamento relativo; nella nostra regressione il rischio di predire la media è maggiore.
- **Token casuali e non blocchi [Nostra argomentazione]:** lasciano ogni articolatore parzialmente visibile nei frame, mentre i tubi a blocchi possono cancellare una mano per tutta la clip.

Usiamo come unità di budget **L = un addestramento completo su ViT-L, braccio A** (§4.14). Il valore in GPU-ore si misura nella dry run (PC7).

### 4.10 Ricetta di addestramento e curriculum

Valori **di partenza**, da calibrare nella dry run **[Nostra scelta]**:

| Voce | Valore |
|---|---|
| Ottimizzatore | AdamW, β = (0,9; 0,999), come V-JEPA 2.1 [Lett. 32] |
| Weight decay | 0,04 costante, come V-JEPA 2.1 [Lett. 32], su matrici LoRA e moduli liberi; 0 su norme, bias, posizioni ed embedding |
| **Epoche** | **15 epoche** sul corpus con didascalie, nelle proporzioni naturali fra lingue (§3.5) **[Nostra scelta, 3/10]**. I video senza didascalia affidabile entrano solo nel passaggio fisico |
| **Early stopping** | **pazienza di 3 epoche, contata nello stadio F** (prima l'encoder non è ancora adattato). La metrica per decidere (media di R@1 T2V e V2T sullo split held-out channel) si valuta a fine epoca; se non migliora per 3 epoche consecutive, la run si ferma e si fa il **cooldown dal checkpoint migliore** (riga «Schedule») **[Nostra scelta, 3/10]** |
| **Batch effettivo** | **128 in tutti i bracci**, InfoNCE compreso (revisione del 29/9, §4.14). 128 è il batch più piccolo testato da LeJEPA, ancora competitivo [Lett. 35], ed è il numero di video per batch di V-JEPA 2.1 [Lett. 32] |
| Micro-batch per SIGReg | 128 clip, uguale in tutti i bracci: 128 campioni per modalità (`{ŷ}` e `{ẽ}` separatamente) e fino a 128 × 32 passi per vista della posa |
| **Passi** | al massimo, senza early stopping: 15 × ~3,2 M ≈ 48 M clip viste → **≈ 375.000 passi** a batch 128 **[Nostra stima, da ricalcolare quando la cardinalità del dataset è definitiva]** |
| Learning rate | ricerca su {1e-4, 2e-4, 5e-4} nella dry run (PC7), a batch 128; **3e-4 per l'encoder di posa** (`worldsign-posa.md` §4.3) |
| **Schedule** | **un gruppo per famiglia** (`worldsign-gerarchia.md` §6.2): ogni famiglia tranne la posa sale linearmente per **2 epoche** dalla sua entrata, poi resta costante; la posa sale sul **20 %** della run, poi scende con un **coseno** a 0. Su tutto, il **cooldown di V-JEPA 2** [Lett. 31]: lineare a 0 sul 5 % dei passi. **Con l'early stopping** il cooldown non resta fissato in fondo: si riparte dal checkpoint migliore e si fa lì; lo schedule lo consente, perché *«si possono avviare più cooldown da checkpoint diversi della fase costante»* [Lett. 31] |
| Peso λ della loss sui token visibili | 0,5 con warm-up progressivo, come V-JEPA 2.1 [Lett. 32] |
| **Durata degli stadi** | P = epoca 1 · F₀ = epoca 2 · F = dall'epoca 3 **[Nostra scelta, 3/10]** |
| Validazione e checkpoint | ogni ~500.000 clip viste (4.000 passi a batch 128), su 2.000 clip dello split held-out channel |
| Precisione | bf16, pesi master in fp32 per le LoRA |
| Clipping del gradiente | nessuno, come V-JEPA 2.1 [Lett. 32] |
| Media esponenziale dei pesi | **nessuna**, né come target né per la valutazione |
| λ | **λ = 0,05** fra termini predittivi e SIGReg, come LeJEPA [Lett. 35], e **λ_P = 0,04** sulla posa (`worldsign-posa.md` §4.2); **pesi uguali** fra i termini predittivi [Lett. 96] (§4.5.7) |
| Braccio InfoNCE | temperatura apprendibile (init 0,07); batch 128, come gli altri bracci |
| ESP-4 (facoltativa) | frazione ε del rilassamento **[Aperto: PC7]** |

**Curriculum** [Nostra scelta, 3/10] (`worldsign-gerarchia.md` §6):

```
Stadio 0    precalcolo: selezione dei 64 frame, pose, embedding testuali, medie per lingua
Stadio P    epoca 1         posa + semantico: encoder di posa, predictor semantico, testa del testo; LoRA ferme
Stadio F₀   epoca 2         + livello fisico: fusione e testa di lettura; LoRA ancora ferme
Stadio F    dall'epoca 3    tutto: anche la LoRA del predictor fisico e dell'encoder video
Stadio 3    solo sul modello finale: fine-tuning su OpenASL, PHOENIX-2014T, CSL-Daily; poi SLT, SLR, SLP
```

**Perché lo stadio P [Nostra argomentazione].** La LoRA non deve inseguire un bersaglio di posa ancora casuale; il semantico parte subito, su V-JEPA 2.1 pre-addestrato, come VL-JEPA [Lett. 34].

**Perché lo stadio F₀** [Lett. 95]. Addestrare insieme una testa casuale e il corpo pre-addestrato *«distorce le feature pre-addestrate»*; addestrare prima la sola testa dà +1 % in distribuzione e +10 % fuori distribuzione rispetto al fine-tuning completo. Stadi da un'epoca hanno precedenti in ULMFiT e LLaVA (`worldsign-gerarchia.md` §6.1).

**Inizializzazione.**
- LoRA con B = 0 e A gaussiana: al passo 0 ogni matrice adattata coincide con quella pre-addestrata [Lett. 67].
- Token di maschera a zero, come nella configurazione di V-JEPA 2.1 [Lett. 72].
- Fusione multi-livello: se PC6 lo consente, inizializzata in modo da riprodurre il predictor distillato sull'ultimo livello, con i pesi degli altri tre livelli a zero; altrimenti inizializzazione standard, protetta dallo stadio F₀ **[Nostra scelta]**.
- Nessuna calibrazione dei pesi delle loss al passo 0: con B = 0 il gradiente sulle matrici A è nullo e le teste nuove sono casuali, quindi le norme misurate in quel momento non rappresentano l'importanza dei termini **[Nostra argomentazione, su Lett. 67]**.

### 4.11 Regolarizzazione contro l'overfitting

| Misura | Motivo |
|---|---|
| Encoder video e predictor fisico congelati + LoRA | il vincolo più forte sulla capacità |
| Tetto di 30 M di parametri addestrabili (≈ 29,8 M usati) | §3.4, `worldsign-posa.md` §3 |
| Stop-gradient fra i livelli; coseno sul learning rate della posa | il video non sposta il suo bersaglio, che rallenta mentre il video lo insegue |
| Ancora di ricostruzione dei keypoint, invarianza fra viste | il target resta fedele al corpo e robusto al rumore del rilevatore |
| Stochastic depth e dropout nel predictor semantico e nelle teste | regolarizzazione standard dei moduli da zero |
| LayerScale nei rami residui del predictor semantico | stabilità con pochi dati |
| Weight decay selettivo | |
| **Early stopping sullo split held-out channel**, pazienza 3 epoche nello stadio F | misura la generalizzazione a segnanti mai visti |
| SIGReg | vincola la distribuzione degli embedding |
| Augmentation (§3.8) | |

**Nota [Nostra argomentazione].** Con un encoder congelato il rischio speculare è l'**underfitting** rispetto al cambio di dominio (mani piccole e veloci, lontane dai video su cui è stato addestrato). Per questo **non si riduce la LoRA per contrastare l'overfitting**.

### 4.12 Valutazione

#### 4.12.1 Controlli preliminari (prima di addestrare)

| ID | Controllo | Cosa decide |
|---|---|---|
| **PC1** | Geometria dei target testuali a 768 dimensioni, con il prefisso fissato e dopo centratura per lingua: effective rank [Lett. 43], spettro, IsoScore [Lett. 44], coseno medio fra coppie casuali, **valore di SIGReg** | nessuna dimensione da scegliere: il vettore si usa intero (§4.4.4). Misurato il 21/9: IsoScore 0,26, effective rank 621, SIGReg ≈ 14 volte il valore di una gaussiana. Se SIGReg sui target centrati resta alto, conflitto di anisotropia da segnalare prima del gate |
| **PC2** | Baseline: casuale; solo statistiche della didascalia; **regressione ridge da feature V-JEPA congelate a embedding testuali**. Tutte misurate **nelle due direzioni** (R@k, Precision@k, Recall@k, MRR, MedR); la penalità della ridge si sceglie sulla media dei due R@1 **[Nostra scelta, 4/10]** | la soglia minima da battere: **una per direzione**, il miglior R@1 T2V e il miglior R@1 V2T fra le baseline (usate dal gate e da F3) |
| **PC3** | 20.000 clip, feature congelate sull'**uscita dell'encoder** di V-JEPA 2.1-L distillato e di V-JEPA 2-L; tre probe lineari: keypoint delle mani dal riquadro (R²), proprietà fonologiche su segni isolati [Lett. 74–76], regressione ridge verso il testo (media di R@1 T2V e V2T, dal 4/10) | scelta dell'encoder video: vince il migliore su almeno 2 probe su 3; a parità, 2.1-L **[Nostra proposta]**. **Chiuso il 29/9: V-JEPA 2.1-L**, a parità sui due probe calcolabili; con la media il risultato non cambia (§4.4.1) |
| **PC4** | Gli stessi probe a 256² e a 384², entrambi con crop | risoluzione: 384² solo se il probe delle mani guadagna più di +0,05 di R², perché il costo dell'encoder triplica **[Nostra proposta]**. **Chiuso il 29/9: 256²**, guadagno +0,005 |
| **PC5** | Scelta dell'encoder di posa fra feature cinematiche, Uni-Sign congelato, MAMP e S-JEPA, sulle stesse clip: R² di una lettura lineare di posizioni e velocità per articolatore (**≥ 0,7**), rango effettivo e IsoScore, riconoscimento di canale e lingua | chiuso il 2026-09-20: S-JEPA. **Superato il 3/10:** l'encoder di posa si addestra da zero (§4.4.3) |
| **PC6** | I checkpoint 2.1-L e 2.1-B contengono i pesi del predictor? Dimensioni d'ingresso e d'uscita? Struttura della proiezione d'ingresso e presenza delle LayerNorm per livello nell'encoder? Con i pesi dei tre livelli aggiuntivi a zero, il predictor con fusione dà la stessa uscita di quello distillato? | **riuso del predictor fisico o ripiego** (§4.4.5); innesto della fusione multi-livello |
| **PC7** | **Dry run**: 1 % dei video (~32.000 clip), architettura completa su **ViT-L**, tutta la diagnostica ogni 50 passi; run brevi (~2.000 passi) con learning rate 10⁻⁴, 2·10⁻⁴ e 5·10⁻⁴ a batch 128, l'unico del piano (§4.14); **misura del costo per passo** (stima 3,7) | learning rate (loss di validazione più bassa senza instabilità), soglie degli allarmi rispetto al loro rumore naturale, valore di L, durata degli stadi, eventuale ripiego sui token del passaggio semantico |
| **PC8** | **Tolto il 29/9** (piano solo ViT-L, §4.14). Era: tre run ViT-B identiche tranne il seed, per stimare `σ_seed` | senza `σ_seed` gli effetti si leggono con gli intervalli bootstrap; **[Aperto]** un secondo seed di θ\* |

#### 4.12.2 Gate (criterio fissato in anticipo)

**Revisione del 29/9 [Nostra scelta].** Senza la scala ViT-B, la run di gate è la **run 1 del piano (§4.14): architettura completa, ViT-L, braccio A** (allineamento + SIGReg, la configurazione della tesi), sul corpus di pretraining. Test diretto su OpenASL, **senza fine-tuning**: il confronto è quindi prudente. Si legge R@1 **in entrambe le direzioni**, ognuna contro i propri riferimenti: C²RL dà T2V = 62,2 e V2T = 61,6, ottenuti con fine-tuning [Lett. 87]; la baseline ridge di PC2 è quella della stessa direzione. **Decide la direzione peggiore** **[Nostra scelta, 4/10]**: il modello serve al retrieval nei due versi, e una direzione buona non deve nasconderne una cattiva.

**Quando partono le altre run.** Gli altri quattro bracci di ESP-1 partono in parallelo quando la run di gate supera la fermata F2 (§4.13.5); se la run di gate si ferma a F3, si fermano anche loro. ESP-2, ESP-6 ed ESP-4 (facoltativa) partono alla fine di ESP-1, sulla configurazione migliore θ\*. Un problema dell'architettura costa così una run sola.

| Esito finale (F4) | Decisione |
|---|---|
| in almeno una direzione, sotto la baseline ridge di PC2 di quella direzione | **stop**: bug o obiettivo controproducente |
| in almeno una direzione, < 0,5 × C²RL di quella direzione (T2V ≈ 31,1, V2T = 30,8) | **stop**: il problema non è risolvibile con aggiustamenti |
| in entrambe le direzioni, ≥ 0,5 × C²RL | si procede: confronto dei bracci e modello finale |

**Soglia `X = 46,5` [Nostra scelta, fissata il 2026-09-21, rivista il 4/10].** Era il secondo gate prima della riga ViT-L di ESP-1. Senza quella riga non c'è più un blocco di budget da proteggere, e `X` resta il criterio della fermata F3: la curva estrapolata della **metrica che decide**, la media di R@1 T2V e V2T, deve essere compatibile con `X`. Perché il confronto sia omogeneo, `X` si calcola sulla media di C²RL: 0,75 × 61,9 = 46,425, arrotondato per eccesso. Prima valeva 0,75 × 62,2 = 46,65 → 46,7, sulla sola direzione T2V.

**Come si legge [Nostra scelta].** La decisione usa la stima puntuale; accanto si riporta l'intervallo bootstrap sulle query (§4.12.3). Le soglie sono codificate in `signworld/experiment/evaluation/gate.py`, con test sui confini, così nessun risultato può spostarle dopo. Il gate è un criterio di sanità, non una claim: il confronto con C²RL non è a parità di dati né di architettura (ResNet-18 a 224²) [Lett. 87].

#### 4.12.3 Protocollo finale

- **Retrieval** (modello finale, dopo fine-tuning): R@1/5/10, MRR e MedR, **T2V e V2T**, sui tre benchmark (§3.9), con intervalli bootstrap sulle query e, accanto, **R@1 tollerante ai duplicati** (§4.13.4) [Lett. 90]. Con un numero pari di query MedR è la media dei due ranghi centrali.
- **Densità delle rappresentazioni** con la metrica di SignCL [Lett. 17]: verifica se SIGReg risolve il problema che SignCL ha individuato.
- **Trasferimento a lingue con pochi dati** (richiede un pretraining in più: **non nel piano attuale**, §4.14): si esclude una lingua dal pretraining e poi si fa fine-tuning con 1, 5 e 20 ore **[Nostra scelta]**, al posto dello zero-shot che nel segnico non ha mai funzionato [Lett. 6, 16].
- **Identificazione della lingua** su rappresentazioni della posa, uscita dell'encoder video e `ŷ` (verifica della clessidra, H5).
- **Probe fonologici** sulle rappresentazioni della posa e sull'uscita dell'encoder video.
- **Coppie minime**: segni che differiscono per un solo parametro, da dataset di segni isolati usati **solo in valutazione** [Lett. 74–76] **[Nostra proposta]**.
- **Test di inversione temporale** e **test con il video sostituito da rumore** (R@1 nelle due direzioni; il calo si misura sulla loro media).
- **Hubness** delle due gallerie: quante volte ogni video compare fra i primi risultati delle didascalie (T2V), e ogni didascalia fra quelli dei video (V2T) [Lett. 83].
- **Profilo temporale di `E_sem`** su un sottoinsieme di BOBSL, come misura di allineamento ai sottotitoli (§4.5.4) **[Nostra proposta]**.
- **Metriche geometriche** (effective rank, IsoScore, condition number) **solo come diagnostica, non come obiettivo**: il legame con il retrieval cross-modale non è dimostrato [Lett. 43].

#### 4.12.4 Valutazione della fisica

**Test di violation-of-expectation specifico per il segnato** **[Nostra proposta]**, costruibile senza annotazione, letto sulla plausibilità `Ē_fis` confrontando la clip integra con quella manipolata:

| Manipolazione | Proprietà violata | Cosa si manipola | Atteso |
|---|---|---|---|
| Inversione temporale | freccia del tempo | video e posa, in modo coerente | `Ē_fis` sale |
| Frame saltati | continuità | video e posa | `Ē_fis` sale |
| Frame duplicati o ricampionati | inerzia | video e posa | `Ē_fis` sale |
| **Controllo:** posa sfasata nel tempo rispetto al video | coerenza fra flussi, **non** fisica | solo posa | `Ē_fis` sale; se non sale, il predictor non sta usando il video |
| **Controllo negativo:** luminosità e colore modificati | nessuna | solo video | `Ē_fis` resta entro la variabilità fra maschere; se sale, il test non è specifico |

Le manipolazioni si applicano al video nativo, **prima** della selezione dei 64 frame (§3.6), così da passare per la stessa pipeline dei dati veri **[Nostra scelta]**.

**Numero di maschere M [Nostra scelta].** Si sceglie il più piccolo M per cui la semi-ampiezza dell'intervallo di confidenza di `Ē_fis`, stimata dalla variabilità fra maschere, resta sotto un terzo dell'effetto atteso; lo si fissa in PC7.

**Nota [Nostra argomentazione].** Manipolare solo la posa, mentre il video mostra il movimento reale, misura un'incoerenza fra i due flussi, non la plausibilità fisica. Per questo resta solo come controllo. Test di **solidità** (una mano che attraversa la testa) richiederebbero video generato: sono fuori portata, e V-JEPA stesso non acquisisce la solidità [Lett. 33].

**IntPhys [Lett. 33].** Richiede un predictor che operi nello spazio dell'encoder. Con il 2.1-L distillato quello spazio è il teacher ViT-G, quindi **non è praticabile**. Lo diventa se PC3 porta a scegliere V-JEPA 2-L, che ha encoder e predictor nativi [Lett. 72]: si confronterebbe l'encoder prima e dopo la LoRA, con il predictor originale **[Aperto]**.

### 4.13 Sistema diagnostico «fail-fast»

**Obiettivo:** possiamo permetterci **al massimo una run fallita**. Ogni fallimento deve quindi emergere presto e indicare subito il componente responsabile. Ogni metrica si calcola **anche al passo 0**, come riferimento; le soglie sono **punti di partenza da calibrare** in PC7 **[Nostra scelta]**.

Quattro strati: collaudo prima di addestrare (§4.13.1) · asserzioni prima di lanciare (§4.13.2) · allarmi durante l'addestramento e diagnostiche dei risultati (§4.13.3–4.13.4) · run di gate con fermate programmate e tabella di triage (§4.13.5–4.13.6).

**Regola di costo [Nostra scelta].** Una diagnostica si calcola, quando possibile, da quantità già prodotte (loss per campione, embedding di validazione, gradienti già separati per passaggio). I forward aggiuntivi girano solo su piccoli sottoinsiemi e a cadenza bassa.

#### 4.13.1 Collaudo prima di addestrare

Nessuna ora-GPU di addestramento reale finché tutti i test non passano **[Nostra scelta]**.

| Test | Cosa verifica | Esito atteso | Costo **[Nostra stima]** |
|---|---|---|---|
| **Riproduzione dei pesi** | encoder video (LoRA a zero) ed EmbeddingGemma confrontati con le uscite del codice ufficiale sugli stessi input (20 clip, 1.000 didascalie); elenco delle chiavi del checkpoint non caricate | coseno medio per token > 0,999 e differenza massima < 10⁻³ in fp32; nessuna chiave mancante inattesa. Intercetta normalizzazione dei pixel, canali o frame in ordine sbagliato, RoPE mal configurata, pesi caricati a metà | < 1 ora |
| **Prompt di EmbeddingGemma** | la model card prescrive prefissi per compito, per esempio `task: search result \| query: …` e `title: none \| text: …` [Lett. 57]; se ne fissa uno, se ne salva l'impronta (hash) accanto agli embedding e si verifica che sia identica in PC1, addestramento e valutazione; si ricodificano 100 didascalie a caso (coseno > 0,9999 con quelle salvate) | stesso prefisso ovunque: un prefisso diverso cambia la geometria senza errori visibili **[Nostra argomentazione]** | minuti |
| **Allineamento video–posa** | 50 clip: i 64 frame si decodificano con il dataloader d'addestramento, si riesegue lo stimatore di posa su quei frame e si confronta con la posa salvata. Decoder diversi o video a frequenza variabile possono sfasare gli indici | errore medio dei keypoint delle mani < 2 % della larghezza del frame, senza sfasamenti sistematici **[Nostra proposta]**; altrimenti `E_fis` ha un pavimento alto e l'errore è silenzioso | < 1 ora |
| **Riquadri e normalizzazione della posa** | riquadri degli articolatori disegnati su 50 clip (controllo a occhio); frazione di riquadri fuori immagine; frazione di frame senza spalle affidabili; media, deviazione standard e massimo dei keypoint normalizzati | riquadri sulle mani nelle 50 clip; riquadri fuori immagine **entro un limite per articolatore** — mani ≤ 5 %, volto ≤ 1 %, corpo solo riportato, perché nelle inquadrature a mezzo busto gomiti e fianchi escono dal bordo per natura **[Nostra proposta, provvisoria]**; nessun NaN; keypoint entro 5 larghezze di spalle. Il limite unico dell'1 % falliva su OpenASL per il corpo (4,4 %) e per le mani (2,7 e 1,6 %): i valori si fissano sulle clip di test di YouTube-SL-25. Stessa soglia di confidenza della pipeline (1,0). Se le spalle mancano, o distano meno di metà della mediana della clip, si usa la scala dell'ultimo frame valido (§3.6) | < 1 ora |
| **Durate delle didascalie** | distribuzione della durata delle frasi, per corpus e per lingua | distribuzione misurata: determina la spaziatura massima dei 64 frame (≈ T/32) | minuti |
| **Frame selezionati contro frame contigui** | 200 clip con due input: i 64 frame selezionati (§3.6) e 64 frame contigui a passo fisso; per entrambi, posa estratta sugli stessi frame e regressione lineare dalle feature del riquadro della mano ai keypoint della mano | R² con i frame selezionati ≥ R² con i contigui − 0,05 **[Nostra proposta]**; altrimenti se ne discute prima del gate | ~1 ora |
| **Contaminazione e duplicati** | P2; stessa didascalia con la stessa durata in split diversi (ricaricamenti su YouTube) | zero sovrapposizioni | minuti |
| **Efficienza di calcolo** | tre misure durante 200 passi reali: (i) clip al secondo del solo dataloader; (ii) FLOP effettivi al secondo confrontati con il picco teorico delle GPU (utilizzo del calcolo, MFU); (iii) memoria GPU occupata e occupazione dei core di calcolo. La percentuale di nvidia-smi da sola non basta: in una run precedente era ~90 % con un utilizzo del calcolo di ~10 % (§4.9) | utilizzo del calcolo ≥ 30 % (i migliori addestramenti su larga scala stanno al 38–46 % [Lett. 97, 98]); memoria GPU ≥ 70 %; dataloader non limitante. Leve, in ordine: micro-batch grande quanto la memoria consente; ricalcolo delle attivazioni solo dove serve; bf16 ovunque; pre-estrazione dei frame solo se il dataloader risulta il limite. È il fattore che decide se una run dura settimane o mesi | ~1 ora |
| **Testare i test** | ogni metrica su casi sintetici con risposta nota: R@k = 1 su embedding identici e ≈ k/N su casuali; allarme di collasso su vettori costanti; SIGReg ≈ 0 su gaussiane, alto su vettori costanti e su miscele; test di leak con token mascherati lasciati apposta nell'input; `ω` su un modello a media semplice; test col rumore su un modello che ignora il video | ogni allarme scatta quando deve, e solo allora | minuti |
| **Stessi tensori** | R@k calcolato **esattamente** sui tensori della loss: `ŷ` dopo la proiezione, `ẽ` dopo la testa | coincidenza: una metrica calcolata su tensori diversi da quelli del punteggio è un errore silenzioso classico **[Nostra argomentazione]** | minuti |
| **Grafo e ottimizzatore** | dopo 2 passi ogni parametro addestrabile ha gradiente non nullo (le matrici A della LoRA lo ricevono solo quando B ≠ 0); i pesi congelati sono invariati; ogni famiglia del curriculum ha il suo gruppo e il suo schedule (`worldsign-gerarchia.md` §6.2) | tutto conforme | minuti |
| **Infrastruttura** | salvataggio e ricaricamento danno la stessa loss su un batch fisso; la ripresa conserva l'ordine del campionatore; lo stesso seed dà le stesse prime loss; parametri identici su tutte le GPU dopo 10 passi; SIGReg sui campioni raccolti da tutte le GPU uguale al calcolo su una sola GPU | tutto conforme | < 1 ora |
| **Mondo giocattolo** | 20.000 clip sintetiche con 4 forme colorate, una per «articolatore», che si muovono con inerzia e gravità e rimbalzano perdendo altezza; la «posa» sono centri e contorni delle forme; didascalie da modelli fissi («il cerchio rosso cade e rimbalza, il quadrato blu va a sinistra»); architettura completa su ViT-B per poche migliaia di passi | livello fisico migliore dell'interpolazione lineare sulle forme mascherate; R@1 > 20 % su 1.000 clip con combinazioni di movimenti mai viste; teletrasporto e video al contrario (rimbalzi che guadagnano altezza) alzano `Ē_fis`, un cambio di luminosità no. Verifica **l'intera catena** — maschere, 3D-RoPE, lettura, pesi della loss, due passaggi, SIGReg, valutazione — prima dei dati veri **[Nostra proposta]** | poche ore-GPU su ViT-B |
| **Overfitting controllato** | 64 clip reali con **tutte** le loss attive | `E_fis` ed `E_sem` → ~0; R@1 = 100 % su quelle clip; R² degli articolatori visibili > 0,95 | < 1 ora |

**Ordine e durata [Nostra stima]:** test automatici e misure ~1 giorno · mondo giocattolo e overfitting controllato ~0,5 · PC1–PC6 ~1 · dry run PC7 ~1 → **circa 3,5 giorni prima del gate**.

#### 4.13.2 Asserzioni prima di lanciare (se una fallisce, la run non parte)

| # | Asserzione |
|---|---|
| P1 | nessun canale in comune fra addestramento e validazione |
| P2 | nessun segmento di YouTube-SL-25 sovrapposto a una clip di validazione o di test di OpenASL; nessun video dei test set di PHOENIX-2014T e CSL-Daily nel corpus (§3.9) |
| P3 | parametri addestrabili tutti nel budget e sotto il tetto di 30 M (≈ 29,8 M) |
| P4 | i pesi congelati sono davvero congelati |
| P5 | checksum dei pesi pre-addestrati |
| P6 | i token mascherati sono **rimossi** dall'input dell'encoder; indici di contesto e di target disgiunti |
| P7 | statistiche della maschera: 8 blocchi brevi e 2 lunghi, rapporto medio ≈ 90 % |
| P8 | nessun flip orizzontale |
| P9 | posa nel formato dell'encoder di posa: 69 keypoint, 32 passi di due frame, normalizzazione e guardie di §3.6 |
| P10 | posa normalizzata: media ≈ 0, distanza fra le spalle ≈ 1 |
| P11 | target testuali a media ≈ 0 per lingua |
| P12 | trasformazioni geometriche applicate in modo identico a video e keypoint |
| P13 | overfitting di un singolo batch con SIGReg disattivato (se non riesce: bug nel grafo) |
| P14 | **la posa non raggiunge il predictor**: il gradiente dell'uscita del predictor fisico rispetto ai keypoint è nullo |
| P15 | nel braccio InfoNCE, nessuna coppia dello stesso video nel denominatore |
| P16 | SIGReg calcolato su tutto il batch effettivo: 128 campioni per modalità, fino a 128 × 32 passi per vista della posa |
| P17 | **i livelli si addestrano separati** (`worldsign-gerarchia.md` §3): `E_fis` non dà gradiente all'encoder di posa, i termini della posa non ne danno al ramo video, i termini semantici non ne danno alla LoRA video (salvo l'ablation «globale») |

#### 4.13.3 Allarmi durante l'addestramento

| Componente | Metrica | Allarme e significato | Costo aggiuntivo |
|---|---|---|---|
| **Leak della maschera** | sostituire il contenuto delle zone mascherate con un altro video non deve cambiare le predizioni | cambiano → **leak: stop** | minimo: 8 clip ogni 500 passi |
| **Collasso** | effective rank e deviazione standard per dimensione di `s`, dell'uscita dell'encoder video mediata, di `ŷ` | < 0,5× il passo 0; **si guarda prima `s`**, il target | nullo |
| **Predictor fisico** | `γ = Var(ŝ)/Var(s)`, `R²`, correlazione | `γ < 0,3` → predice la media; `R² > 0,98` subito → leak | nullo |
| **Predictor semantico** | `γ = tr Cov(ŷ)/tr Cov(ẽ)`; `R² = 1 − media ‖ẽ − ŷ‖² / tr Cov(ẽ)` per lingua, in train e validazione | `γ < 0,3` → media; `R²` basso **anche in train** → ambiguità o sottoadattamento | nullo |
| **Dinamica** | R² di `ŝ_t` sui passi soprattutto mascherati contro la baseline «media di `s_t` sui passi soprattutto visibili della stessa clip» | non la supera → nessuna dinamica appresa | nullo |
| **Localizzazione** (§4.6) | `E_fis` con il riquadro spostato in una zona mascherata dove l'articolatore non c'è | cresce meno del 10 % → predizioni non localizzate | nullo: solo una lettura in più |
| **Ordine temporale** | `ω = cos(ŷ(V), ŷ(V invertito))` | > 0,95 → predictor cieco all'ordine | basso: ogni 2.000 passi |
| **Deriva dell'encoder video** | R² con cui l'uscita adattata predice la parte delle feature V-JEPA originali **non spiegata dai keypoint** (feature originali salvate una volta) | < 0,5 → l'encoder si sta appiattendo sulla posa | una tantum + regressione ridge ogni 2.000 passi |
| **Encoder di posa** | `L_inv`, `L_anchor`, `SIGReg_posa`; effective rank, deviazione, IsoScore e SIGReg di `s` | rango o deviazione sotto 0,5× il passo 0 → collasso; SIGReg oltre 1,5× il passo 0 → il target lascia `N(0, I)` | nullo |
| Testa testuale | correlazione di Spearman fra le similarità prima e dopo la testa | < 0,8 → la testa sta alterando la semantica | nullo |
| **Conflitto fra loss** | coseno fra i gradienti di `E_sem` e di SIGReg rispetto a `ŷ`; coseno fra i gradienti dei due passaggi sulla LoRA video | < −0,3 per 500 passi → target anisotropo, oppure posa e significato in conflitto | nullo: gradienti già separati |
| LoRA | `‖ΔW‖ / ‖W‖` per blocco (video, predictor fisico) | crescita rapida, oppure > 0,1 | nullo |
| **Lingue piccole e sbilanciamento** | scarto train/val di `E_sem` per lingua in funzione delle epoche; R@1 ripartito per lingua (§4.13.4) | lo scarto cresce, o le lingue piccole restano molto indietro → lo sbilanciamento pesa: si cerca una soluzione (§3.5) | nullo |
| **Modality gap** | accuratezza di un classificatore logistico che distingue `ŷ` da `ẽ` | > 95 % → spazi separati | minimo |
| **Hubness** | asimmetria della distribuzione di quante volte ogni elemento della galleria compare nei primi k risultati, **per entrambe le gallerie**: i video per le query testuali (T2V), le didascalie per le query video (V2T) [Lett. 83] | cresce in T2V → `ŷ` schiacciate verso il centro, cioè verso la media; cresce in V2T → pochi `ẽ` attirano i video | nullo: dalla matrice di similarità di validazione |
| **Tabella energetica 2×2** | `E_fis` (una maschera) incrociata con `E_sem` in validazione | cresce «fisica bassa, semantica alta» → didascalie disallineate o semantica non appresa; cresce «fisica alta, semantica bassa» → errori dello stimatore di posa o occlusioni | basso: un passaggio fisico in validazione |
| **Coda ad alta energia** | `E_sem` ed `E_fis` per campione di addestramento, per lingua, salvate ai checkpoint | solo audit delle coppie disallineate [Lett. 82]; **mai filtri automatici**, che colpirebbero le lingue con pochi dati **[Nostra argomentazione]** | nullo: logging |
| **Scorciatoia sul testo** | R@1 T2V e V2T con il video sostituito da rumore | la media cala meno del 50 % → il modello non guarda il video | basso: ogni 500 passi |
| **Retrieval** | R@k, MRR e MedR held-out channel, T2V e V2T; decide la media di R@1 | sotto la baseline ridge (media delle due direzioni) dopo il warmup | la validazione che serve comunque |
| **Velocità del target di posa** | CKA lineare fra `s` al passo 0 e ora, su un batch sonda fisso [Lett. 91] | **diagnostica, non ferma il training**: rendere isotropo il target la abbassa per costruzione, perché la CKA lineare non è invariante a una mappa lineare non ortogonale [Lett. 91]; il contenuto si sorveglia con la riga successiva | nullo: 256 clip ogni 500 passi |
| **Target di posa: isotropia e contenuto** | IsoScore, rango e SIGReg di `s`; R² di posizione e velocità dei keypoint di ogni articolatore letti linearmente da `s` sul batch sonda | R² sceso oltre 0,02 sotto **il suo massimo** → il target perde cinematica **[Nostra proposta, soglie da calibrare in PC7]** | basso: stesso batch sonda |
| **Termini della posa in conflitto** | coseni fra i gradienti di `L_inv`, `L_anchor` e `SIGReg_posa` sull'encoder di posa, e loro quote | coseno ancora–SIGReg stabilmente sotto −0,3 → SIGReg combatte l'ancora **[Nostra proposta]** | nullo |
| **Lettura per passo** | R² su tutti i passi, sui passi soprattutto visibili e soprattutto mascherati, per copertura della maschera ρ | R² dei passi visibili non alto già all'inizio → bug di lettura o di riquadri; R² che non scende al crescere di ρ → maschera ignorata o leak | nullo |
| **Errore in keypoint** | keypoint decodificati con `D` dai passi soprattutto mascherati, confrontati con interpolazione dai passi visibili e con velocità costante | non batte le baseline → nessuna dinamica utile; è anche il risultato fisico **interpretabile** | nullo |
| Articolatori esclusi | frazione di riquadri esclusi per bassa confidenza, per articolatore | mani escluse in massa → il livello fisico ignora proprio le mani | nullo |
| **Query del predictor semantico** | coseno medio fra le uscite delle 8 query | ≈ 1 → query collassate, pooling inutile | nullo |
| **Attenzione del predictor semantico** | entropia dell'attenzione delle query e quota di attenzione sui riquadri degli articolatori rispetto allo sfondo | attenzione uniforme o concentrata sullo sfondo → scorciatoia di sfondo o di canale | basso: batch piccolo ogni 500 passi |
| LoRA per blocco | norma del gradiente della LoRA video per blocco | blocchi a gradiente nullo o esplosivo | nullo |
| **Quota di gradiente per termine** | frazione della norma del gradiente sulla LoRA video dovuta a ciascun termine; con l'addestramento per livello i termini semantici devono valere 0 | un termine semantico diverso da 0 → lo stop-gradient non agisce (bug); nell'ablation «globale», un termine sopra il 90 % per 500 passi → pesi sbilanciati | nullo |
| **Numerica** | NaN e Inf; picchi di loss oltre 6σ dalla media mobile; frazione di passi tagliati dal clipping; overflow in bf16; throughput e attesa del dataloader | qualunque NaN → stop; picchi ripetuti → instabilità; throughput sotto la stima di PC7 → budget a rischio | nullo |

#### 4.13.4 Diagnostiche dei risultati

| Diagnostica | Cosa cattura | Costo |
|---|---|---|
| **R@1 tollerante ai duplicati**, accanto a R@1 standard: una didascalia identica o quasi identica a quella corretta conta come successo | falsi negativi della valutazione: su MS-COCO mancano ×3,6 associazioni immagine→didascalia e ×8,5 didascalia→immagine, e le classifiche fra modelli cambiano [Lett. 90] | nullo |
| **Scarto fra canali visti e held-out channel**: una validazione con video nuovi di canali visti, più un probe lineare di canale e di segnante su `ŷ` | scorciatoia di canale o di identità | basso |
| **R@1 su un sottoinsieme di addestramento** della stessa dimensione della validazione | scarto di overfitting | basso |
| **R@1 per fasce** di durata e di lunghezza della didascalia, T2V e V2T (la query di entrambe le direzioni è una clip, quindi le fasce sono le stesse) | dove si concentrano gli errori | nullo |
| **R@1 ripartito per lingua**, T2V e V2T, solo come diagnostica | il valore misto è una media pesata per numero di video, dominata dall'ASL: la ripartizione mostra le lingue con pochi dati **[Nostra argomentazione]** | nullo |
| **Intervalli bootstrap** sulle query per **ogni** R@k riportato, nelle due direzioni; in validazione per R@1 T2V e V2T | differenze non significative lette come effetti | nullo |
| **Errore in keypoint** del livello fisico contro le baseline (§4.13.3) | risultato fisico interpretabile | nullo |

#### 4.13.5 La run di gate come run diagnostica: fermate programmate

La run di gate (§4.12.2) si ferma in quattro punti fissati in anticipo; a ogni fermata si salva un checkpoint e si leggono i criteri **[Nostra scelta, soglie da calibrare in PC7]**.

| Fermata | Criteri (devono passare tutti) | Se uno fallisce |
|---|---|---|
| **F1 · fine dello stadio P** (epoca 1) | **posa:** `L_anchor` e `SIGReg_posa` in calo · effective rank di `s` > 0,5× il passo 0 · IsoScore di `s` ≥ 0,8 · R² di posizione delle mani da `s` ≥ 0,9. **Semantico:** `γ_sem` > 0,3 · `SIGReg_sem` in calo · R@1 held-out > 5× il caso · test col rumore superato · query non collassate | stop → triage della posa o del semantico |
| **F2 · fine dello stadio F₀** (epoca 2) | `E_fis` in calo · R² dei passi soprattutto visibili > 0,9 · la dinamica batte la baseline · nessun leak | stop → triage del livello fisico |
| **F3 · fine della prima epoca di F** (epoca 3) | `E_fis` in calo · nessuna LoRA sposta il suo strato oltre il 10 % · deriva dell'encoder ≥ 0,5 · media di R@1 T2V e V2T > media della baseline ridge · `ω` < 0,95 · hubness stabile in entrambe le gallerie · curva estrapolata della media di R@1 compatibile con la soglia `X` (calcolata sulla media di C²RL, §4.12.2) · nessun conflitto stabile fra gradienti | stop o correzione |
| **F4 · fine** | tabella del gate su OpenASL | come da gate |

Una correzione che tocca solo componenti **successivi** alla fermata riparte dall'ultimo checkpoint valido; una che tocca componenti **precedenti** riparte da zero **[Nostra proposta]**.

#### 4.13.6 Tabella di triage

| Sintomo | Cosa guardare, in quest'ordine | Causa probabile | Correzione |
|---|---|---|---|
| `E_fis` non scende | sincronia video–posa → R² dei riquadri visibili → R² per copertura ρ | posa sfasata · riquadri o lettura sbagliati · maschera o 3D-RoPE | correggere la pipeline |
| `E_fis` crolla subito (R² > 0,98) | test di leak → P6 → P14 | leak | correggere il mascheramento |
| Collasso di `s` | R² dei keypoint da `s` per articolatore → ancora → SIGReg ed effective rank di `s` → coseno ancora–SIGReg | ancora troppo debole · viste troppo forti · SIGReg che combatte l'ancora | learning rate della posa più basso; peso dei termini della posa; ripiego a due stadi [Lett. 84] |
| R@1 al caso | test sintetici delle metriche → stessi tensori → `γ_sem` → `SIGReg_sem` → query → test col rumore | bug di valutazione · predizione media · collasso · query collassate · scorciatoia sul testo | secondo la causa |
| R@1 alto su canali visti, basso su held-out | probe di canale → attenzione sullo sfondo | scorciatoia di canale | più crop e jitter; meno parametri liberi |
| Validazione buona, OpenASL basso | contaminazione (P2) → R@1 tollerante ai duplicati → fasce di durata | incoerenza fra validazione e benchmark | secondo la causa |
| NaN o picchi di loss | overflow in bf16 → quota di gradiente per termine → valori di SIGReg | instabilità numerica | SIGReg in fp32; learning rate più basso |
| Encoder video appiattito sulla posa | deriva dell'encoder → conflitto fra gradienti → quota di gradiente | peso del livello fisico troppo alto | ridurre il peso di `E_fis`, alla fermata successiva |

**Cadenza:** energie, norme e quote di gradiente, numerica a ogni passo · collasso, predictor, lettura per passo, dinamica, errore in keypoint, localizzazione, LoRA, query e conflitti ogni 100 passi · retrieval, hubness, tabella 2×2, modality gap, attenzione, velocità del target e test col rumore ogni 500 · ordine temporale, deriva dell'encoder, probe fonologici e test di plausibilità ogni 2.000 · **tutto al passo 0** · diagnostiche dei risultati a ogni validazione completa · profilo temporale su BOBSL solo nella valutazione finale. **Cadenze e finestre degli allarmi espresse in passi sono in unità di 1.024 clip viste; con il batch unico di 128 (§4.14) si moltiplicano per 8.**

**Costo complessivo della diagnostica [Nostra stima]:**

```
validazione su 2.000 clip ogni ~500.000 clip viste:  2.000 × (1,04 + 0,18)      ≈  2.400 forward equivalenti
test col rumore sulle stesse clip:              2.000 × 1,04               ≈  2.100
~500.000 clip viste di addestramento:           500.000 × 3,7              ≈  1,85 milioni

→ circa 0,24 %; con le voci ogni ~2 milioni di clip viste e le feature salvate una tantum, sotto l'1 % del calcolo
```

Il collaudo (§4.13.1) costa poche ore-GPU, una volta sola **[Nostra stima]**.

### 4.14 Piano sperimentale

**Revisione del 29/9/2026 [Nostra scelta].** Il piano è ridotto per la sottomissione a **CVPR 2027** (registrazione del paper il 10 novembre, invio il 16 novembre 2026): **una sola scala, ViT-L**. Rispetto alla versione precedente sono tolti la riga ViT-B di ESP-1 e le congetture di scala che la richiedevano (S2, S3), InfoNCE a batch 1.024 con GradCache, la LoRA nell'encoder testuale, il Tier 2 e il Tier 3 (§4.16). La variabile latente resta, come ablation facoltativa.

**Principi [Nostra scelta]:**
- tutte le run su **ViT-L**, con le stesse epoche (stessi dati visti) e lo **stesso batch effettivo, 128**, in tutti i bracci;
- **configurazione di riferimento θ\* = il braccio migliore di ESP-1** secondo la metrica per decidere; ESP-2, ESP-6 ed ESP-4 si applicano a θ\* (`worldsign-ablation.md`);
- una variabile alla volta rispetto a θ\*; solo ESP-1 confronta più loss;
- **significatività**: intervalli bootstrap sulle query per ogni R@k. Con un solo seed per run `σ_seed` non si stima: dove il documento chiede `|Δ| ≥ 2·σ_seed`, vale la non sovrapposizione degli intervalli bootstrap. **[Aperto]** un secondo seed di θ\*, se il budget misurato in PC7 lo consente;
- **metrica per decidere**: R@1 held-out channel; **metrica di benchmark**: R@1 sul test di OpenASL **senza fine-tuning**, sensata perché l'addestramento di OpenASL è già quasi tutto nel pretraining [Lett. 6, 88];
- il **fine-tuning** su OpenASL, PHOENIX-2014T e CSL-Daily, con traduzione, riconoscimento e produzione, si fa **solo sul modello finale**.

#### ESP-1 · Loss (5 run)

```
                                senza SIGReg_sem        con SIGReg_sem
solo allineamento                    A₀                       A
allineamento + L_unif                B₀                       B
InfoNCE, batch 128                                            C
```

| Run | Braccio | `L_pred_sem` | `SIGReg_sem` |
|---|---|---|---|
| 1 | **A** (gate, §4.12.2) | `E_sem` | `½ [SIGReg({ŷ}) + SIGReg({ẽ})]` |
| 2 | **A₀** | `E_sem` | — |
| 3 | **B₀** | `E_sem + L_unif` [Lett. 40] | — |
| 4 | **B** | `E_sem + L_unif` | `½ [SIGReg({ŷ}) + SIGReg({ẽ})]` |
| 5 | **C** | `L_InfoNCE(ŷ, ẽ)`, batch 128, clip dello stesso video escluse dal denominatore (P15) | `½ [SIGReg({ŷ}) + SIGReg({ẽ})]` |

Le formule sono in §4.5.7. Il livello della posa e quello fisico sono identici in tutti i bracci.

**Letture:**
- A₀ → A: quanto porta SIGReg;
- A₀ → B₀: quanto porta il termine di uniformity a coppie;
- **A contro B₀: il vincolo distribuzionale contro l'uniformity a coppie**, la domanda centrale della tesi;
- B contro A e B₀: i due meccanismi si sommano?
- A contro C: cosa aggiungono i negativi, a parità di batch e di dati visti.

**Perché InfoNCE solo a batch 128 [Nostra scelta].** C ha lo stesso SIGReg di A, lo stesso batch e gli stessi dati visti: isola il contributo dei negativi dal loro numero. Il prezzo è che la tesi non si pronuncia più su InfoNCE a batch grande (§2.8), il cui limite informativo cresce con `log N` [Lett. 38, 39]; 128 è il batch a cui LeJEPA resta competitivo [Lett. 35].

#### Ablation sulla configurazione migliore θ\*

| Run | Esperimento | Differenza rispetto a θ\* | Domanda |
|---|---|---|---|
| 6 | **ESP-2** | nessun livello fisico: niente encoder di posa, predictor fisico, ancora e SIGReg sulla posa; solo passaggio semantico | il livello fisico migliora il semantico? (H3) |
| 7 | **ESP-6** | **senza encoder di posa**: il bersaglio del livello fisico è il target encoder di V-JEPA 2.1 (bi-encoder video–video) | la posa come bersaglio serve, rispetto al bersaglio video di V-JEPA 2.1? (`worldsign-ablation.md` §2.2) |
| 8 | **ESP-4 (facoltativa)** | `K = 4` ipotesi dalle 8 query (§4.5.9), a zero parametri aggiuntivi | l'ambiguità delle didascalie richiede una variabile latente? (H7) |

#### Ordine e costo **[Nostra stima]**

```
collaudo (§4.13.1)  →  PC1–PC7  →  run 1 (A, gate) fino alla fermata F2 (fine dello stadio F₀)
         →  run 2–5 in parallelo  →  fine di ESP-1: scelta di θ* (metrica per decidere)
         →  run 6–7 (e 8, facoltativa) in parallelo su θ*  →  modello finale: fine-tuning sui benchmark
```

Unità: **L = un addestramento completo su ViT-L, braccio A**; il valore in GPU-ore si misura in PC7 e dipende dalla dimensione del corpus.

| Run | Costo |
|---|---|
| A, A₀, B₀, B, C | ≈ 1 L ciascuna: stesso batch, nessun GradCache |
| ESP-2 | ≈ 0,85 L: manca il passaggio fisico, ≈ 15 % del passo (§4.9) |
| ESP-6 | ≈ 1 L, più il forward del target encoder [da misurare in PC7] |
| ESP-4 (facoltativa) | ≈ 1 L |
| **Totale** | **≈ 6,85 L**, ≈ 7,85 L con ESP-4, più il fine-tuning del modello finale |

Il piano ha due ondate: i cinque bracci di ESP-1, poi le ablation su θ\*. Con due GPU per run servono 10 GPU per la prima ondata e 6 per la seconda; il tempo reale è circa 2 L se le GPU ci sono.

### 4.15 Leggi di scala: forma attesa e protocollo

**Non nel piano attuale** (revisione del 29/9, §4.14): ESP-5 è tolto; il protocollo resta per un lavoro successivo.

```
R@1(D)  =  R∞  −  A · D^(−α)

   D    numero di VIDEO distinti (non di clip)
   R∞   prestazione asintotica
   A    ampiezza
   α    esponente di scala
```

Si stima con ESP-5 (25/50/100 %). Lettura: curva ancora ripida → limitati dai dati; curva piatta con scarto train/val piccolo → limitati dalla capacità; curva piatta con scarto grande → overfitting **[Nostra scelta]**. Le congetture di scala sono in §5.2.

### 4.16 Alternative considerate e scartate

| Alternativa | Motivo dello scarto |
|---|---|
| Ricostruzione di pixel | la predizione di feature è superiore e dà fisica intuitiva [Lett. 30, 33] |
| InfoNCE come metodo | contrario alla tesi; resta come baseline (ESP-1 C) **[Nostra scelta]** |
| LeVJEPA così com'è | senza predictor non c'è world model; un frammento di frase segnata non significa quanto la frase intera **[Nostra argomentazione]** |
| Moduli di fusione su misura per livello, su profondità scelte da noi | sostituiti dalla fusione multi-livello di V-JEPA 2.1, che usa i blocchi 6/12/18/24 con una ricetta validata (§4.4.5) |
| LoRA solo sui blocchi alti, o blocchi alti congelati | LoRA su tutti i blocchi **[Nostra scelta]** |
| Predire un articolatore dagli altri (mano non dominante dalla dominante) | le condizioni di simmetria e dominanza non sono universali [Lett. 77] |
| Maschere su misura (interpolazione, estrapolazione, articolatore) | il multi-blocco batte le varianti causali [Lett. 30]; si replica la ricetta validata |
| Predizione del futuro o dell'enunciato successivo | contraria all'evidenza dell'ablazione causale [Lett. 30]; fuori scope (§1.5) |
| EMA sul target di posa | contesto e target sono reti diverse; VL-JEPA addestra il target senza EMA [Lett. 34] |
| Secondo teacher V-JEPA | escluso per parsimonia **[Nostra scelta]** |
| Maschera al livello semantico (20–30 %) | VL-JEPA non maschera [Lett. 34]; un segno nascosto rende il bersaglio non determinato **[Nostra argomentazione]**; lo scarto casuale di token resta solo come ripiego di costo (§4.9) |
| Variabile latente in tutte le run | solo nell'ablation facoltativa ESP-4 (§4.5.9) **[Nostra scelta]** |
| `z` continua ottimizzata, calcolata dalla didascalia, o embedding probabilistici | §4.5.9 |
| Predictor semantico da Llama-3.2-1B | 490 M, fuori budget [Lett. 34] |
| Target di posa 3D | più rumoroso del 2D [Lett. 24, 26] |
| Uni-Sign come encoder di posa | faccia sotto la media (R² −1,75), mani 0,75 contro 0,94 di S-JEPA (PC5, §4.4.3) |
| MAMP come encoder di posa | mani 0,75 / 0,41 in posizione / velocità, volto 0,19 (PC5, §4.4.3) |
| Feature cinematiche come target | nessuna astrazione: il livello fisico ricopierebbe le coordinate; restano il tetto di riferimento dei probe (PC5) |
| Augmentation sulla sola posa (rotazione 3D, proporzioni degli arti) | bersaglio incoerente con il video (§3.8) |
| Partizione dell'embedding | §4.4.8 |
| Whitening fisso del testo | [Lett. 46, 50, 51, 52] |
| PE-Core-G, SONAR, BGE-M3 | §4.4.4 |
| Spreadthesign / SP-10 | dizionario di segni isolati, non ridistribuibile [Lett. 16, 21] |
| Allineamento lessicale con clustering | §4.5.5 |
| Augmentation delle didascalie in più lingue | §3.8 |
| Ribilanciamento delle lingue nel campionamento | non adottato: si valuta e si fa fine-tuning su lingue con dati sufficienti, e ribilanciare ripeterebbe molte volte le lingue piccole [Lett. 64]; si riconsidera solo se lo sbilanciamento pesa sulle prestazioni (§3.5) |
| Zero-shot su lingue non viste | mai riuscito nel segnico [Lett. 6, 16] |
| Riga ViT-B di ESP-1 e confronto fra scale (S2, S3) | tolti il 29/9 per i tempi della sottomissione a CVPR 2027: tutte le run su ViT-L (§4.14) |
| InfoNCE a batch 1.024 con GradCache | tolto il 29/9: il confronto con InfoNCE è a parità di batch, 128 (§4.14) |
| LoRA nell'encoder testuale (ex ESP-3) | tolta il 29/9: costa e fa perdere la precomputazione del testo [Lett. 68, 69] |
| Tier 2 e Tier 3 (scala dei dati, dimensione MRL, capacità per lingua, senza ancora, BOBSL non allineato, profondità relativa) | tolti il 29/9 per i tempi; il testo non si tronca (§4.4.4) |

---

## 5. Aspettative, ipotesi e pericoli

### 5.1 Le ipotesi della tesi

Per ogni ipotesi riportiamo l'enunciato, la base in letteratura, il test previsto e che cosa la falsificherebbe.

**H1 — L'uniformity può venire da un vincolo sulla distribuzione, non dai negativi** **[Nostra ipotesi]**
- *Enunciato.* Con allineamento + SIGReg il retrieval è confrontabile con InfoNCE a parità di dati visti e di batch, e nettamente migliore del solo allineamento.
- *Base.* InfoNCE equivale ad allineamento più uniformity [Lett. 40]. Il costo di togliere l'uniformity è stato misurato: −18,6 R@1 [Lett. 34]. SIGReg rende la distribuzione sparsa senza usare negativi [Lett. 35]. Il problema della densità delle rappresentazioni nel segnico è documentato [Lett. 17].
- *Test.* ESP-1: il braccio A confrontato con A₀, B₀, B e C, su ViT-L; in più, la densità delle rappresentazioni misurata con la metrica di SignCL.
- *Falsificazione.* A ≈ A₀, oppure A < C oltre l'incertezza (§4.14).

**H2 — Un target di posa addestrato da zero resta informativo e non collassa** **[Nostra ipotesi — la più rischiosa]**
- *Enunciato.* Con lo stop-gradient fra i livelli e i termini della posa (invarianza fra viste, SIGReg, ancora), un encoder di posa addestrato da zero insieme al resto dà un bersaglio **isotropo e informativo** per tutto l'addestramento: IsoScore di `s` ≥ 0,8 e R² di posizione delle mani ≥ 0,9 alla fine dello stadio P, senza cadute oltre 0,02 dal massimo in seguito (`worldsign-posa.md` §6).
- *Base.* LeJEPA rinuncia a EMA e stop-gradient con garanzia anti-collasso, grazie a SIGReg e all'invarianza fra viste [Lett. 35]; lo stop-gradient sul bersaglio è la difesa di V-JEPA e S-JEPA contro un target che si muove [Lett. 30, 99]. **Però** la validazione di LeJEPA è unimodale e con viste forti; le nostre viste sono deboli (rotazione e rumore), e l'astrazione del bersaglio dipende anche dall'ancora.
- *Test.* Rango, IsoScore e SIGReg di `s`, R² dei keypoint per articolatore sul batch sonda, coseno ancora–SIGReg, fin dalla fermata F1 (§4.13).
- *Falsificazione.* Il rango di `s` crolla, l'IsoScore non arriva a 0,8, oppure il R² dei keypoint scende oltre 0,02 sotto il suo massimo. *Ripiego:* S-JEPA adattato come livello 0 (`worldsign-ablation.md` §3.3, P2), o un bersaglio EMA (P1).

**H3 — Il livello fisico migliora la rappresentazione semantica** **[Nostra ipotesi]**
- *Enunciato.* L'encoder video adattato dal livello fisico dà un retrieval e una discriminazione fine migliori dell'encoder non adattato, letto dallo stesso livello semantico (ESP-2: lo schema di VL-JEPA).
- *Base.* H-JEPA: predizione a tutti i livelli [Lett. 27]; struttura gerarchica del segnato [Lett. 1]; la loss densa di V-JEPA 2.1 migliora le feature locali [Lett. 32].
- *Test.* ESP-2: retrieval, probe fonologici, coppie minime.
- *Falsificazione.* Nessuna differenza oltre 2·σ_seed fra θ* e la variante senza livello fisico.

**H4 — La fisica si eredita dall'encoder e si specializza sul corpo** **[Nostra ipotesi]**
- *Enunciato.* L'adattamento con LoRA conserva la fisica intuitiva dell'encoder [Lett. 33], e il livello fisico aggiunge aspettative sulla cinematica di chi segna.
- *Test.* Probe sull'uscita dell'encoder prima e dopo l'addestramento; test di violation-of-expectation sul segnato (§4.12.4); IntPhys solo se l'encoder scelto è V-JEPA 2-L.
- *Falsificazione.* I probe peggiorano dopo l'adattamento, oppure le manipolazioni temporali non vengono rilevate.

**H5 — Clessidra multilingua** **[Nostra ipotesi]**
- *Enunciato.* L'informazione sulla lingua dei segni è bassa nelle rappresentazioni della posa, più alta nell'uscita dell'encoder video, di nuovo bassa in `ŷ`.
- *Base.* Corpo e iconicità sono in larga parte condivisi fra lingue [Lett. 3, 4]; i lessici differiscono e le lingue interferiscono fra loro [Lett. 5, 65]; l'identità della lingua occupa un sottospazio a basso rango, individuabile con probe e rimovibile con una centratura [Lett. 61, 62, 63].
- *Test.* Probe di identificazione della lingua sulle tre rappresentazioni.
- *Falsificazione.* La lingua è già ben identificabile nella posa, oppure `ŷ` è identificabile quanto l'uscita dell'encoder.

**H6 — L'energia come misura di plausibilità per la produzione** **[Nostra ipotesi, speculativa]**
- *Enunciato.* `Ē_fis` ed `E_sem` forniscono un punteggio di plausibilità utile a valutare segnato generato.
- *Stato.* **Fuori dal piano sperimentale attuale**: servirebbero giudizi umani.

**H7 — Una variabile latente discreta aiuta il retrieval** **[Nostra ipotesi, debole]**
- *Enunciato.* Con K = 4 ipotesi il retrieval migliora, perché le traduzioni valide ma lontane fra loro non vengono più mediate.
- *Base a favore.* Più ipotesi evitano la predizione della media [Lett. 81]; K > 1 batte K = 1 nel retrieval cross-modale [Lett. 80].
- *Base contro.* L'Y-Encoder di VL-JEPA assorbe già la variabilità superficiale del testo [Lett. 34]; una `z` abbassa anche l'energia delle coppie sbagliate [Lett. 28].
- *Test.* ESP-4, facoltativa (§4.14).
- *Falsificazione.* Δ R@1 entro 2·σ_seed o negativo, oppure un'ipotesi vince oltre l'80 % delle volte.

### 5.2 Congetture sulla scala

**Non testate nel piano attuale** (§4.14): S1 richiede ESP-5, S2 e S3 due scale di encoder.

**S1 [Nostra ipotesi].** Con un encoder video congelato e adattato solo con LoRA, l'esponente α di §4.15 è piccolo e la curva satura presto: la prestazione asintotica dipende soprattutto dall'encoder. *Test:* ESP-5.

**S2 [Nostra ipotesi].** Con pochi dati un encoder congelato più grande non aumenta l'overfitting: ViT-L ≥ ViT-B in tutti i bracci di ESP-1, senza uno scarto train/val maggiore. *Test:* confronto fra le due righe di ESP-1.

**S3, riformulata [Nostra ipotesi].** A parità di dati visti, il vantaggio di InfoNCE sul braccio SIGReg (C − A) si riduce con un encoder migliore: feature più ricche richiedono meno negativi. *Test:* ESP-1.
*Nota.* La versione originale **non viene testata**. Diceva che, a memoria fissa, un modello più grande costringe a un batch più piccolo, penalizzando InfoNCE, il cui limite informativo cresce con `log N` [Lett. 38, 39]. Con GradCache, però, il batch effettivo è uguale nelle due righe (§4.14).

### 5.3 Previsioni fissate in anticipo

| Esperimento | Previsione | Motivo |
|---|---|---|
| PC1 | valore di SIGReg già basso sui target di EmbeddingGemma | i modelli contrastivi moderni sono isotropi [Lett. 51] |
| ESP-1 | A ≫ A₀; B₀ > A₀; A ≈ B₀; B ≈ A; A ≈ C entro l'incertezza | H1 |
| ESP-2 | θ* > senza livello fisico, con lo scarto più grande su coppie minime e probe fonologici che sul retrieval | H3 |
| ESP-6 | **[Aperto]**: previsione da fissare prima del lancio | H3 |
| ESP-4 (facoltativa) | guadagno piccolo o nullo; se c'è, concentrato sulle didascalie più libere (BOBSL, YouTube-SL-25) | H7, [Lett. 34, 80] |
| H2 | nessun collasso di `s` grazie allo stop-gradient, a SIGReg, all'invarianza e all'ancora | [Lett. 35] |
| Plausibilità | manipolazioni temporali rilevate da `Ē_fis`; il controllo con la posa sfasata rilevato | [Lett. 33] |

### 5.4 Come leggere gli esiti: nessun esito è inutile

| Esito | Interpretazione |
|---|---|
| A ≈ C e A ≫ A₀ | **tesi sostenuta**: retrieval cross-modale senza negativi, a parità di dati visti e di batch |
| A ≈ A₀ | SIGReg non fornisce un'uniformity utile nel cross-modale; delimita la portata di [Lett. 35] |
| A ≪ C | i negativi aggiungono qualcosa oltre all'uniformity; il divario è quantificato e l'architettura resta usabile con InfoNCE |
| B > A | serve una dispersione a coppie in più rispetto al vincolo sulle proiezioni 1-D **[Nostra interpretazione]** |
| A ≈ B₀ | il vincolo distribuzionale fornisce l'uniformity quanto il termine a coppie, senza calcolare distanze fra campioni |
| collasso del target di posa addestrato da zero | la garanzia di LeJEPA non basta con viste deboli; si passa a S-JEPA adattato o a un bersaglio EMA (`worldsign-ablation.md` §3.3) |
| il livello fisico non migliora il semantico | risultato di semplificazione: basta lo schema VL-JEPA con SIGReg |
| bersaglio video ≈ o > bersaglio di posa (ESP-6) | la posa come bersaglio non serve: il modello si semplifica |
| la variabile latente migliora (ESP-4, se fatta) | l'ambiguità delle didascalie è rilevante: la `z` entra nel modello finale |
| risultato sotto la baseline ridge | le feature congelate contengono già la struttura utile; è comunque un'informazione su V-JEPA applicato al segnato |

### 5.5 Pericoli e mitigazioni

| Pericolo | Perché può accadere | Come ce ne accorgiamo | Mitigazione |
|---|---|---|---|
| **Collasso del target di posa** | l'encoder di posa si addestra da zero, senza EMA (H2) | effective rank, IsoScore e SIGReg di `s` (fermata F1) | più peso all'ancora; S-JEPA adattato o bersaglio EMA (`worldsign-ablation.md` §3.3) |
| **Il target di posa si semplifica** | invarianza e SIGReg non impediscono la perdita di informazione | R² dei keypoint per articolatore letti da `s` contro il loro massimo | più peso all'ancora |
| **Outlier nella normalizzazione della posa** | lo stimatore sovrappone le spalle: l'1 % delle clip arrivava a centinaia di larghezze di spalle | raggio massimo nel collaudo della posa (§4.13.1) | guardie di §3.6, già nel codice |
| **Encoder video appiattito sulla posa** | il gradiente fisico domina l'adattamento | deriva dell'encoder; coseno fra i gradienti dei due passaggi | ridurre il peso di `E_fis` |
| **Il predictor predice la media** | relazioni uno-a-molti [Lett. 37, 81] | `γ`, `R²`, hubness | bracci B₀, B e C; ESP-4 (facoltativa) |
| **Leak nella maschera** | errore silenzioso: la loss scende comunque | test controfattuale, P6, P14 | correggere il mascheramento prima di proseguire |
| **Localizzazione regalata dal riquadro** | il riquadro di lettura viene dai keypoint del bersaglio | energia con riquadri spostati | **[Aperto]** termine sulle predizioni fuori riquadro (§4.6) |
| **Energia alta per ambiguità, non per implausibilità** | la L1 porta alla mediana, non a un modo; occlusioni | test di plausibilità per confronto (§4.12.4) | media su più maschere |
| **Ordine ignorato nel predictor semantico** | attenzione e pooling invarianti alle permutazioni | `ω` | 3D-RoPE nel predictor |
| **Discriminazione fine insufficiente** (coppie minime) | nessun negativo esplicito; problema di densità [Lett. 17] | margine sulle coppie minime | più risoluzione sulle mani; dimensioni maggiori; se persiste, rivedere la tesi |
| **Scorciatoia sul testo** | il modello impara la distribuzione delle didascalie | video sostituito da rumore; baseline sulle sole didascalie | più peso al livello fisico |
| **Modality gap** | non si chiude da solo [Lett. 41] | classificatore che distingue `ŷ` da `ẽ` | SIGReg su ciascuna modalità (già previsto) |
| **Target testuale anisotropo** | allineamento e SIGReg avrebbero ottimi diversi | PC1; coseno fra i gradienti | testa MLP |
| **Overfitting sui canali** | ~100 clip per video | split held-out channel | meno parametri liberi; early stopping |
| **Underfitting al cambio di dominio** | encoder addestrato su video naturali | probe sull'encoder; curve train/val | non ridurre la LoRA; PC3, PC4 |
| **Sbilanciamento fra lingue** | ASL e CSL coprono gran parte dei dati; nessun ribilanciamento (§3.5) | R@1 ripartito per lingua; scarto train/val per lingua | soluzione alternativa solo se lo sbilanciamento pesa sulle prestazioni **[Aperto]** |
| **Rumore di allineamento in BOBSL** | ritardo medio di ~2,7 s [Lett. 9] | tabella 2×2 | al livello semantico solo clip allineate |
| **Stimatore di posa inaffidabile sulle mani** | mani piccole, veloci, in contatto [Lett. 26] | distribuzione delle confidenze; frazione di articolatori esclusi | pesatura per confidenza |
| **Formato di posa diverso fra pre-addestramento e addestramento** | errore silenzioso: stessi 69 keypoint, normalizzazione diversa | P9 | un solo modulo di tokenizzazione della posa, condiviso |
| **Pesi del predictor assenti nel checkpoint** | fonti discordanti [Lett. 32, 72] | PC6: presenti in 2.1-B (22,9 M) e 2.1-L (23,0 M) | predictor da zero a 2 blocchi, o V-JEPA 2-L |
| **Encoder distillato poco adatto** | la distillazione non usa supervisione profonda [Lett. 32] | PC3 | passare a V-JEPA 2-L |
| **Costo del passaggio semantico non mascherato** | ≈ 85 % del passo | PC7 | scarto casuale del 50 % dei token [Lett. 78, 79] |
| **Costo della risoluzione** | 18.432 token a 384² | PC4; dry run | crop a 256² |
| **Budget di calcolo** | sette run ViT-L (otto con ESP-4) a circa sei settimane dalla scadenza di CVPR 2027 | ordine fail-fast; costo L misurato in PC7 | bracci di ESP-1 solo dopo F2 della run di gate; ESP-4 facoltativa è la prima a cadere |
| **Encoder testuale sub-ottimo** | la scelta dell'encoder testuale sposta fino a +7,9 R@1 [Lett. 34] | — | limite dichiarato |

### 5.6 Limiti dichiarati

- **Dati:** sbilanciamento geografico e scarsa presenza di tonalità di pelle scure [Lett. 6].
- **Posa 2D:** i vincoli tridimensionali (solidità, profondità dello spazio segnico) non sono pienamente esprimibili [Lett. 26]; le augmentation sulla sola posa non sono utilizzabili (§3.8).
- **Fisica:** il corpus contiene persone che segnano, non oggetti; si può apprendere al più la cinematica del corpo, non la fisica degli oggetti **[Nostra argomentazione]**. V-JEPA non acquisisce solidità né collisioni [Lett. 33], e noi non possiamo testare la solidità senza video generato.
- **Scope:** nessun allineamento lessicale, nessuno spazio segnico, nessuna coerenza fra enunciati, nessuno zero-shot.
- **Valutazione:** IntPhys utilizzabile solo con V-JEPA 2-L; le congetture di scala S1–S3 non sono testate, H7 solo se si fa ESP-4; InfoNCE è confrontato solo a batch 128; il benchmark di retrieval multilingue non è standard; il gate confronta un modello senza fine-tuning con C²RL, che lo fa, e senza parità di dati; OpenASL è un benchmark in dominio (§3.9).
- **Scelte prese dalla letteratura e non verificate da noi:** niente whitening fisso [Lett. 46, 50, 51, 54]; target di posa in 2D [Lett. 24]; maschera e loss di V-JEPA 2.1 senza ablazioni nostre [Lett. 30, 32].
- **Stime:** parametri e costi sono approssimati.

### 5.7 Punti ancora aperti

| Punto | Stato | Come si chiude |
|---|---|---|
| Encoder: V-JEPA 2.1-L distillato o V-JEPA 2-L | **chiuso il 29/9: V-JEPA 2.1-L** (PC3, §4.4.1) | — |
| Risoluzione: 256² o 384² | **chiuso il 29/9: 256² con crop** (PC4, §4.4.1) | — |
| Dimensione di troncamento MRL | **chiuso il 29/9: nessun troncamento**, il vettore si usa a 768 (§4.4.4) | — |
| Encoder di posa | **chiuso il 3/10, rivisto il 6/10**: l'encoder di worldSign adattato, da zero, senza dropout, `C = 192`, 8,03 M; 4 viste, SIGReg per passo, λ = 0,04 (`worldsign-posa.md`) | — |
| Gerarchia e stadi | **chiuso il 3/10**: per livello, stadi P/F₀/F a confini di epoca, 15 epoche, pazienza 3 nello stadio F (`worldsign-gerarchia.md`) | — |
| Soglie della fermata F1 sulla posa | proposta: IsoScore ≥ 0,8, R² di posizione delle mani ≥ 0,9, caduta dal massimo ≤ 0,02 | PC7 |
| Probe della lingua separato dall'identità del canale | split per canale implementato; **non calcolabile** finché l'ASL delle clip di test viene da un solo canale | nuove clip di test dopo il download con l'ordine mescolato |
| Pesi del predictor nel checkpoint 2.1 | **presenti** (2.1-B 22,9 M, 2.1-L 23,0 M); restano dimensioni e fusione | PC6 |
| Tempo di calcolo | cinque bracci di ESP-1 più ESP-2 ed ESP-6 (≈ 6,85 L), ESP-4 facoltativa (≈ 1 L), più il fine-tuning del modello finale | misura di L in PC7 |
| Secondo seed di θ\* (`σ_seed`) | aperto | budget, dopo la misura di L in PC7 |
| Soglia `X` del gate | **chiusa: 46,5** (0,75 × la media di C²RL nelle due direzioni; era 46,7 sulla sola T2V fino al 4/10); dopo la revisione del 29/9 è il criterio della fermata F3 (§4.12.2) | — |
| Learning rate, durata degli stadi, ε di ESP-4 (facoltativa) | valori di partenza | PC7 |
| Costo reale del passo e ripiego sui token | stima 3,7 | PC7 |
| Localizzazione regalata dal riquadro di lettura | allarme previsto | mitigazione da definire se l'allarme scatta |
| Collocazione della capacità specifica per lingua | nessun modulo; ESP-7 tolto (§4.14) | eventuale estensione, dai probe di lingua |
| Canale di profondità relativa | tolto dal piano attuale (§4.14) | — |
| Ribilanciamento fra lingue | non previsto | solo se lo sbilanciamento pesa sulle prestazioni (§3.5) |
| Benchmark di retrieval multilingue | da definire | — |

---

## 6. Conclusione

### 6.1 Sintesi

Il retrieval per la lingua dei segni oggi si basa su obiettivi contrastivi, che richiedono batch grandi e hanno un limite informativo legato alla dimensione del batch [Lett. 38, 39]. VL-JEPA misura quanto costa togliere l'uniformity da un JEPA cross-modale, ma non verifica se l'uniformity possa venire da un vincolo sulla distribuzione [Lett. 34]. SIGReg fornisce questo vincolo con garanzie teoriche, finora validate solo in contesti unimodali [Lett. 35].

Proponiamo un **world model a energia con due livelli di astrazione** per la lingua dei segni continua e multilingua che:
- **percepisce** con un encoder video congelato, adattato con LoRA su tutti i blocchi;
- **predice lo stato del corpo** dal video in gran parte mascherato, con la maschera e la loss di V-JEPA 2.1, verso le rappresentazioni di un encoder di posa addestrato da zero e letto con lo stop-gradient;
- **predice il significato** dal video intero, con lo schema di VL-JEPA, verso l'embedding della didascalia;
- **allinea senza negativi**, imponendo l'uniformity con SIGReg;
- usa l'**energia** — l'errore di predizione — per il retrieval e come misura di plausibilità;
- **resta sotto i 30 M di parametri addestrabili** (≈ 29,8 M, di cui 8,06 M nell'encoder di posa con il decoder), perché il corpus conta ~6.650 ore ma solo ~41.000 video e il rischio di overfitting è alto.

### 6.2 Perché il progetto è informativo in ogni caso

L'esperimento centrale (ESP-1) confronta cinque configurazioni di loss su ViT-L con un protocollo controllato: stesso batch (128), stessi dati visti, stessi moduli addestrabili. Qualunque sia l'esito (§5.4), si ottiene un risultato: la conferma della tesi, oppure una misura di quanto aggiungono i negativi e dei limiti di SIGReg. ESP-2 dice se il livello fisico serve; ESP-6 se serve la posa come bersaglio, rispetto al bersaglio video; ESP-4, se c'è tempo, se l'ambiguità delle didascalie richiede una variabile latente. Le diagnostiche fail-fast (§4.13), a un costo sotto l'1 % del calcolo, servono a far emergere i problemi tecnici alla prima run, non a budget esaurito.

### 6.3 Prossimi passi

1. Collaudo (§4.13.1) e controlli preliminari PC1–PC7.
2. Run di gate: θ\* = braccio A, ViT-L, con le fermate F1–F3 (§4.13.5).
3. Superata F2: gli altri quattro bracci di ESP-1 in parallelo — A₀, B₀, B, C a batch 128 (§4.14).
4. Scelta di θ\* con la metrica per decidere; ESP-2 ed ESP-6 su θ\*, ESP-4 facoltativa.
5. Modello finale: fine-tuning su OpenASL, PHOENIX-2014T e CSL-Daily; traduzione, riconoscimento continuo (su PHOENIX-2014T e CSL-Daily, che hanno le gloss [Lett. 12, 13]) e produzione.
6. Stesura per CVPR 2027 (registrazione 10 novembre, invio 16 novembre 2026), seguendo il protocollo di comparabilità (§3.9).

---

## Riferimenti

**Linguistica delle lingue dei segni**
1. W. C. Stokoe (1960). *Sign Language Structure.* Studies in Linguistics.
2. R. Battison (1978). *Lexical Borrowing in American Sign Language.* Linstok Press.
3. M. Perlman et al. (2018). *Iconicity in Signed and Spoken Vocabulary.* Frontiers in Psychology. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01433/full
4. S. Parkhurst, D. Parkhurst. *Lexical comparisons of signed languages and the effects of iconicity.* SIL Work Papers. https://commons.und.edu/sil-work-papers/vol47/iss1/2/
5. *L2M1 and L2M2 Acquisition of Sign Lexicon* (2022). Frontiers in Psychology. https://pmc.ncbi.nlm.nih.gov/articles/PMC9231460/

**Dataset e benchmark**
6. G. Tanzer, B. Zhang (2025). *YouTube-SL-25.* ICLR. https://arxiv.org/abs/2407.11144
7. S. Albanie et al. (2021). *BBC-Oxford British Sign Language Dataset.* https://arxiv.org/abs/2111.03635
8. Z. Li et al. (2025). *Uni-Sign: Toward Unified Sign Language Understanding at Scale.* ICLR. https://arxiv.org/abs/2501.15187
9. *Deep Understanding of Sign Language for Sign to Subtitle Alignment* (2025). https://arxiv.org/abs/2503.03287
10. L. Momeni et al. (2022). *Automatic Dense Annotation of Large-Vocabulary Sign Language Videos.* ECCV. https://link.springer.com/chapter/10.1007/978-3-031-19833-5_39
11. A. Duarte et al. (2021). *How2Sign.* CVPR. https://arxiv.org/abs/2008.08143
12. N. C. Camgöz et al. (2018). *Neural Sign Language Translation* (PHOENIX-2014T). CVPR.
13. H. Zhou et al. (2021). *Improving Sign Language Translation with Monolingual Data by Sign Back-Translation* (CSL-Daily). CVPR.

**Retrieval, traduzione e rappresentazioni per il segnato**
14. Y. Cheng et al. (2023). *CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning.* CVPR. https://arxiv.org/abs/2303.12793
15. *SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval* (2024). https://arxiv.org/abs/2407.16394
16. Z. Jiang et al. (2024). *SignCLIP.* EMNLP. https://aclanthology.org/2024.emnlp-main.518/
17. J. Ye et al. (2024). *Improving Gloss-free Sign Language Translation by Reducing Representation Density.* NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c225136cfe52a8fd66658bbcf9d894ab-Abstract-Conference.html
18. S. Gueuwou et al. (2025). *SHuBERT.* ACL. https://aclanthology.org/2025.acl-long.1397/
19. B. Zhang, G. Tanzer, O. Firat (2024). *Scaling Sign Language Translation.* NeurIPS. https://arxiv.org/abs/2407.11855
20. *SONAR-SLT: Multilingual Sign Language Translation via Language-Agnostic Sentence Embedding Supervision* (2025). https://arxiv.org/abs/2510.19398
21. A. Yin et al. (2022). *MLSLT: Towards Multilingual Sign Language Translation.* CVPR. https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_MLSLT_Towards_Multilingual_Sign_Language_Translation_CVPR_2022_paper.pdf
22. *Multilingual Gloss-free Sign Language Translation: Towards Building a Sign Language Foundation Model* (2025). ACL. https://arxiv.org/abs/2505.24355
23. *Improving Continuous Sign Language Recognition with Cross-Lingual Signs* (2023). ICCV. https://arxiv.org/abs/2308.10809

**Posa**
24. *Evaluating the Immediate Applicability of Pose Estimation for Sign Language Recognition* (2021). https://arxiv.org/abs/2104.10166
25. *Towards the extraction of robust sign embeddings for low resource sign language recognition* (2023). https://arxiv.org/abs/2306.17558
26. M. Ivashechkin, O. Mendez, R. Bowden (2023). *Improving 3D Pose Estimation for Sign Language.* https://arxiv.org/abs/2308.09525

**JEPA, world model, energy-based model**
27. Y. LeCun (2022). *A Path Towards Autonomous Machine Intelligence.* Position paper, OpenReview.
28. A. Dawid, Y. LeCun (2023). *Introduction to Latent Variable Energy-Based Models.* https://arxiv.org/abs/2306.02572
29. M. Assran et al. (2023). *I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR. https://arxiv.org/abs/2301.08243
30. A. Bardes et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA). https://arxiv.org/abs/2404.08471
31. M. Assran et al. (2025). *V-JEPA 2.* https://arxiv.org/abs/2506.09985
32. *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning* (2026). https://arxiv.org/abs/2603.14482
33. Q. Garrido et al. (2025). *Intuitive physics understanding emerges from self-supervised pretraining on natural videos.* https://arxiv.org/abs/2502.11831
34. D. Chen, M. Shukor, …, Y. LeCun, P. Fung (2025). *VL-JEPA.* https://arxiv.org/abs/2512.10942
35. R. Balestriero, Y. LeCun (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* https://arxiv.org/abs/2511.08544
36. *LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics* (2026). https://arxiv.org/abs/2608.27395
37. *Large Concept Models: Language Modeling in a Sentence Representation Space* (2024). https://arxiv.org/abs/2412.08821

**Apprendimento contrastivo e geometria degli embedding**
38. A. van den Oord, Y. Li, O. Vinyals (2018). *Representation Learning with Contrastive Predictive Coding.* https://arxiv.org/abs/1807.03748
39. B. Poole et al. (2019). *On Variational Bounds of Mutual Information.* ICML. https://arxiv.org/abs/1905.06922
40. T. Wang, P. Isola (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML. https://arxiv.org/abs/2005.10242
41. W. Liang et al. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS. https://arxiv.org/abs/2203.02053
42. L. Gao, Y. Zhang, J. Han, J. Callan (2021). *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup* (GradCache). RepL4NLP. https://aclanthology.org/2021.repl4nlp-1.31/
43. Q. Garrido et al. (2023). *RankMe.* ICML. https://arxiv.org/abs/2210.02885
44. W. Rudman et al. (2022). *IsoScore: Measuring the Uniformity of Embedding Space Utilization.* Findings of ACL.
45. J. von Kügelgen et al. (2021). *Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style.* NeurIPS. https://arxiv.org/abs/2106.04619

**Embedding testuali, anisotropia, whitening**
46. T. Gao, X. Yao, D. Chen (2021). *SimCSE.* EMNLP. https://arxiv.org/abs/2104.08821
47. J. Su et al. (2021). *Whitening Sentence Representations for Better Semantics and Faster Retrieval.* https://arxiv.org/abs/2103.15316
48. J. Huang et al. (2021). *WhiteningBERT.* Findings of EMNLP. https://aclanthology.org/2021.findings-emnlp.23/
49. *WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings* (2023). ACL. https://arxiv.org/abs/2305.17746
50. M. Ait-Saada, M. Nadif (2023). *Is Anisotropy Truly Harmful? A Case Study on Text Clustering.* ACL. https://aclanthology.org/2023.acl-short.103/
51. *Redundancy, Isotropy, and Intrinsic Dimensionality of Prompt-based Text Embeddings* (2025). https://arxiv.org/abs/2506.01435
52. *Whitening Not Recommended for Classification Tasks in LLMs* (2024). https://arxiv.org/abs/2407.12886
53. J. Mu, P. Viswanath (2018). *All-but-the-Top: Simple and Effective Postprocessing for Word Representations.* ICLR. https://arxiv.org/abs/1702.01417
54. H. Jégou, O. Chum (2012). *Negative Evidences and Co-occurrences in Image Retrieval: The Benefit of PCA and Whitening.* ECCV. https://link.springer.com/chapter/10.1007/978-3-642-33709-3_55
55. F. Radenović, G. Tolias, O. Chum (2018). *Fine-tuning CNN Image Retrieval with No Human Annotation.* TPAMI. https://arxiv.org/abs/1711.02512
56. A. Kusupati et al. (2022). *Matryoshka Representation Learning.* NeurIPS. https://arxiv.org/abs/2205.13147
57. Google (2025). *EmbeddingGemma model card.* https://ai.google.dev/gemma/docs/embeddinggemma/model_card
58. P.-A. Duquenne, H. Schwenk, B. Sagot (2023). *SONAR: Sentence-Level Multimodal and Language-Agnostic Representations.* https://arxiv.org/abs/2308.11466
59. J. Chen et al. (2024). *M3-Embedding (BGE-M3).* Findings of ACL. https://aclanthology.org/2024.findings-acl.137/
60. Meta (2025). *Perception Encoder: The best visual embeddings are not at the output of the network.* https://openreview.net/pdf/13c5dcc0a71a6363ecf0568e449798970fe54388.pdf

**Multilingualità**
61. J. Libovický, R. Rosa, A. Fraser (2020). *On the Language Neutrality of Pre-trained Multilingual Representations.* Findings of EMNLP. https://aclanthology.org/2020.findings-emnlp.150/
62. T. A. Chang, Z. Tu, B. K. Bergen (2022). *The Geometry of Multilingual Language Model Representations.* EMNLP. https://arxiv.org/abs/2205.10964
63. *Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations* (2022). EMNLP. https://aclanthology.org/2022.emnlp-main.379/
64. A. Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). ACL. https://arxiv.org/abs/1911.02116
65. Z. Wang, Z. C. Lipton, Y. Tsvetkov (2020). *On Negative Interference in Multilingual Models.* EMNLP. https://aclanthology.org/2020.emnlp-main.359/

**Adattamento efficiente**
67. E. J. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. https://arxiv.org/abs/2106.09685
68. *Measuring Catastrophic Forgetting in Cross-Lingual Transfer Paradigms: Exploring Tuning Strategies* (2023). https://arxiv.org/abs/2309.06089
69. *REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning* (2026). https://arxiv.org/abs/2604.17257
70. H. Liu et al. (2024). *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). CVPR. https://arxiv.org/abs/2310.03744

**Varie**
71. K. Renz et al. (2021). *Sign Language Segmentation with Temporal Convolutional Networks.* ICASSP. https://arxiv.org/abs/2011.12986
72. Meta. *V-JEPA 2 / 2.1 — repository ufficiale* (`src/hub/backbones.py`). https://github.com/facebookresearch/vjepa2
73. *Uni-Sign — repository ufficiale e checkpoint.* https://github.com/ZechengLi19/Uni-Sign
74. *ASL Citizen* (2023). NeurIPS Datasets & Benchmarks.
75. *The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes* (2023). ASSETS.
76. *WLASL: Word-level Deep Sign Language Recognition from Video* (2020). WACV.

**Aggiunti in questa revisione**
77. P. Eccarius, D. Brentari (2007). *Symmetry and dominance: A cross-linguistic study of signs and classifier constructions.* Lingua. https://www.sciencedirect.com/science/article/abs/pii/S0024384106000210
78. K. Li et al. (2023). *Unmasked Teacher: Towards Training-Efficient Video Foundation Models.* ICCV. https://arxiv.org/abs/2303.16058
79. Y. Li, H. Fan et al. (2023). *Scaling Language-Image Pre-training via Masking* (FLIP). CVPR. https://arxiv.org/abs/2212.00794
80. Y. Song, M. Soleymani (2019). *Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval* (PVSE). CVPR. https://arxiv.org/abs/1906.04402
81. C. Rupprecht et al. (2017). *Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses.* ICCV. https://arxiv.org/abs/1612.00197
82. Z. Huang et al. (2021). *Learning with Noisy Correspondence for Cross-modal Matching* (NCR). NeurIPS. https://proceedings.neurips.cc/paper/2021/hash/f5e62af885293cf4d511ceef31e61c80-Abstract.html
83. M. Radovanović, A. Nanopoulos, M. Ivanović (2010). *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* JMLR. https://jmlr.org/papers/v11/radovanovic10a.html
84. X. Li et al. (2025). *Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers* (SALT). https://arxiv.org/abs/2509.24317
85. S. Chun et al. (2021). *Probabilistic Embeddings for Cross-Modal Retrieval* (PCME). CVPR. https://arxiv.org/abs/2101.05068
86. B. Shi, D. Brentari, G. Shakhnarovich, K. Livescu (2022). *Open-Domain Sign Language Translation Learned from Online Video* (OpenASL). EMNLP. https://arxiv.org/abs/2205.12870
87. *C²RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval* (2025). IEEE TCSVT. https://arxiv.org/abs/2408.09949
88. D. Uthus, G. Tanzer, M. Georg (2023). *YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus.* NeurIPS Datasets & Benchmarks. https://arxiv.org/abs/2306.15162
89. J. Li et al. (2021). *Align before Fuse: Vision and Language Representation Learning with Momentum Distillation* (ALBEF). NeurIPS. https://arxiv.org/abs/2107.07651
90. S. Chun, W. Kim, S. Park, M. Chang, S. J. Oh (2022). *ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO.* ECCV. https://arxiv.org/abs/2204.03359
91. S. Kornblith, M. Norouzi, H. Lee, G. Hinton (2019). *Similarity of Neural Network Representations Revisited* (CKA). ICML. https://arxiv.org/abs/1905.00414
92. S. Malladi, K. Lyu, A. Panigrahi, S. Arora (2022). *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS. https://arxiv.org/abs/2205.10287
93. Y. Zhi, Z. Tong, L. Wang, G. Wu (2021). *MGSampler: An Explainable Sampling Strategy for Video Action Recognition.* ICCV. https://arxiv.org/abs/2104.09952
94. R. Xian et al. (2024). *PMI Sampler: Patch Similarity Guided Frame Selection for Aerial Action Recognition.* WACV. https://arxiv.org/abs/2304.06866
95. A. Kumar, A. Raghunathan, R. Jones, T. Ma, P. Liang (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT). ICLR. https://arxiv.org/abs/2202.10054
96. V. Kurin, A. De Palma, I. Kostrikov, S. Whiteson, M. P. Kumar (2022). *In Defense of the Unitary Scalarization for Deep Multi-Task Learning.* NeurIPS. https://arxiv.org/abs/2201.04122
97. A. Chowdhery et al. (2022). *PaLM: Scaling Language Modeling with Pathways.* https://arxiv.org/abs/2204.02311
98. A. Grattafiori et al. (2024). *The Llama 3 Herd of Models.* https://arxiv.org/abs/2407.21783

**Encoder di posa**
99. M. Abdelfattah, A. Alahi (2024). *S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition.* ECCV. https://sjepa.github.io/
100. Y. Mao, J. Deng, W. Zhou, Y. Fang, W. Ouyang, H. Li (2023). *Masked Motion Predictors are Strong 3D Action Representation Learners* (MAMP). ICCV. https://arxiv.org/abs/2308.07092
