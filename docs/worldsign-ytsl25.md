# YouTube-SL-25: stato dei dati scaricati, integrità, testo e split

**Data:** 12/10/2026 (aggiornato dopo il recupero dei sottotitoli, la ri-codifica degli AV1 e la deduplica). **Cosa descrive:** il sottoinsieme di YouTube-SL-25 effettivamente presente su disco (`~/dataset/youtube-sl-25`), non il rilascio completo. Codice: `signworld/data/analysis/youtube_sl25*.py`; risultati grezzi, tabelle e figure in `reports/youtube_sl25/` (`report.md` per le figure, `integrity.json`, `text_and_splits.json`, `splits_proposal.csv`).

## 1. Cosa abbiamo

|                                             | Valore                                                                      |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| ID nel rilascio                             | 39.197 (tutti unici, 56 lingue dei segni)                                   |
| Video scaricati e validi                    | **13.536 (34,5 %)**, 1.088 ore, 542 GB                                      |
| Video con metadati (`info/*.json`)          | 13.455                                                                      |
| Video con almeno una traccia di sottotitoli | 13.454                                                                      |
| Video con la traccia nella propria lingua   | 12.852                                                                      |
| ID non scaricati                            | 25.661: 22.383 mai provati, 2.640 «non disponibili», 561 blocchi bot, 77 altro |

(I 13.536 sono i video attuali: 2 duplicati sono stati messi in quarantena, §2.)

**Il download dei video si è fermato per un rate-limit di YouTube**. Le liste `failed_*.txt` (ferme al 7/10) vengono riscritte a ogni run, e con il rate-limit attivo molti video riprovabili finiscono tra i «non disponibili»: il numero di «non disponibili» **non è una misura della disponibilità reale**, è un limite superiore, da rimisurare con un download pulito. In più, **22.383 ID non sono mai stati tentati**: la copertura reale dipende soprattutto da questo.

**Recupero dei sottotitoli (12/10).** Con i cookie di Chrome rinnovati, `download_subtitles.py` (un worker) ha ripreso gli 803 video senza metadati: 719 recuperati, 79 «non disponibili» (rimossi o privati su YouTube: verificato a mano su un campione, alcuni si scaricano e altri danno `Video unavailable`) e 2 con errore generico. **Restano 81 video senza metadati né testo**, da escludere dal training o lasciare senza testo. Per i 4 VTT con caratteri di sostituzione il ri-download dà file identici: il carattere � è già nel sottotitolo su YouTube, non è una corruzione del download.

## 2. Controlli di integrità

Script: `python -m signworld.data.analysis.youtube_sl25_integrity` (sola lettura; con `--quarantine` sposta i file inutilizzabili senza cancellarli). Esito sul disco al 12/10:

| Controllo                                                                    | Esito                                                                  |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| ID duplicati nel CSV del rilascio                                            | **0**                                                                  |
| Video con più contenitori (stesso ID, `.mp4` e `.webm`)                      | 0                                                                      |
| Video fuori dal rilascio                                                     | 0                                                                      |
| File temporanei (`.part`, `.ytdl`, `.fNNN`)                                  | 0                                                                      |
| Video illeggibili per ffprobe, di dimensione zero, senza video o senza audio | **0**                                                                  |
| Sottotitoli vuoti                                                            | 0                                                                      |
| Sottotitoli senza video corrispondente                                       | 0                                                                      |
| **Video senza testo**                                                        | **82** (81 senza metadati, 1 con metadati ma senza traccia: `6WIkVegHA1s`) |
| Coppie video + testo                                                         | **13.450**                                                             |
| Durata nel file `info` diversa da quella del file di oltre 2 s               | 7: `jyQg5Lyny0Q` (265 s dichiarati, 547,9 s nel file), `uu1FvPlZnG8` (124 / 144,6), `3jZkrtOtrfA` (314 / 320,0), `eofoHkOkASE`, `kn_0P5FKcrM`, `K1vAz1dcmAc`, `izES-hb4FQY` |
| Sottotitoli illeggibili (caratteri di sostituzione)                          | 4 (� presente già su YouTube, non da correggere)                       |
| Sottotitoli con cue oltre la fine del video (più di 5 s)                     | 53                                                                     |
| **Video identici (stesso contenuto)**                                        | **0** dopo la deduplica (erano 2 coppie, vedi sotto)                   |
| Video diversi con sottotitoli identici                                       | 58 gruppi (45 sulla sola traccia usata per le clip, 91 video)          |
| Video con frequenza di fotogrammi anomala (< 10 fps)                         | 8                                                                      |
| **Video che decord non apre** (il lettore del training)                      | **0**: i 176 AV1 sono stati ri-codificati e decord apre tutti e 176   |

**Cosa ne segue.**

1. **Duplicati (fatto).** Delle 2 coppie di video identici (`-urXXDI0OiE`/`z3thwypVsjw`, `EMqOLXi0nmw`/`XH6nn3kH4d4`) ne è tenuta una per coppia (`-urXXDI0OiE`, `EMqOLXi0nmw`); `z3thwypVsjw` e `XH6nn3kH4d4` (mp4, vtt e info) sono in `quarantine_duplicati/`, non cancellati. I gruppi con testo identico (video diversi) restano: vanno tenuti nello stesso split (lo fa la proposta di §4) e, per l'addestramento, scartati tranne uno per gruppo.
2. **AV1 (fatto).** Erano 176 video (ffprobe li dava validi, decord no). Sono stati ri-codificati in H.264 (`libx264`, crf 18, `yuv420p`, audio AAC 128k, stessa risoluzione e fps), durata uguale all'originale entro 1 s e decord li apre tutti (primo, centrale e ultimo fotogramma). Gli originali AV1 sono in `quarantine_av1_originali/` (5,6 GB), da cancellare dopo il controllo. Codec ora: H.264 13.300, VP9 236.
3. **Durata.** I 7 video elencati sopra vanno guardati a mano (file troncato o video sbagliato; `jyQg5Lyny0Q` è il caso più grave, il file dura più del doppio). Non sono stati toccati.
4. **Coppie video + testo.** La regola «per ogni video un testo» **non è ancora soddisfatta**: mancano gli 82 di cui sopra (81 rimossi o privati su YouTube). Esiste un'eccezione legittima, 602 video con tracce ma nessuna nella propria lingua (tradotte soltanto), che restano utilizzabili solo per il livello fisico (§3.3 del progetto).
5. **Tracce multiple.** 12.574 video hanno una traccia, 880 ne hanno da 2 a 61 (traduzioni e varianti `en-XXXX`). Il progetto tiene la sola traccia nella lingua propria (`own_language_track`).

## 3. Analisi dei dati

### 3.1 Video

- **1.088 ore** in 13.536 video: mediana 3,1 min, media 4,8 min, 95° percentile 14,6 min. L'1 % dei video più lunghi vale il 9 % delle ore, il 10 % ne vale il 38 %.
- **Risoluzione**: 720p o più per 11.529 video (85 %), il resto inferiore; il downloader limita a 720p. Codec: H.264 13.300, VP9 236 (AV1 non più presenti, §2).
- **Lingue**: `ase` (ASL) è il **42,7 % del rilascio** e il 38,2 % dei video scaricati (5.168); Gini sui video per lingua 0,76, equivalenti a 5,0 lingue ugualmente frequenti su 56; 17 lingue hanno meno di 100 video nel rilascio. Il progetto non ribilancia (§3.5): conta che le metriche medie descrivono soprattutto ASL e vanno riportate **anche per lingua**.
- **Bias di recupero**: la disponibilità dipende dalla lingua (chi² p < 1e-10, Cramér V 0,29; scostamento totale del mix di lingue 0,21), ma questo numero è inquinato dal rate-limit (§1) e dai 22.383 ID mai tentati. Da rimisurare.
- Ore stimate sull'intero rilascio: 3.297 contro le 3.200 dichiarate.

### 3.2 Testo

Le clip seguono `corpus.builders.YouTubeSL25Manifest`: **una clip per cue** della traccia nella lingua propria.

|                                                     | Valore                                                          |
| --------------------------------------------------- | --------------------------------------------------------------- |
| Clip (cue)                                          | **635.807** in 12.850 video, 870 ore di tempo coperto dalle cue |
| Clip per video                                      | mediana 28, media 49, 95° percentile 150, massimo 2.415         |
| Durata di una clip                                  | mediana 4,1 s, 5°–95° percentile 1,2–11,1 s, massimo 605 s      |
| Parole per clip                                     | mediana 8, 95° percentile 17 (5,3 milioni di parole in tutto)   |
| Cue sul tempo del video                             | mediana 85 %: le didascalie coprono quasi tutto il video        |
| Clip più brevi di 1 s                               | 19.827 (3,1 %); più lunghe di 20 s: 3.379 (0,5 %)               |
| Annotazioni non parlate (`[Music]`, `♪`)            | 3.209                                                           |
| Clip di una sola parola                             | 45.413 (7,1 %)                                                  |
| Clip che si sovrappongono alla successiva           | 2.633                                                           |
| Clip con testo ripetuto nello stesso video          | 17.580 (2,8 %)                                                  |
| Clip con stesso testo e stessi tempi in altri video | 5.612 (ricaricamenti)                                           |
| Clip oltre la fine del video                        | 594                                                             |

**Clip utilizzabili** (durata 1–20 s, non annotazioni; ora non serve più escludere gli AV1): **609.619, pari a 837,9 ore**. Per split proposti: train 492.716 (680,8 h), validation 66.684 (98,8 h), test 50.219 (58,2 h). È il **28 %** delle 2,16 milioni di coppie clip–testo che il progetto stima per il rilascio intero (§3.2 del progetto).

Avvertenze sul testo:

- **Le parole non misurano il testo per le lingue senza spazi.** Il giapponese (`jsl`: 21.462 clip e 28.010 parole, 1,3 per clip) va misurato in caratteri.
- **L'allineamento non è misurabile da qui.** Le cue sono tempi di sottotitolo, non tempi del segnato: in BOBSL il segnato arriva in media 2,7 s dopo (§3.3 del progetto). Con clip mediane di 4 s è un errore dello stesso ordine della clip. Va misurato sul dry run (PC7), non dal disco.
- **Le traduzioni sono rumorose** e alcune tracce sono marcate col codice della lingua dei segni (779 video per `ase`, in inglese).

### 3.3 Per lingua (le prime)

| Lingua | Video con testo | Clip    | Ore di cue | Canali | Canale più grande |
| ------ | --------------- | ------- | ---------- | ------ | ----------------- |
| `ase`  | 5.090           | 292.949 | 359        | 1.538  | 4 %               |
| `ils`  | 502             | 41.160  | 73         | 10     | **95 %**          |
| `bfi`  | 681             | 28.425  | 42         | 46     | 36 %              |
| `hsh`  | 1.262           | 27.293  | 46         | 9      | **81 %**          |
| `ise`  | 741             | 27.021  | 39         | 20     | 14 %              |
| `bzs`  | 397             | 26.381  | 40         | 26     | 14 %              |
| `jsl`  | 484             | 21.462  | 24         | 20     | 16 %              |
| `pso`  | 324             | 18.798  | 26         | 29     | 31 %              |

### 3.4 Figure e come leggerle

Le figure sono in `reports/youtube_sl25/` (generate da `youtube_sl25_report`). I codici sono ISO 639-3 (`ase` = ASL, `ils` = lingua dei segni internazionale, `hsh` = ungherese, ecc.). Tutte descrivono i **13.536 video scaricati** o il rilascio di 39.197 ID, non le clip. Le figure sono state rigenerate il 12/10 sui dati attuali; i numeri nei commenti sotto sono stati aggiornati dove citano conteggi del §1–§2, mentre i conteggi approssimativi (frame rate, aspect ratio, copertura delle didascalie, ranghi per lingua) risalgono al 10/10 e cambiano di poco.

#### Figura 1: video per lingua, rilascio contro scaricato

![Video per lingua](../reports/youtube_sl25/01_composition.png)

**Cosa mostra.** Per ogni lingua, una barra grigia (video nel rilascio) e una blu (video scaricati), su **scala logaritmica**, ordinate per dimensione.

**Come leggerla.** La parte grigia visibile oltre il blu è quello che non abbiamo. Sulla scala log un tratto di grigio apparentemente piccolo in alto vale migliaia di video (`ase`: 16.724 nel rilascio, 5.168 scaricati). Dove il blu copre il grigio (`svk`, `isg`, `ssr`, `asq`, `aed`) abbiamo quasi tutto; `sfb` non ha nessun video scaricato.

**Cosa ne segue.** Il sottoinsieme non è una copia ridotta del rilascio: alcune lingue sono molto più complete di altre (`hsh`, `ise`, `swl`, `isr`, `ssp` quasi complete; `gsg`, `fsl`, `rsl`, `pks` sotto il 20 %). Compaiono anche codici da normalizzare (`slovenia`, `vietnam`) e `???` (193 video con lingua ignota).

#### Figura 2: quanto è sbilanciato il corpus

![Curva di Lorenz e rango-frequenza](../reports/youtube_sl25/02_balance.png)

**Cosa mostra.** A sinistra la curva di Lorenz: in ascissa la quota di lingue (dalla più piccola), in ordinata la quota cumulata di video. A destra il grafico rango-frequenza (log-log): ogni punto è una lingua, in ascissa la posizione per numerosità, in ordinata i video; la retta rossa è l'adattamento a una legge di potenza.

**Come leggerla.** Con un corpus perfettamente bilanciato la curva sarebbe la diagonale tratteggiata; più è lontana, più è sbilanciato. L'area tra le due è il **Gini (0,76)**. Qui l'80 % delle lingue più piccole contiene circa il 20 % dei video, e l'ultimo tratto della curva, quasi verticale, è `ase` (42,7 %). A destra, una pendenza di −1,56 indica una coda pesante; la retta spiega bene il grosso (R² 0,91) ma i dati si discostano: la fascia centrale (rango 6–35) sta sopra la retta e le ultime lingue cadono sotto, cioè la coda è più corta di una legge di potenza pura.

**Cosa ne segue.** Una media sul corpus intero descrive soprattutto ASL e le prime cinque lingue (63,2 % dei video). Le metriche vanno riportate **per lingua**, non solo in media.

#### Figura 3: cosa cambia ribilanciando con la temperatura

![Effetto della temperatura di campionamento](../reports/youtube_sl25/08_sampling_temperature.png)

**Cosa mostra.** Se si campiona una lingua con probabilità `p ∝ nᵗ` (n = video della lingua, t = temperatura), a sinistra quante «lingue effettive» (inverso dell'indice di Simpson) ottengo al variare di t; a destra come cambia la probabilità delle sei lingue maggiori.

**Come leggerla.** t = 1 è il campionamento proporzionale ai dati (5,0 lingue effettive, `ase` al 42,7 %); t = 0 è uniforme (56 lingue effettive, 1,8 % ciascuna). Valori intermedi: t = 0,7 dà 14,1 lingue e `ase` al 22,2 %; t = 0,5 dà 27,8 e 12,4 %; t = 0,3 dà 44,5 e 6,2 %.

**Cosa ne segue.** Abbassare t dà più peso alle lingue piccole, ma le fa ripetere molte volte (rischio di memorizzazione di pochi video). È un compromesso da scegliere: il progetto, per ora, non ribilancia (§3.5), ma la figura dice quanto costerebbe farlo.

#### Figura 4: cosa è stato recuperato e con che bias

![Disponibilità per lingua e stato di recupero](../reports/youtube_sl25/04_retrieval.png)

**Cosa mostra.** A sinistra, per ogni lingua, la frazione `scaricati / (scaricati + non disponibili)` con il suo intervallo di confidenza di Wilson al 95 %. A destra, il conteggio degli ID per esito (scaricato 13.536, non disponibile 2.640, bot 561, altro 77, mai provato 22.383).

**Come leggerla.** Il punto è la stima, la barra l'incertezza: le barre lunghe sono lingue con pochi video. Se la disponibilità non dipendesse dalla lingua, tutti i punti starebbero intorno alla stessa frazione; invece vanno da 1,0 (`ssr`, `isg`, `svk`) a 0 (`sfb`), con `ase` intorno a 0,31. È l'effetto che il test chi² misura (p < 1e-10, Cramér V 0,29, scostamento totale del mix 0,21).

**Cosa ne segue e cautela.** Il sottoinsieme non rappresenta il rilascio. Ma il 64,4 % di «non disponibili» è inquinato dal rate-limit di YouTube (§1): è un limite superiore, e il bias va **rimisurato dopo un download pulito**, non usato così com'è.

#### Figura 5: durate dei video

![Durate: istogramma, ECDF e per lingua](../reports/youtube_sl25/05_durations.png)

**Cosa mostra.** A sinistra la distribuzione delle durate (asse x logaritmico) con l'adattamento log-normale (KS 0,01); al centro le curve cumulative della quota di **video** e della quota di **ore** in funzione della durata; a destra i boxplot (scatola = quartili, riga arancione = mediana) per le 15 lingue maggiori, in scala log.

**Come leggerla.** Il log-normale descrive bene il corpo (KS basso, mediana 3,1 min), ma c'è un eccesso di video molto brevi (sotto ~0,3 min) che la curva non spiega. Al centro, la curva delle ore sta a destra di quella dei video: metà dei video dura meno di circa 3 minuti, ma servono video fino a circa 7 minuti per arrivare a metà delle ore; l'1 % dei video più lunghi vale il 9 % delle ore, il 10 % ne vale il 39 %. A destra, la durata dipende dalla lingua (Kruskal-Wallis ε² 0,14): `ils` ha mediana alta, `hsh` e `ssp` bassa.

**Cosa ne segue.** Pochi video lunghi pesano molto in ore: campionare per video e per ore dà mix diversi. I video brevissimi vanno controllati.

#### Figura 6: video contro ore, e durata mediana per lingua

![Video contro ore e durata mediana](../reports/youtube_sl25/03_hours_vs_videos.png)

**Cosa mostra.** A sinistra, per lingua, la quota di video scaricati (asse x) contro la quota di ore (asse y), in log-log. A destra la durata mediana per le lingue con almeno 20 video.

**Come leggerla.** Un punto sulla diagonale ha ore in proporzione ai video; **sopra** la diagonale i video sono più lunghi della media (`ils`: circa il 4 % dei video ma l'8 % delle ore), **sotto** più corti (`hsh`: circa il 9 % dei video e il 5 % delle ore). A destra le mediane vanno da circa 10 min (`csn`) a poco più di 1 (`???`, `ssp`, `hsh`).

**Cosa ne segue.** «Quanti dati ha una lingua» dipende dall'unità: per `ils` le ore sono il doppio di quanto suggerisce il numero di video, per `hsh` circa la metà. Per decisioni sul mix (§3.5) conviene ragionare in **ore o clip**, non in video.

#### Figura 7: qualità tecnica

![Altezza, fps, aspect ratio e bit per pixel](../reports/youtube_sl25/06_technical.png)

**Cosa mostra.** Quattro istogrammi sui video scaricati: altezza del fotogramma, frame rate, rapporto d'aspetto, bit per pixel per fotogramma (proxy del dettaglio). Frame rate e rapporto d'aspetto hanno l'asse y in scala log.

**Come leggerla.** L'altezza è troncata a 720p dal downloader (circa 85 % dei video a 720 o più), quindi non dice nulla sulle risoluzioni originali superiori. Il frame rate è **misto**: 30 fps (circa 7.000 video), 25 (circa 3.400), 24 (circa 1.200), 50 e 60 (circa 600 ciascuno) e una coda di valori rari, 8 video sotto i 10 fps. Il rapporto d'aspetto è quasi sempre 16:9 (circa 12.000), poi 4:3, verticale, quadrato, ultralargo. I bit per pixel hanno una campana intorno a 0,05: la coda sinistra sono video molto compressi (mani sfocate), la destra video ricchi di dettaglio.

**Cosa ne segue.** Serve un ricampionamento a frame rate fisso e una regola di ritaglio coerente per i formati non 16:9 (un ritaglio centrale su video verticali o 4:3 può tagliare le mani). I bit per pixel sono solo un'approssimazione: non misurano la qualità né la visibilità delle mani.

#### Figura 8: risoluzione per lingua

![Mix di risoluzioni per lingua](../reports/youtube_sl25/07_language_resolution.png)

**Cosa mostra.** Una matrice: ogni riga è una lingua (le 20 maggiori), ogni colonna una classe di altezza, il colore è la quota dei video della lingua in quella classe (ogni riga somma a 1).

**Come leggerla.** Quasi tutto è nella colonna 720–1079 (colore scuro). Le righe più chiare in quella colonna hanno risoluzioni più basse: `ssp` ha circa il 60 % a 720p e il resto a 360–719; `dse` e `jsl` hanno una quota evidente a 480–719 o 360–479; `ins`, `pso`, `pks`, `isr`, `icl` sono quasi tutte a 720p.

**Cosa ne segue.** La risoluzione è **correlata con la lingua** (in pratica con i canali di origine): un modello potrebbe imparare la qualità dell'immagine come scorciatoia per riconoscere la lingua. Va controllato nelle valutazioni per lingua e, se serve, compensato con augmentation di qualità.

#### Figura 9: didascalie

![Copertura e densità delle didascalie](../reports/youtube_sl25/09_subtitles.png)

**Cosa mostra.** A sinistra la copertura (quota della durata del video coperta da cue di sottotitolo); a destra la densità (parole per secondo di didascalia).

**Come leggerla.** La copertura ha il picco intorno a 0,85–0,9 (mediana 84 %): quasi sempre una parte del video resta senza testo, con una coda verso lo zero (tracce quasi vuote). Valori leggermente sopra 1 indicano cue che superano la fine del video (53 casi oltre 5 s, §2). La densità ha picco tra 1 e 2 parole al secondo (mediana 1,64); i valori sopra 5 sono rari e vanno ispezionati.

**Cosa ne segue.** Queste figure dicono **quanto** testo c'è, non se è **allineato** al segnato: le cue sono tempi di sottotitolo, non del segnato (§3.2). Per le lingue senza spazi (`jsl`) le parole al secondo non sono una misura valida: servono i caratteri.

## 4. Gli split

**Il dataset non ha split predefiniti.** Il CSV del rilascio ha solo `video_id,language`, e il README e il paper non ne definiscono: YouTube-SL-25 è un corpus di pre-addestramento, i risultati si riportano su benchmark esterni. Va quindi definito da noi, ed è una decisione scientifica: il progetto l'ha già fissata (§3.4: **per canale, mai per clip**; validation held-out-channel e held-out-language).

### 4.1 Com'è fatto il corpus rispetto a quel vincolo

- **1.958 canali** per 13.455 video: mediana **1 video per canale**, 1.131 canali (58 %) con un solo video, media 6,9, massimo 1.019. Gini 0,77. I 10 canali maggiori valgono il 23,4 % dei video, l'1 % dei canali il 32 %.
- **Concentrazione per lingua**: `ase` ha 1.538 canali e nessuno pesa più del 4 %; le altre lingue ne hanno pochissimi. **`ils` ha 10 canali ma uno solo pesa il 95 %; `hsh` ne ha 9 e uno pesa l'81 %.** Per queste lingue «held-out channel» e «held-out language» coincidono di fatto: non c'è un secondo canale su cui validare.
- 15 canali pubblicano in più di una lingua dei segni.
- Date di caricamento 2008–2024, con il picco nel 2020 (2.294 video).
- **Perché per canale**: con uno split casuale per video, **1.259 dei 1.394 video di validation (90 %) condividono il canale con il train**, cioè stesso segnante, stesso sfondo, stessa luce.

### 4.2 La proposta (`reports/youtube_sl25/splits_proposal.csv`)

Deterministica (hash SHA-256 del canale con sale per lingua: stabile se arrivano nuovi video):

1. train / validation / test = 80 / 10 / 10 **per canale**, stratificato per lingua;
2. i canali legati da un duplicato (stesso file o stesso testo) sono fusi e vanno insieme (10 gruppi);
3. una lingua con meno di 10 canali (dopo la fusione) è solo train: 42 etichette di lingua, tra cui `???`, `slovenia` e `vietnam` che andrebbero prima normalizzate;
4. i video senza metadati (81) non hanno canale: **non assegnati**, fuori da ogni split finché non si recuperano.

| Split         | Video  | Canali | Clip utilizzabili | Ore | Lingue |
| ------------- | ------ | ------ | ----------------- | --- | ------ |
| train         | 11.437 | 1.586  | 492.716           | 681 | 55     |
| validation    | 1.179  | 173    | 66.684            | 99  | 11     |
| test          | 839    | 199    | 50.219            | 58  | 10     |
| non assegnati | 81     | n/d    | 0                 | n/d | 7      |

**Limiti da tenere presenti.**

- Validation e test coprono **solo 12 lingue**, e il 57 % dei video di validation è `ase` (test 54 %, train 35 %). Dice come generalizza a canali nuovi, non a lingue nuove.
- Il 10 % dei **canali** non è il 10 % dei **dati** (validation 11 %, test 8 % delle clip): con canali di dimensioni così diverse la quota si ottiene solo in media.
- `held_out_languages` (§3.4) non è definibile da questi dati senza scegliere: tenere fuori `ils` o `hsh` toglie il 95 % o l'81 % dei loro dati, ed è una scelta di progetto, non statistica.
- La validation «canali visti» (`seen_video_fraction: 0.02`) si estrae dal train e non è nella proposta.

### 4.3 Come procedere

1. **Terminare il recupero**: scaricare gli ID mai tentati (22.383) e riprovare i «non disponibili» (2.640) e i blocchi bot (561) con un cookie valido e il downloader con pause (con `-t sleep`, come suggerisce yt-dlp). Gli 81 video già presenti senza testo sono quasi tutti rimossi da YouTube. Rieseguire integrità, report e split: la proposta è deterministica, cambiano solo le assegnazioni dei canali nuovi.
2. ~~Sistemare i 174 AV1~~ e ~~deduplicare le 2 coppie di video~~: **fatto** (§2). Restano i 7 video con durata diversa da guardare a mano, i 4 VTT con � (da tenere o correggere a mano) e i 45 gruppi di testo identico da ridurre a un video.
3. **Congelare** l'elenco degli ID e `splits_proposal.csv` (il progetto lo richiede, §3.9): da quel momento non si ricalcola.
4. **Decidere i due punti aperti**: soglia di canali per tenere fuori una lingua (ora 10), e se tenere un held-out language (e quale).
5. **Audit di contaminazione con OpenASL** (P2): `ase` condivide i due canali maggiori con OpenASL. Prima di usare `ase` come validation, rimuovere dal corpus i segmenti sovrapposti a clip di test di OpenASL; non l'ho fatto qui, richiede il manifest di OpenASL.
6. **Misurare l'allineamento** testo–segnato su un campione prima di fidarsi delle 610 mila clip come coppie (PC7).

## 5. Come si rifà

```bash
python -m signworld.data.analysis.youtube_sl25_integrity --root ~/dataset/youtube-sl-25 --out reports/youtube_sl25
python -m signworld.data.analysis.youtube_sl25_report    --root ~/dataset/youtube-sl-25 --out reports/youtube_sl25 --reuse-probe
python -m signworld.data.analysis.youtube_sl25_text      --root ~/dataset/youtube-sl-25 --out reports/youtube_sl25
```

Il report richiede `matplotlib` e `scipy`, che non sono dipendenze di `pyproject.toml`. Il controllo di decodifica con decord non è nello script di integrità: è stato eseguito a parte sui 176 AV1 ri-codificati (176 aperti, 0 falliti); `reports/youtube_sl25/decode_check.json` è del 10/10 (prima della ri-codifica) e non è stato rigenerato. `usable_clips.json` è rigenerato (stessa definizione, senza esclusione AV1).

Per copiare la cartella su un altro server (addestramento multinodo): `slurm/sync_dataset.sh` (rsync riprendibile, `-p N` per più processi, `-n` per la prova, `-c` per la verifica finale).
