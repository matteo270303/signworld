# YouTube-SL-25: analisi del dataset

## Diagnostiche

- **alto** · `ase` è il 43% del rilascio: una media sul corpus intero descrive soprattutto quella lingua.
- **alto** · Gini sui video per lingua 0.76; equivalgono a 5.0 lingue ugualmente frequenti su 56.
- **medio** · 17 lingue hanno meno di 100 video nel rilascio: troppo poche per addestrare o valutare da sole, utili solo in un mix.
- **medio** · Codici lingua non ISO 639-3: slovenia, vietnam. Vanno normalizzati prima di raggruppare per lingua.
- **medio** · 197 video (0.5%) hanno lingua `???`.
- **alto** · La disponibilità dipende dalla lingua (chi² p=<1e-10, Cramér V=0.29): il sottoinsieme scaricato non rappresenta il rilascio.
- **info** · Distanza di variazione totale tra mix di lingue del rilascio e dello scaricato: 0.210 (KL 0.154 nat).
- **medio** · 22383 ID non sono ancora stati tentati: i numeri sullo scaricato cambieranno a download finito.
- **info** · Ore stimate sull'intero rilascio (durata media scaricata × conteggio): 3,297 h contro le 3,200 dichiarate (103%).
- **info** · La durata dei video varia con la lingua (Kruskal-Wallis p=<1e-10, ε²=0.14).

## 1. Com'è fatto

Il rilascio elenca **39,197** video unici in **56** lingue dei segni. Ogni riga del CSV ha solo `video_id` e `language`: nessuna durata, split, canale, firmatario o data. Le didascalie non sono nel CSV; si scaricano da YouTube insieme al video.

| lingua | video | quota | scaricati | ore scaricate | durata media (min) |
|---|---|---|---|---|---|
| ase | 16,724 | 42.7% | 5,168 | 446 | 5.2 |
| ins | 3,023 | 7.7% | 490 | 38 | 4.7 |
| pso | 1,698 | 4.3% | 327 | 30 | 5.6 |
| hsh | 1,687 | 4.3% | 1,262 | 54 | 2.6 |
| ils | 1,634 | 4.2% | 507 | 87 | 10.3 |
| asf | 1,098 | 2.8% | 438 | 25 | 3.5 |
| jsl | 1,075 | 2.7% | 485 | 31 | 3.8 |
| bfi | 1,026 | 2.6% | 681 | 49 | 4.3 |
| gsg | 1,024 | 2.6% | 146 | 17 | 7.0 |
| ise | 929 | 2.4% | 747 | 47 | 3.8 |
| fsl | 900 | 2.3% | 149 | 9 | 3.6 |
| bzs | 846 | 2.2% | 398 | 51 | 7.6 |
| rsl | 715 | 1.8% | 107 | 7 | 3.9 |
| ssp | 701 | 1.8% | 524 | 25 | 2.9 |
| pks | 581 | 1.5% | 80 | 4 | 2.6 |

![composizione](01_composition.png)

## 2. Bilanciamento

| indice | valore |
|---|---|
| Gini (video per lingua) | 0.759 |
| entropia normalizzata | 0.641 |
| lingue effettive (inverso di Simpson) | 5.0 |
| quota della prima lingua | 42.7% |
| quota delle prime 5 | 63.2% |
| pendenza Zipf (log-log) | -1.56 (R² 0.91) |
| lingue con < 100 video | 17 |
| lingue con < 500 video | 41 |

Campionando con `p ∝ nᵗ` il mix cambia così:

| t | lingue effettive | quota della prima | entropia norm. |
|---|---|---|---|
| 1 | 5.0 | 42.7% | 0.64 |
| 0.7 | 14.1 | 22.2% | 0.83 |
| 0.5 | 27.8 | 12.4% | 0.92 |
| 0.3 | 44.5 | 6.2% | 0.97 |
| 0 | 56.0 | 1.8% | 1.00 |

![bilanciamento](02_balance.png)
![temperatura](08_sampling_temperature.png)

## 3. Cosa è stato recuperato e con che bias

| stato | ID | quota |
|---|---|---|
| downloaded | 13,536 | 34.5% |
| corrupt | 0 | 0.0% |
| unavailable | 2,640 | 6.7% |
| bot | 561 | 1.4% |
| other | 77 | 0.2% |
| untried | 22,383 | 57.1% |

Tra gli ID con esito noto (scaricato o non disponibile), la disponibilità dipende dalla lingua: chi² = 1367, p = <1e-10, Cramér V = 0.29. Distanza di variazione totale del mix di lingue: 0.210. Le cause di fallimento sono unite su più run e alcune vecchie possono essere sbagliate (cookie scaduti): gli ID `unavailable` di run vecchi possono essere ritentati con cookie validi, e questo bias va rimisurato a download finito.

![recupero](04_retrieval.png)

## 4. Durate

13,536 video validi, **1,088 h**. Mediana 3.1 min, media 4.8 min, 5°–95° percentile 0.6–14.6 min, 99° 30 min. Il log-normale ha μ=5.20, σ=0.96 sui secondi (KS=0.01: valori alti indicano che non è una buona descrizione). L'1 % dei video più lunghi vale il 9% delle ore, il 10 % il 38%.

![durate](05_durations.png)
![ore e video](03_hours_vs_videos.png)

Ore stimate sull'intero rilascio per le 10 lingue maggiori (assume che la disponibilità non dipenda dalla durata):

| lingua | ore stimate | intervallo 95 % |
|---|---|---|
| ase | 1,442 | 1,390–1,493 |
| ils | 281 | 261–300 |
| ins | 234 | 223–245 |
| pso | 157 | 140–177 |
| gsg | 120 | 92–156 |
| bzs | 107 | 100–115 |
| bfi | 73 | 67–79 |
| hsh | 72 | 67–76 |
| jsl | 69 | 64–74 |
| asf | 64 | 57–73 |

## 5. Qualità tecnica

Altezza: <240 0, 240-359 52, 360-479 932, 480-719 1,023, 720-1079 11,529, >=1080 0. Senza audio: 0. Codec video: h264 13,300, vp9 236. Il downloader limita a 720p, quindi la distribuzione dell'altezza è troncata dall'alto.

![tecnica](06_technical.png)
![risoluzione](07_language_resolution.png)

## 6. Anomalie

8 file anomali (`anomalies.csv`), 0 gruppi di probabili duplicati.

## 7. Didascalie

15,171 file di sottotitoli. Copertura mediana del video 84%; parole al secondo mediane 1.65.

![didascalie](09_subtitles.png)

## 8. Per quali task, e cosa non si può dire

Dalla struttura dei dati, non dal paper: il corpus offre video con etichetta di lingua dei segni e, dove le didascalie sono presenti, testo con tempi. Questo regge (a) identificazione della lingua dei segni, (b) pre-addestramento multilingue su video, (c) traduzione o retrieval segno→testo se le didascalie sono allineate. Il CSV non dà split, canale, firmatario né data, per cui: dividere per ID video evita solo il leakage più ovvio; i ricaricamenti dello stesso contenuto e i video dello stesso canale possono finire su lati diversi dello split. Per analizzare l'andamento nel tempo o per canale servono i metadati di YouTube (`--write-info-json`).
