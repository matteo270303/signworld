# YouTube-SL-25: analisi del dataset

## Diagnostiche

- **alto** · `ase` è il 43% del rilascio: una media sul corpus intero descrive soprattutto quella lingua.
- **alto** · Gini sui video per lingua 0.76; equivalgono a 5.0 lingue ugualmente frequenti su 56.
- **medio** · 17 lingue hanno meno di 100 video nel rilascio: troppo poche per addestrare o valutare da sole, utili solo in un mix.
- **medio** · Codici lingua non ISO 639-3: slovenia, vietnam. Vanno normalizzati prima di raggruppare per lingua.
- **medio** · 197 video (0.5%) hanno lingua `???`.
- **alto** · La disponibilità dipende dalla lingua (chi² p=<1e-10, Cramér V=0.43): il sottoinsieme scaricato non rappresenta il rilascio.
- **info** · Distanza di variazione totale tra mix di lingue del rilascio e dello scaricato: 0.214 (KL 0.160 nat).
- **medio** · 2 ID non sono ancora stati tentati: i numeri sullo scaricato cambieranno a download finito.
- **info** · Ore stimate sull'intero rilascio (durata media scaricata × conteggio): 3,299 h contro le 3,200 dichiarate (103%).
- **info** · La durata dei video varia con la lingua (Kruskal-Wallis p=<1e-10, ε²=0.14).
- **medio** · 2 gruppi di ID con stessa dimensione e durata: probabili ricaricamenti dello stesso video (rischio di leakage tra split).

## 1. Com'è fatto

Il rilascio elenca **39,197** video unici in **56** lingue dei segni. Ogni riga del CSV ha solo `video_id` e `language`: nessuna durata, split, canale, firmatario o data. Le didascalie non sono nel CSV; si scaricano da YouTube insieme al video.

| lingua | video | quota | scaricati | ore scaricate | durata media (min) |
|---|---|---|---|---|---|
| ase | 16,724 | 42.7% | 5,096 | 440 | 5.2 |
| ins | 3,023 | 7.7% | 471 | 37 | 4.7 |
| pso | 1,698 | 4.3% | 323 | 30 | 5.6 |
| hsh | 1,687 | 4.3% | 1,259 | 54 | 2.6 |
| ils | 1,634 | 4.2% | 498 | 85 | 10.3 |
| asf | 1,098 | 2.8% | 434 | 25 | 3.5 |
| jsl | 1,075 | 2.7% | 481 | 31 | 3.9 |
| bfi | 1,026 | 2.6% | 682 | 49 | 4.3 |
| gsg | 1,024 | 2.6% | 136 | 16 | 7.0 |
| ise | 929 | 2.4% | 747 | 47 | 3.8 |
| fsl | 900 | 2.3% | 146 | 9 | 3.6 |
| bzs | 846 | 2.2% | 396 | 50 | 7.6 |
| rsl | 715 | 1.8% | 106 | 7 | 3.9 |
| ssp | 701 | 1.8% | 523 | 25 | 2.9 |
| pks | 581 | 1.5% | 75 | 3 | 2.7 |

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
| downloaded | 13,377 | 34.1% |
| corrupt | 0 | 0.0% |
| unavailable | 25,238 | 64.4% |
| bot | 162 | 0.4% |
| other | 418 | 1.1% |
| untried | 2 | 0.0% |

Tra gli ID con esito noto (scaricato o non disponibile), la disponibilità dipende dalla lingua: chi² = 7264, p = <1e-10, Cramér V = 0.43. Distanza di variazione totale del mix di lingue: 0.214. Le cause di fallimento sono unite su più run e alcune vecchie possono essere sbagliate (cookie scaduti): gli ID `unavailable` di run vecchi possono essere ritentati con cookie validi, e questo bias va rimisurato a download finito.

![recupero](04_retrieval.png)

## 4. Durate

13,377 video validi, **1,074 h**. Mediana 3.1 min, media 4.8 min, 5°–95° percentile 0.6–14.5 min, 99° 30 min. Il log-normale ha μ=5.20, σ=0.96 sui secondi (KS=0.01: valori alti indicano che non è una buona descrizione). L'1 % dei video più lunghi vale il 9% delle ore, il 10 % il 39%.

![durate](05_durations.png)
![ore e video](03_hours_vs_videos.png)

Ore stimate sull'intero rilascio per le 10 lingue maggiori (assume che la disponibilità non dipenda dalla durata):

| lingua | ore stimate | intervallo 95 % |
|---|---|---|
| ase | 1,445 | 1,396–1,504 |
| ils | 280 | 262–299 |
| ins | 235 | 224–247 |
| pso | 158 | 141–178 |
| gsg | 120 | 91–157 |
| bzs | 107 | 101–116 |
| bfi | 74 | 68–81 |
| hsh | 72 | 67–77 |
| jsl | 69 | 64–75 |
| asf | 64 | 56–74 |

## 5. Qualità tecnica

Altezza: <240 0, 240-359 52, 360-479 930, 480-719 1,009, 720-1079 11,386, >=1080 0. Senza audio: 0. Codec video: h264 12,967, vp9 236, av1 174. Il downloader limita a 720p, quindi la distribuzione dell'altezza è troncata dall'alto.

![tecnica](06_technical.png)
![risoluzione](07_language_resolution.png)

## 6. Anomalie

7 file anomali (`anomalies.csv`), 2 gruppi di probabili duplicati.

Primi gruppi di duplicati: -urXXDI0OiE, z3thwypVsjw; EMqOLXi0nmw, XH6nn3kH4d4

## 7. Didascalie

13,697 file di sottotitoli. Copertura mediana del video 84%; parole al secondo mediane 1.64.

![didascalie](09_subtitles.png)

## 8. Per quali task, e cosa non si può dire

Dalla struttura dei dati, non dal paper: il corpus offre video con etichetta di lingua dei segni e, dove le didascalie sono presenti, testo con tempi. Questo regge (a) identificazione della lingua dei segni, (b) pre-addestramento multilingue su video, (c) traduzione o retrieval segno→testo se le didascalie sono allineate. Il CSV non dà split, canale, firmatario né data, per cui: dividere per ID video evita solo il leakage più ovvio; i ricaricamenti dello stesso contenuto e i video dello stesso canale possono finire su lati diversi dello split. Per analizzare l'andamento nel tempo o per canale servono i metadati di YouTube (`--write-info-json`).
