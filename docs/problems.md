# WorldSign — Problemi aperti

Raccolti il 18/10/2026 durante la prova OpenASL v5 (`runs/worldsign-openasl-trial-v5`, braccio A). I numeri sono quelli dell'epoca 6 (passo 4.307), salvo dove indicato. Ogni voce dice cosa si osserva, da quale componente dipende e come verificarlo. Da discutere a fine training, prima di toccare il codice.

## Livello fisico

**1. Il bersaglio di posa non contiene la velocità.**
- Osservato: R² della velocità da s fra 0,05 e 0,25, contro 0,83–0,97 per la posizione.
- Componenti:
  - l'àncora chiede solo la posizione media dello step e prende il 60% del gradiente della posa;
  - il rumore delle viste è indipendente per frame, quindi `L_inv` premia chi ignora le differenze fra frame;
  - i frame sono scelti per moto cumulato e senza Δt, il che appiattisce lo spostamento fra frame.
- Verifica: R² della velocità da (s_t, s_{t+1}) e dalle feature d'ingresso di G_ω; ripetere sulle pose `contiguous_`.

**2. Il bersaglio è anisotropo.**
- Osservato: IsoScore di s pari a 0,05; `SIGReg_posa` a 2,1, contro 1,06 per una gaussiana.
- Componenti: l'àncora domina; `SIGReg_posa` ha peso 0,04 su gruppi di al più 128 clip per step, in 192 dimensioni.

**3. Il bersaglio si muove ancora.**
- Osservato: la correlazione dei canali di s fra due epoche è 0,90 (0,92 dopo la rotazione migliore): cambia il contenuto, non solo la base.
- Componenti:
  - s è lo stato corrente di G_ω, senza EMA e senza congelamento;
  - l'LR della posa segue un coseno fino alla fine della run ed è ancora al 78% del picco.

**4. La predizione fisica è debole.**
- Osservato: `E_fis` ferma a 0,60, contro circa 0,80 se si predice zero; R² di validazione sugli step nascosti 0,36; rapporto di varianza γ 0,50.
- Componenti:
  - le maschere a tubo coprono tutti i 32 step, quindi una mano che resta nella zona coperta non si vede mai: è una parte irriducibile dell'errore;
  - il peso n^m dà più importanza agli step più nascosti;
  - la media nel box perde la configurazione della mano;
  - il bersaglio è mobile (punto 3).

**5. Possibile fuga di informazione dai box.**
- Osservato: errore sui keypoint del modello 0,149, contro 0,83 per l'interpolazione.
- Componente: i box sono calcolati dagli stessi keypoint del bersaglio e sono noti anche negli step nascosti, quindi il read-out sa dove si trova la mano anche senza vederla. s è fatto soprattutto di posizione.
- Verifica (**da fare per prima**):
  - una ridge dai soli box a LN(s_t) e ai keypoint;
  - il read-out con il video di un'altra clip e gli stessi box.

**6. Il read-out degli step quasi visibili non funziona.**
- Osservato: R² fra −4 e −57; il margine della dinamica è instabile.
- Componenti:
  - la testa lineare è addestrata su box fatti quasi solo di token mascherati, quindi i token visibili sono fuori distribuzione;
  - con maschere al 70–90% condivise dal batch, gli step quasi visibili sono pochissimi e la misura è inaffidabile.

**7. La LoRA dell'encoder deriva.**
- Osservato: 12,9%, sopra l'allarme del 10%.
- Componenti:
  - l'unico gradiente sull'encoder è `E_fis`, verso un bersaglio mobile;
  - la LoRA ha lo stesso LR delle teste (2e-4);
  - la fusione MLP parte da zero e ha avuto una sola epoca di F0.
- Effetto indiretto: il livello semantico legge lo stesso encoder con lo stop-gradient e deve inseguirlo.

**8. Difetti delle misure.**
- L'errore sui keypoint riporta la predizione alla scala di s usando media e deviazione standard della vera s.
- Le soglie di confidenza sono 1,0 per i giunti e 0,3 per i box: un box può esistere senza giunti, e viceversa.

Non sono cause probabili:
- la capacità di G_ω: l'àncora è a 0,025 e il pavimento del decoder a 0,05;
- l'assenza di attenzione fra le parti e il viso senza sopracciglia né occhi. Limitano il contenuto di s, ma non spiegano le metriche.

**9. Da rivalutare: l'encoder di posa e la sua interazione con l'encoder video.**
- Cosa chiede G_ω (àncora, viste, `SIGReg_posa`).
- Come lo legge il livello fisico (box, media, LN, L1, peso degli step).
- Quando si ferma il bersaglio.
- Quanto `E_fis` può muovere l'encoder condiviso con il livello semantico.

## Livello semantico

**10. Le query collassano.**
- Osservato: `query_cosine` 0,97 e in crescita; `order_cosine` 0,97, cioè il predictor ignora l'ordine temporale.
- Componente: le 8 query apprese (inizializzate a N(0, 0,02²)) vengono mediate in un solo ŷ con K = 1. La loss vede solo la media e nulla le spinge a differenziarsi.
- Da rivalutare come si costruiscono le query.

La possibile soluzione é applicare un at pool e un projector finale togliendo queste brutte query, come fanno in vl jepa 

**11. R@1 basso in proporzione a R@5 e R@10.**
- Osservato: R@1 circa 21% contro R@10 circa 54% (MedR 8–10).
- Ipotesi da valutare: problemi su clip particolari, brevi o con didascalie corte.
- Prima lettura (fine epoca 7), che va nel senso opposto:
  - le fasce 0–2 s e 1–5 parole hanno l'R@1 più alto (T2V 0,39 e 0,41; V2T 0,36 e 0,36);
  - le più basse sono 6–10 parole (0,22 e 0,17) e 2–5 s (0,26 e 0,19).
- Attenzione: le didascalie corte sono spesso ripetute, e la validazione conta come corrette le didascalie uguali, il che gonfia le loro fasce.
- Da valutare:
  - R@1 per frequenza della didascalia;
  - quasi-duplicati e parafrasi nella galleria;
  - gli errori «video giusto, frase sbagliata»;
  - il peso di ‖ŷ‖² nell'energia (in T2V: R@1 per energia 20,8% contro 24,0% per coseno).

## Dalla fine della run (19/10/2026, passo 10.770)

**12. La loss della posa è sbilanciata.**
- Osservato, sui valori pesati: 0,96·`L_anchor` = 0,018, 0,96·`L_inv` = 0,004, 0,04·`SIGReg_posa` = 0,070. `SIGReg_posa` vale il 76% della loss di posa.
- Quote del gradiente su G_ω: àncora da 86% a 36%, SIGReg da 14% a 58%, invarianza da 0% a 6%. I coseni fra i gradienti sono circa 0: non c'è conflitto, solo pesi diversi.
- Lettura:
  - l'àncora è già risolta (1,7% della varianza dei keypoint);
  - l'invarianza non porta quasi segnale, perché le viste sono troppo facili;
  - resta attivo solo `SIGReg_posa`, che però non corregge l'anisotropia: rango effettivo di s 40 su 192, IsoScore 0,06, mentre la statistica per step sulla vista pulita è 0,96 volte quella gaussiana. Lungo una direzione casuale in 192 dimensioni la varianza è circa la media degli autovalori, quindi le proiezioni casuali vedono poco l'anisotropia.
- Componenti: λ_P = 0,04 viene da LeJEPA, dove il termine predittivo è l'invarianza fra viste. Qui la parte predittiva è àncora + invarianza, entrambe quasi a zero; in più le viste sono deboli.
- Da valutare: bilanciare i termini sul gradiente o sul valore iniziale, rafforzare le viste, misurare l'isotropia di s e non solo SIGReg.

**13. Il retrieval non era ancora a convergenza.**
- Osservato: l'R@1 di validazione è salito fino all'ultima epoca:
  - epoca 13: 36,7 / 31,6;
  - modello finale: 40,1 / 33,8, con intervallo al 95% di 37,0–43,4 in T2V e 30,9–37,0 in V2T;
  - circa +1 punto per epoca nelle epoche 10–13, poi +3,2 / +1,6 nel cooldown;
  - nello stesso tempo il rango effettivo di ŷ è passato da 104 a 196 e `y_sigreg` da 80 a 5,6.
- Da tenere in conto:
  - 15 epoche non dicono dov'è il tetto;
  - confronti fra bracci a parità di epoche premiano chi converge prima.
- Allo stesso tempo cresce il divario fra training e validazione:
  - R@1 sul sottoinsieme di training 75,6 / 54,2, con un divario di 35 / 20 punti (era 5 / 2 all'epoca 6);
  - `E_sem` di validazione è ferma dall'epoca 8 (da 1,29 a 1,27), mentre quella di training scende (da 1,20 a 1,09).
- Da valutare per YouTube-SL-25 (circa 23 volte le clip di OpenASL): la durata, il cooldown, la regolarizzazione del livello semantico e lo stop sul criterio.
