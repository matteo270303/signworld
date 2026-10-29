# TODO — encoder di posa e livello semantico

Modifiche da fare dopo la prova OpenASL v5 (problemi in `docs/problems.md`). Niente è ancora implementato.

## Encoder di posa (G_ω)

1. **Tempo sui 64 frame.** Trasformatori spaziali per frame e trasformatore temporale sui 64 frame, non sui 32 step.
2. **Fusione a 32 step solo alla fine.** `s̃_t = W_R[z_{2t−1}; z_{2t}]` (1024 → 512), poi LN e Linear → 192. La concatenazione conserva il verso del moto; s_t resta allineato al tubelet video.
3. **Tempo reale.** Portare nel batch l'istante τ_f di ogni frame (`frame_indices` / fps). Nel trasformatore temporale RoPE 1D su τ_f al posto delle 32 posizioni apprese; canale di velocità diviso per Δτ_f.
4. **Niente decoder.** Togliere `L_anchor` e D. Posizione e velocità si leggono con sonde ridge senza gradiente.
5. **Loss di LeJEPA.** `(1−λ)·L_inv + λ·SIGReg`, λ = 0,05. Nessuna energia fra t e t+1.
6. **SIGReg con più campioni.** 4 step a caso per clip, ad almeno 8 step di distanza: N = 512 per vista invece di 128 o meno.
7. **Viste più difficili che non cancellano la velocità:**
   - camera sempre: rotazione ±20°, scala 0,8–1,2, shear ±0,3;
   - rumore di Ornstein–Uhlenbeck sul tempo reale (ℓ = 0,5 s, σ tre volte quello di oggi);
   - occlusione di un articolatore intero (8–32 frame) e del 30–50% dei giunti di dita e viso;
   - perdita della detection con p 0,1;
   - niente deformazioni del tempo, niente ribaltamenti.
8. **Addestramento a parte, poi congelato.** Pre-addestramento solo sulla posa, poi bersaglio fisso del livello fisico (niente EMA). Soglie da fissare prima dell'addestramento: R² della velocità per mano ≥ 0,5, R² della posizione ≥ 0,9, IsoScore > 0,19.

## Livello semantico

1. **Togliere le 8 query.** Sostituire l'attuale predictor semantico (4 blocchi su più di 8.200 token, query mediate) con l'AT-Pool di worldSign.
2. **AT-Pool sulle ancore.** Per ogni step, media dei token dell'encoder (con stop-gradient) in 5 regioni: corpo, mano sinistra, mano destra, viso, frame intero. Come sesta regione, s del livello di posa. Il risultato sono 32 × 6 ancore da 384. Seguono 4 blocchi fattorizzati (fra regioni e nel tempo) con posizione temporale; i blocchi 1 e 3 fanno anche cross-attention sulla griglia piena.
3. **Lettura doppia.** 32 stati con un pooling sulle regioni senza query, più 1 vettore globale con una sola query. Poi Linear 384 → 256 e BatchNorm.
4. **Testo per token.** Un embedding per token: EmbeddingGemma prima del pooling, oppure BGE-M3 + ColBERT, da decidere. Testa per token; resta l'ẽ della frase per il termine globale.
5. **Energia duale.** `E_sem = ½·‖ŷ_g − LN(sg ẽ)‖²/d + ½·mean_l min_t ‖ŷ_t − LN(sg ẽ_l)‖²/d`, con SIGReg per step sugli stati. Il retrieval ordina per −E_sem; il braccio C usa InfoNCE sullo stesso score.
6. **Verifiche prima delle run lunghe:**
   - due prove brevi solo semantiche su OpenASL: prima solo il termine globale, poi anche quello per token;
   - test del rumore e varianza degli stati fra clip, contro il rischio di stati «universali» senza negativi;
   - in worldSign, R@1 per lunghezza sul checkpoint `predictor-v2`, per capire se il merito è del testo per token.
