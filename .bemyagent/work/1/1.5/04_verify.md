# Verify 1.5: PASS_WITH_CAVEATS

| Criterio | Evidenza | Esito |
|---|---|---|
| Test verdi | `pytest tests/data/test_loaders.py tests/models/worldsign/test_text_branch.py tests/experiment/train tests/cli` → 30 + 179 passed, 5 skipped (lenti) | PASS |
| Stessi numeri | `out/check_stats.py` su 945.688 clip: lingue uguali, `means` bit-equal (max diff 0.0), `keypoint_variance` 0.2913013994693756 uguale | PASS |
| `fit_statistics` < 2 min su nodo GPU | non misurato: job 63996342 cancellato da `scancel --me`. Sul login node pose 63 s (da ~500), caption 258 s ma ora su ogni rank prima della barrier → attesa alla barrier ≈ pose + fit ≈ 1 min | CAVEAT |
| Drift | nessuna collective nuova, stesso campione (`rng.choice` invariato), niente cache | PASS |

Caveat: la lettura a blocchi degli embedding non è risultata più veloce sul login node (limitato dalla banda: anche `np.load` intero 193 s). Il guadagno sulle caption viene dal calcolarle in parallelo su tutti i rank e una sola volta, non dalla lettura in sé. Da rimisurare su nodo GPU.
