"""The WorldSign model (§4): two prediction levels on one adapted V-JEPA 2.1 encoder.

This package holds the model itself. The pieces are small and independent, so each can be
built, tested and replaced alone: the configuration (``config``), frame sampling
(``sampling``), augmentation (``augmentation``), masking (``masking``), LoRA (``lora``),
the encoder (``backbone``), the multi-level fusion (``fusion``), the physical predictor and
its per-articulator read-out (``physical``, ``readout``), the semantic predictor
(``semantic``, ``rope``), the energies and regularisers (``losses``), and their assembly
(``video_branch``). The pose and text branches are added in later iterations.
"""
