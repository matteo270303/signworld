"""The WorldSign model (§4): a hierarchy of three levels trained level by level.

Level 0 is the pose encoder, trained from scratch (``docs/worldsign-posa.md``); levels 1 and 2,
the physical and semantic predictors, read one V-JEPA 2.1 encoder that only the physical level
adapts (``docs/worldsign-gerarchia.md``). This package holds the networks. The pieces are
small and independent, so each can be built, tested and replaced alone: masking
(``masking``), LoRA (``lora``), the encoder (``backbone``), the multi-level fusion
(``fusion``), the physical predictor and its per-step read-out (``physical``, ``readout``), the
semantic predictor (``semantic``, ``rope``), the pose encoder and its input (``pose_encoder``,
``pose_features``), the three branches (``video_branch``, ``pose_branch``, ``text_branch``),
the plausibility tests (``plausibility``) and their assembly (``model``). The configuration lives in
``signworld.experiment.train.config``, the energies and regularisers in ``signworld.loss``,
frame sampling and augmentation in ``signworld.data``.
"""
