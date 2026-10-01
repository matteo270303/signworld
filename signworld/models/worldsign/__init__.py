"""The WorldSign model (§4): two prediction levels on one adapted V-JEPA 2.1 encoder.

This package holds the networks. The pieces are small and independent, so each can be built,
tested and replaced alone: masking (``masking``), LoRA (``lora``), the encoder (``backbone``),
the multi-level fusion (``fusion``), the physical predictor and its per-articulator read-out
(``physical``, ``readout``), the semantic predictor (``semantic``, ``rope``), the three
branches (``video_branch``, ``pose_branch``, ``text_branch``), the plausibility tests
(``plausibility``) and their assembly (``model``). The configuration lives in
``signworld.experiment.train.config``, the energies and regularisers in ``signworld.loss``,
frame sampling and augmentation in ``signworld.data``.
"""
