"""S-JEPA as its paper describes it (Abdelfattah and Alahi, ECCV 2024) [Lett. 99] (posa §7).

A faithful reproduction on the paper's data, 3D skeleton sequences (NTU: 25 joints), kept apart
from WorldSign: the base of a later adaptation, not a part of the model. The official code is
not public; what S-JEPA inherits from MAMP (the segment embedding, the motion-aware masking and
its temperature, the predictor's positions, the weight-decay groups) follows MAMP's official
code [Lett. 100]. Values the paper does not give are marked ``[Aperto]``.

* ``config``: the paper's hyperparameters;
* ``views``: the views (rotation about the skeleton's vertical axis, translation, flip) and the
  trim-and-resize of the input;
* ``masking``: MAMP's motion-aware masking;
* ``model``: the view encoder, its EMA target encoder, the predictor and the loss;
* ``pretrain``: the learning-rate and momentum schedules and the pre-training loop.
"""
