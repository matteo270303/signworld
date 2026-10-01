"""WorldSign: an energy-based world model for multilingual sign-language retrieval."""

import os
from importlib.metadata import version

# The ReCaS nodes expose hundreds of cores and BLAS spreads every call over all of them: on a
# 384-core node one 210x210 ridge solve took 2.4 s instead of 3 ms. The libraries read these
# variables when they load, which happens on the first numpy or torch import, so they are set
# here, before any of them; a job that knows better exports its own value.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "8")

__version__ = version("signworld")
