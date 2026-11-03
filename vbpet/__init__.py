"""vbPET: Variational Bayes parametric imaging of dynamic PET."""

import jax

# Kinetic models mix rate constants spanning several orders of magnitude; keep
# float64 available (the VB core still runs in float32 when asked to).
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0.dev0"
