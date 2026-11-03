"""vbPET: Variational Bayes parametric imaging of dynamic PET."""

import jax

# Kinetic models mix rate constants spanning several orders of magnitude; keep
# float64 available (the VB core still runs in float32 when asked to).
jax.config.update("jax_enable_x64", True)
# no TF32 for float32 matmuls on Ampere+ GPUs: the VB stopping rule compares
# free energies that TF32 rounding makes too noisy (68 vs 7 iterations on an A40)
jax.config.update("jax_default_matmul_precision", "highest")

__version__ = "0.1.0.dev0"
