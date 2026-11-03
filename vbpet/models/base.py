"""Kinetic model interface."""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Frames(NamedTuple):
    start: jnp.ndarray
    end: jnp.ndarray
    mid: jnp.ndarray

    @classmethod
    def from_start_duration(cls, start, duration):
        start = jnp.asarray(start, dtype=float)
        end = start + jnp.asarray(duration, dtype=float)
        return cls(start, end, 0.5 * (start + end))

    def as_dict(self):
        return {"start": self.start, "end": self.end, "mid": self.mid}


class PlasmaInput(NamedTuple):
    """Input sampled on a uniform grid ``tp`` starting at 0 (minutes)."""

    tp: jnp.ndarray
    Cp: jnp.ndarray  # metabolite-corrected arterial plasma
    Cb: jnp.ndarray  # whole blood (for the Vb term)


@dataclass(frozen=True)
class Model:
    """A kinetic model: ``predict(theta, inp, frames) -> (n_frames,)``.

    ``derived`` maps names to scalar functions of theta (e.g. VT, Ki); their
    variance is obtained by the delta method with autodiff gradients.
    """

    name: str
    param_names: tuple
    p0: tuple
    predict_fn: Callable
    derived: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    label: str = ""

    @property
    def n_params(self):
        return len(self.param_names)

    def predict(self, theta, inp, frames):
        return self.predict_fn(theta, inp, frames)

    def fix(self, fixed):
        """Model with some parameters held at fixed values, e.g. ``{"Vb": 0.05}``."""
        if not fixed:
            return self
        names = self.param_names
        free = tuple(i for i, n in enumerate(names) if n not in fixed)
        full0 = jnp.array([fixed.get(n, 0.0) for n in names], dtype=float)
        free_idx = jnp.array(free)

        def expand(theta):
            return full0.at[free_idx].set(theta)

        fn = self.predict_fn
        derived = {k: (lambda t, f=f: f(expand(t))) for k, f in self.derived.items()}
        tag = ",".join(f"{k}={v:g}" for k, v in fixed.items())
        return Model(f"{self.name}", tuple(names[i] for i in free),
                     tuple(self.p0[i] for i in free),
                     lambda theta, inp, frames: fn(expand(theta), inp, frames), derived,
                     fixed={**self.fixed, **fixed}, label=f"{self.name}[{tag}]")

    def derived_with_var(self, theta, cov):
        """Return {name: (value, variance)} for one parameter vector."""
        out = {}
        for k, f in self.derived.items():
            v = f(theta)
            g = jax.grad(f)(theta)
            out[k] = (v, g @ cov @ g)
        return out
