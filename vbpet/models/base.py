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
    ``lower``: per-parameter lower bound (None = 0 for all, as MATLAB vbPET);
    ``prior_sd_min``: absolute floor of the regional prior sd per parameter
    (needed for parameters whose value can be ~0, e.g. a delay).
    """

    name: str
    param_names: tuple
    p0: tuple
    predict_fn: Callable
    derived: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    label: str = ""
    lower: tuple = None
    prior_sd_min: dict = field(default_factory=dict)

    @property
    def lower_bounds(self):
        return tuple(0.0 for _ in self.param_names) if self.lower is None else self.lower

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
        lb = self.lower_bounds
        return Model(f"{self.name}", tuple(names[i] for i in free),
                     tuple(self.p0[i] for i in free),
                     lambda theta, inp, frames: fn(expand(theta), inp, frames), derived,
                     fixed={**self.fixed, **fixed}, label=f"{self.name}[{tag}]",
                     lower=tuple(lb[i] for i in free), prior_sd_min=self.prior_sd_min)

    def with_delay(self, lower=-1.0, prior_sd=0.1):
        """Add a blood arrival delay ``delay`` (min, input shifted: C(t - delay)).

        Needed for whole-body PET, where arrival differs along the body. The
        lower bound allows negative delays down to ``lower`` minutes.
        """
        fn = self.predict_fn
        P = self.n_params

        def predict(theta, inp, frames):
            d = theta[P]
            tp = inp.tp
            shifted = type(inp)(tp, jnp.interp(tp - d, tp, inp.Cp, left=0.0),
                                jnp.interp(tp - d, tp, inp.Cb, left=0.0))
            return fn(theta[:P], shifted, frames)

        derived = {k: (lambda t, f=f: f(t[:P])) for k, f in self.derived.items()}
        return Model(self.name, self.param_names + ("delay",), self.p0 + (0.0,), predict,
                     derived, fixed=self.fixed, label=(self.label or self.name) + "+delay",
                     lower=self.lower_bounds + (float(lower),),
                     prior_sd_min={**self.prior_sd_min, "delay": float(prior_sd)})

    def derived_with_var(self, theta, cov):
        """Return {name: (value, variance)} for one parameter vector."""
        out = {}
        for k, f in self.derived.items():
            v = f(theta)
            g = jax.grad(f)(theta)
            out[k] = (v, g @ cov @ g)
        return out
