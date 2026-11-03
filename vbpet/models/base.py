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
    log_params: tuple = ()  # parameters estimated as log θ (see log_space)

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
                     lower=tuple(lb[i] for i in free), prior_sd_min=self.prior_sd_min,
                     log_params=tuple(p for p in self.log_params if p not in fixed))

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

    def log_space(self):
        """Same model with log θ for the parameters bounded at 0 (rates, Vb).

        VB then never hits the bound (no clipping, VT stays finite) and the
        regional prior becomes log-normal. Other parameters (e.g. ``delay``) stay
        linear. Use :meth:`prior_to_internal` / :meth:`to_natural` to convert.
        """
        if self.log_params:
            return self
        lb = self.lower_bounds
        idx = tuple(i for i, b in enumerate(lb) if b == 0.0)
        if not idx:
            return self
        jidx = jnp.array(idx)
        fn = self.predict_fn

        def natural(t):
            return t.at[jidx].set(jnp.exp(t[jidx]))

        derived = {k: (lambda t, f=f: f(natural(t))) for k, f in self.derived.items()}
        p0 = tuple(float(jnp.log(max(v, 1e-3))) if i in idx else v for i, v in enumerate(self.p0))
        return Model(self.name, self.param_names, p0,
                     lambda t, inp, frames: fn(natural(t), inp, frames), derived,
                     fixed=self.fixed, label=(self.label or self.name) + "[log]",
                     lower=tuple(-jnp.inf if i in idx else b for i, b in enumerate(lb)),
                     prior_sd_min=self.prior_sd_min,
                     log_params=tuple(self.param_names[i] for i in idx))

    def _log_mask(self):
        import numpy as np
        return np.array([p in self.log_params for p in self.param_names])

    def prior_to_internal(self, mean, sd, floor_rel=1e-3):
        """Natural-unit prior (mean, sd) -> internal coordinates.

        Log parameters: log-normal with median = mean and the same coefficient of
        variation, sd_log = sqrt(log(1 + (sd/mean)^2)). Means at or below 0 (a
        regional fit at the bound) are floored to ``floor_rel`` x sd.
        """
        import numpy as np
        mean = np.array(mean, float)
        sd = np.array(sd, float)
        lm = self._log_mask()
        if not lm.any():
            return mean, sd
        mu = np.maximum(mean[..., lm], np.maximum(floor_rel * sd[..., lm], 1e-12))
        cv = sd[..., lm] / mu
        mean[..., lm] = np.log(mu)
        sd[..., lm] = np.sqrt(np.log1p(cv**2))
        return mean, sd

    def to_natural(self, m, cov):
        """Internal posterior (m, cov) -> natural-unit point estimate and sd.

        Log parameters: median exp(m) and the log-normal sd
        sqrt((exp(v) - 1) exp(2m + v)).
        """
        import numpy as np
        m = np.asarray(m, float)
        var = np.einsum("...ii->...i", np.asarray(cov, float))
        est, sd = m.copy(), np.sqrt(np.clip(var, 0, None))
        lm = self._log_mask()
        if lm.any():
            v = np.clip(var[..., lm], 0, 50)
            est[..., lm] = np.exp(m[..., lm])
            sd[..., lm] = np.sqrt(np.expm1(v) * np.exp(2 * m[..., lm] + v))
        return est, sd

    def derived_with_var(self, theta, cov):
        """Return {name: (value, variance)} for one parameter vector."""
        out = {}
        for k, f in self.derived.items():
            v = f(theta)
            g = jax.grad(f)(theta)
            out[k] = (v, g @ cov @ g)
        return out
