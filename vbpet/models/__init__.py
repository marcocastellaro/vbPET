"""Kinetic models."""

from . import plasma, reference

INPUT_KIND = {**{m: "blood" for m in plasma.AVAILABLE},
              **{a: "blood" for a in plasma.ALIASES},
              **{m: "reference" for m in reference.AVAILABLE}}


def get_model(name, conv="linear", sampling="mean", **kw):
    """Model by name; ``kw`` goes to the reference models (e.g. ``k2p``)."""
    if name in reference.AVAILABLE:
        return reference.get_model(name, conv=conv, sampling=sampling, **kw)
    return plasma.get_model(name, conv=conv, sampling=sampling)
