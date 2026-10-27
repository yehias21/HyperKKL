"""Mechanical decision-rule engine. Pure functions of paired comparisons; no I/O.

A comparison is the dict returned by hyperkkl.metrics.paired_relative
    {"rel", "ci": [lo, hi], "n_seeds", "ci_method", "better", "worse"}
(relative change of a against b, negative = a better; better/worse = 95% CI excludes 0).

Vote counting with pending data (the wave is still running)
    Each expected system contributes True (criterion met), False (not met) or is missing.
    With k met, f failed, m missing and a threshold `need`:
        PASS          k >= need                      (cannot be undone by missing systems,
                                                      unless a veto may still fire: see vote())
        FAIL          k + m < need                   (cannot be rescued)
        INCONCLUSIVE  otherwise                       (the missing systems decide)
        PENDING       nothing to count yet
Equivalence (TOST) for control nodes
    Two one-sided tests at level 0.05 = the 90% CI lies inside (-margin, +margin).
    Non-inferiority ("within margin or better") = the 90% CI upper bound < +margin.
    Underpowered: fewer than 3 seeds, or the 90% CI half-width >= margin (the test could
    not conclude equivalence even at a true effect of zero).
Climatology
    A baseline FAILED in a block if its mean NRMSE exceeds the climatology NRMSE (constant
    mean-state predictor) of that block, or is not finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PASS, FAIL, INCONCLUSIVE, PENDING, REPORTED = "PASS", "FAIL", "INCONCLUSIVE", "PENDING", "REPORTED"
MARGIN = 0.10


@dataclass
class Vote:
    k: int
    n: int                    # systems with data
    n_expected: int
    need: int
    met: list
    not_met: list
    missing: list
    vetoed: list = field(default_factory=list)
    label: str = PENDING

    @property
    def frac(self) -> str:
        return f"{self.k}/{self.n}"

    def text(self) -> str:
        s = f"{self.k}/{self.n} systems (need {self.need}/{self.n_expected})"
        if self.met:
            s += f"; met: {', '.join(self.met)}"
        if self.vetoed:
            s += f"; veto: {', '.join(self.vetoed)}"
        if self.missing:
            s += f"; pending: {', '.join(self.missing)}"
        return s


def vote(per_system: dict, expected: list, need: int, veto: dict = None) -> Vote:
    """per_system {sys: bool}; expected: the systems the rule is about; veto {sys: bool}
    (True = the system triggers the rule's 'and ... on none' clause)."""
    met = [s for s in expected if per_system.get(s) is True]
    nmet = [s for s in expected if per_system.get(s) is False]
    miss = [s for s in expected if s not in per_system or per_system[s] is None]
    vetoed = [s for s in expected if (veto or {}).get(s)]
    k, m = len(met), len(miss)
    v = Vote(k, k + len(nmet), len(expected), need, met, nmet, miss, vetoed)
    if k + len(nmet) == 0:
        v.label = PENDING
    elif vetoed or k + m < need:
        v.label = FAIL
    elif k >= need and (not miss or veto is None):
        v.label = PASS
    else:
        v.label = INCONCLUSIVE
    return v


def combine(labels: list) -> str:
    """AND of several sub-rules (e.g. 'on shape_ood and on sequence_ood')."""
    if not labels:
        return PENDING
    if FAIL in labels:
        return FAIL
    if all(l == PASS for l in labels):
        return PASS
    if all(l == PENDING for l in labels):
        return PENDING
    return INCONCLUSIVE


# -- comparisons -------------------------------------------------------------

def better(p) -> bool:
    return bool(p) and bool(p.get("better"))


def worse(p) -> bool:
    return bool(p) and bool(p.get("worse"))


def within_ci(p) -> bool:
    """'within CI' as written in the rules: the 95% CI of the paired change includes 0."""
    return bool(p) and not p.get("better") and not p.get("worse")


def no_loss(p) -> bool:
    """Not significantly worse (within CI, or significantly better)."""
    return bool(p) and not p.get("worse")


def tost(p90, margin=MARGIN) -> dict:
    """p90: paired_relative with alpha=0.10. -> {equivalent, noninferior, underpowered, half}."""
    if not p90 or p90.get("n_seeds", 0) < 2 or not np.all(np.isfinite(p90["ci"])):
        return {"equivalent": None, "noninferior": None, "underpowered": True, "half": float("nan")}
    lo, hi = p90["ci"]
    half = (hi - lo) / 2
    return {"equivalent": bool(lo > -margin and hi < margin), "noninferior": bool(hi < margin),
            "underpowered": bool(p90["n_seeds"] < 3 or half >= margin), "half": float(half)}


# -- selection ---------------------------------------------------------------

def argmin(means: dict, candidates) -> str:
    c = [(means[m], m) for m in candidates if m in means and np.isfinite(means[m])]
    return min(c)[1] if c else None


def failed(mean_nrmse: float, clim: float) -> bool:
    """FAILED baseline: worse than the constant mean-state predictor (or non-finite)."""
    if mean_nrmse is None or not np.isfinite(mean_nrmse):
        return True
    return bool(np.isfinite(clim) and mean_nrmse > clim)


def best_working(means: dict, filters, clim: float) -> tuple:
    """(best working filter or None, [failed filters], [filters present])."""
    present = [f for f in filters if f in means]
    bad = [f for f in present if failed(means[f], clim)]
    good = [f for f in present if f not in bad]
    return argmin(means, good), bad, present


def knee(Ls, rel_to_ref: dict, tol=0.05):
    """Smallest window L such that every L' >= L (excluding the reference) has a relative
    change vs the reference of at most +tol. rel_to_ref {L: rel}."""
    Ls = sorted(Ls)
    for i, L in enumerate(Ls):
        if all(rel_to_ref.get(Lp, 0.0) <= tol for Lp in Ls[i:]):
            return L
    return None
