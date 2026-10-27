"""Bounded variants of two held-out test plants (2026-10-26; design: experiments/cartpole_bounded/DESIGN.md).

    cartpole_b      cart-pole with the pole-balancing loop PLUS a weak, slow cart-centering loop
                    (a balancing controller on a finite track), so the cart position stays bounded
    two_link_arm_b  two-link arm with an elbow torsion spring (rest pose: straight elbow), so the
                    second link swings but cannot spin full revolutions

Same contract as systems.py. Own module (not systems.py / systems_extra.py) so the source of the
cache-hashed modules (cache.DATA_MODULES) is untouched and every existing cache key stays valid.
hyperkkl/__init__.py imports this module on the first registry lookup of one of its names.
The existing CartPole / TwoLinkArm classes are not modified; these subclasses only add terms.
NOTE: this module is not part of DATA_MODULES; do not edit it after runs exist (cache keys would
not change). Its sha1 is recorded in experiments/cartpole_bounded/DESIGN.md.
"""

from __future__ import annotations

import torch

from .systems import CartPole, TwoLinkArm, register


@register("cartpole_b")
class CartPoleBounded(CartPole):
    """CartPole (same physics, same partial model) with cart feedback added to the stabilising loop.

    F = f_scale (u - K x), K = (k_x, k_v, k_th, k_om): the pole gains are the original K_BAL;
    k_x = -0.8, k_v = -0.6 (input units per m, per m/s) add a slow centering loop. Linearised closed
    loop (with the saturating actuator's back-EMF): pole pair -7.85 +- 0.75j (tau 0.13 s), centering
    pair -0.345 +- 1.07j (tau 2.9 s, zeta 0.31); static gain x_ss / u = -1.25 m per unit command, so
    the protocol box |u| <= 1 keeps the cart within a gym-like +-2.4 m track. The negative signs are
    the non-minimum-phase cart-pole structure (to move the cart one way, the base must first lean
    the pole that way). f_known (the modeller's model) uses the same K: the controller is known;
    the UNMODELLED large-angle terms are exactly those of CartPole.
    """
    K_BAL = (-0.8, -0.6, -9.889688, -2.181384)
    UNMODELLED = CartPole.UNMODELLED


@register("two_link_arm_b")
class TwoLinkArmBounded(TwoLinkArm):
    """TwoLinkArm (same physics, same partial model) plus an elbow torsion spring tau_2 = -k_e q2.

    k_e = m2 g lc2 (= 4.905 N m / rad at nominal parameters): the spring is as stiff as link 2's
    own gravity stiffness, so the static gravity sag of the elbow is at most 1 rad and the second
    link swings but cannot wind up. Elbow natural frequency sqrt(k_e / (m2 lc2^2 + I2)) = 3.8 rad/s,
    close to the shoulder pendulum (~3 rad/s): both links stay excited. A generalised force on joint 2
    enters through M(q)^-1 like the input torque on joint 1. k_e is tied to (m2, g, lc2), so it
    follows the 30 % mismatch of those parameters (a physical spring sized for the link). The spring
    is part of the known model; UNMODELLED stays the Coriolis / centrifugal terms of TwoLinkArm.
    """
    UNMODELLED = TwoLinkArm.UNMODELLED
    KE_FACTOR = 1.0                                       # k_e = KE_FACTOR * m2 g lc2

    def _spring_acc(self, x):
        q2 = x[:, 1]
        a = self.m1 * self.lc1 ** 2 + self.I1 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2) + self.I2
        b = self.m2 * self.l1 * self.lc2
        d = self.m2 * self.lc2 ** 2 + self.I2
        c2 = torch.cos(q2)
        m11, m12, m22 = a + 2 * b * c2, d + b * c2, d + 0 * c2
        det = m11 * m22 - m12 ** 2
        t2 = -self.KE_FACTOR * (self.m2 * self.g * self.lc2) * q2          # elbow spring torque (tau_1 = 0)
        z = 0 * q2
        return torch.stack([z, z, -m12 * t2 / det, m11 * t2 / det], 1)

    def f(self, x, u):
        return super().f(x, u) + self._spring_acc(x)

    def f_known(self, x, u):
        return super().f_known(x, u) + self._spring_acc(x)
