"""Optimize staged cable forces only (ignore deck shape), allowing strand changes.

Thin wrapper over :mod:`scripts.optimize_cables` that flips the defaults to the
"forget the deck line, just make the cable stresses feasible and uniform"
experiment: the shape objective is disabled and the integer strand-count layer
is enabled, so the optimizer drives every cable into ``[stress_lower,
stress_upper]`` by jointly adjusting strands and pretension.

Example::

    py -3.12 -m scripts.optimize_cable_forces --n 3
    py -3.12 -m scripts.optimize_cable_forces --n 3 --tension-bound-stress 2000 --quiet
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import optimize_cables  # noqa: E402


def main() -> None:
    p = optimize_cables.build_parser()
    # Force-only experiment defaults: drop the deck-shape objective and let the
    # integer strand-count layer run so cable stresses can reach feasibility.
    p.set_defaults(
        weight_shape=0.0,
        outer_iterations=6,
        continuous_maxiter=60,
        out="results/cable_force_opt",
    )
    args = p.parse_args()
    optimize_cables.run(args)


if __name__ == "__main__":
    main()
