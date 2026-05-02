"""Locate the sibling ``myosuite_elbow_architectureA`` repo and add it to sys.path.

A_gamma inherits from ``myosuite_elbow_arch_a.MyoElbowReflexEnvA``. That package
lives in a separate repo, not on the Python install path. Search order:

1. ``MYOSUITE_ELBOW_A_PATH`` environment variable (explicit override).
2. Sibling directory ``../myosuite_elbow_architectureA`` (WSL local layout).
3. Sibling directory ``../../myosuite_elbow_architectureA``
   (when this file is one level deeper, e.g. inside the package).

If none resolves, import will fail at the next ``from myosuite_elbow_arch_a ...``
import with a clear error.
"""

from __future__ import annotations

import os
import sys


def ensure_arch_a_on_path() -> str | None:
    if "myosuite_elbow_arch_a" in sys.modules:
        return None

    explicit = os.environ.get("MYOSUITE_ELBOW_A_PATH")
    if explicit and os.path.isdir(os.path.join(explicit, "myosuite_elbow_arch_a")):
        if explicit not in sys.path:
            sys.path.insert(0, explicit)
        return explicit

    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, "..", "..", "myosuite_elbow_architectureA")),
        os.path.normpath(os.path.join(here, "..", "myosuite_elbow_architectureA")),
        os.path.normpath(os.path.join(here, "..", "..", "..", "myosuite_elbow_architectureA")),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "myosuite_elbow_arch_a")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None
