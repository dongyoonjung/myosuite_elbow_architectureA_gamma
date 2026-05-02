"""Set up sys.path so scripts can import both A and A+γ packages.

Inserts:
    1. Repo root (this file's parent).parent — for ``myosuite_elbow_arch_a_gamma``.
    2. Sibling A repo or env-var override — for ``myosuite_elbow_arch_a``.

Use:
    from _paths_agamma import setup
    setup()

The module is named ``_paths_agamma`` (not the bare ``_path``) to avoid
shadowing collisions during pytest discovery if a sibling repo also exposes
a ``scripts/_path.py``.
"""

from __future__ import annotations

import os
import sys


def setup() -> str | None:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    explicit = os.environ.get("MYOSUITE_ELBOW_A_PATH")
    if explicit:
        if os.path.isdir(os.path.join(explicit, "myosuite_elbow_arch_a")):
            if explicit not in sys.path:
                sys.path.insert(0, explicit)
            return explicit
        raise RuntimeError(
            f"MYOSUITE_ELBOW_A_PATH={explicit!r} does not contain a "
            f"'myosuite_elbow_arch_a' package directory."
        )

    candidates = [
        os.path.normpath(os.path.join(repo_root, "..", "myosuite_elbow_architectureA")),
        os.path.normpath(os.path.join(repo_root, "myosuite_elbow_architectureA")),
        os.path.normpath(os.path.join(repo_root, "..", "..", "myosuite_elbow_architectureA")),
        # VESSL container default layout: both repos under /root/.
        "/root/myosuite_elbow_architectureA",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "myosuite_elbow_arch_a")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c

    raise RuntimeError(
        "Could not locate the sibling 'myosuite_elbow_architectureA' repo. "
        "Tried:\n  - $MYOSUITE_ELBOW_A_PATH (unset)\n  - "
        + "\n  - ".join(candidates)
        + "\nClone the A repo next to this one, or set MYOSUITE_ELBOW_A_PATH."
    )
