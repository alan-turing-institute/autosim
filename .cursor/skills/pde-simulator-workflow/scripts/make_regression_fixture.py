from __future__ import annotations

"""Cursor skill wrapper.

The IDE-generic implementation lives at `scripts/pde_sim/make_regression_fixture.py`.
"""

import runpy


if __name__ == "__main__":
    runpy.run_path("scripts/pde_sim/make_regression_fixture.py", run_name="__main__")
