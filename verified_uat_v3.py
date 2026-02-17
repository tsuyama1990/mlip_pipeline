
__generated_with = "0.19.11"

# %%
import marimo as mo

# %%
mo.md(
    r"""
    # PYACEMAKER Tutorial: Fe/Pt Deposition on MgO

    This interactive notebook demonstrates the **PYACEMAKER** automated MLIP construction system.

    **How to Run:**
    Execute this notebook using Marimo:
    ```bash
    uv run marimo run tutorials/UAT_AND_TUTORIAL.py
    ```

    **Scenario:** We will simulate the deposition of Iron (Fe) and Platinum (Pt) atoms onto a Magnesium Oxide (MgO) (001) substrate.

    **Workflow:**
    1.  **Phase 1 (Active Learning):** Train a hybrid ACE potential for Fe-Pt-Mg-O.
    2.  **Phase 2 (MD Deposition):** Use the trained potential to simulate deposition.
    3.  **Phase 3 (Analysis):** Analyze long-term ordering (mocked aKMC results).

    **Dual-Mode Strategy:**
    *   **Mock Mode (CI):** Fast execution using simulated data (default).
    *   **Real Mode (Production):** Full execution using Quantum Espresso and LAMMPS.
    """
)

# %%
mo.md(
    """
    ### Step 1: Environment Setup

    In this step, we configure the Python environment. We ensure the `pyacemaker` source code is accessible (checking for `src/pyacemaker/__init__.py`) and import necessary libraries.

    We also set environment variables to configure the system for this tutorial.
    """
)

# %%
mo.md(
    """
    ### Dependency Verification

    Before proceeding, we verify that the core `pyacemaker` library was successfully imported.
    If not, we halt the tutorial with clear instructions.
    """
)

# %%
mo.md(
    """
    ### Step 2: Mode Detection

    We detect whether to run in **Mock Mode** (CI) or **Real Mode** (Production) based on the `CI` environment variable.

    *   **Mock Mode**: Uses simulated data and skips external binary calls (QE, LAMMPS). Suitable for quick verification.
    *   **Real Mode**: Attempts to run full physics simulations. Suitable for actual research.
    """
)

# %%
mo.md(
    r"""
    ### Step 3: Configuration Setup

    The following cell sets up the **PYACEMAKER** configuration.
    It defines parameters for the Orchestrator, DFT Oracle, Trainer, and Dynamics Engine.

    **Detailed Parameter Explanations:**

    *   **`gamma_threshold` (Extrapolation Grade Limit)**:
        This is a critical hyperparameter for the Active Learning loop.

        **Analogy: The "Safe Zone" vs. "Uncharted Territory"**
        Think of the ML potential as a hiker with a map (the training data).
        *   **$\gamma$ (Gamma)** represents how far the current location is from the known paths on the map.
        *   **Low $\gamma$**: The hiker is on a known trail (Safe Zone). Predictions are reliable.
        *   **High $\gamma$**: The hiker is wandering into the wilderness (Uncharted Territory). Predictions are likely wrong.
        *   **Action**: When $\gamma > \gamma_{threshold}$, the hiker stops and asks for directions (calls the DFT Oracle). The new path is then added to the map (Training Set).

        *   **Values**:
            *   Mock Mode: `0.5` (Lower to trigger halts frequently for demonstration).
            *   Real Mode: `2.0` (Standard production value).

    *   `n_active_set_select`: The number of structures to select from the candidate pool using D-optimality (MaxVol).
    """
)

# %%
mo.md(
    r"""
    ## Step 4: Phase 1 - Active Learning Loop

    This phase demonstrates the core of **PYACEMAKER**. The `Orchestrator` manages a cyclical process to iteratively improve the Machine Learning Interatomic Potential (MLIP).

    **Active Set Optimization (MaxVol):**
    To minimize expensive DFT calculations, we don't label every structure.
    1.  We generate a pool of candidate structures.
    2.  We compute the **D-Optimality** criterion for each.
    3.  Using the **MaxVol** algorithm, we select the subset that maximizes the information gain (determinant of the Fisher information matrix).

    **The Loop Steps:**
    1.  **Generation:** Create new candidate atomic structures.
    2.  **Oracle (DFT):** Calculate "ground truth" energy/forces.
    3.  **Training:** Train the ACE potential on the Active Set.
    4.  **Exploration:** Run MD. If $\gamma > \text{threshold}$, halt and learn.
    5.  **Validation:** Test against hold-out data.
    """
)

# %%
mo.md("Initializing the `Orchestrator` with the configuration defined above.")

# %%
mo.md(
    """
    ### Step 5: Running the Active Learning Loop

    The code below demonstrates a "Cold Start" followed by the main active learning cycles.

    *   **Cold Start**: Initially, we have no data. We generate random structures and label them to bootstrap the potential.
    *   **Main Loop**: The Orchestrator iteratively improves the potential by finding high-uncertainty regions.
    """
)

# %%
mo.md(
    """
    ### Step 6: Visualization

    We now visualize the training convergence by plotting the Energy RMSE (Root Mean Square Error) across the active learning cycles.
    Decreasing RMSE indicates the potential is learning effectively.
    """
)

# %%
mo.md(
    """
    ## Step 7: Phase 2 - Dynamic Deposition (MD)

    Using the trained potential, we now simulate the physical process of depositing Fe/Pt atoms onto the MgO substrate.
    """
)

# %%
mo.md(
    """
    ## Step 8: Phase 3 - Long-Term Ordering (aKMC)

    **The Problem:** Standard MD is limited to nanoseconds. The ordering of Fe-Pt into the L10 phase (which gives it high magnetic anisotropy) happens over milliseconds or hours.

    **The Solution:** Adaptive Kinetic Monte Carlo (aKMC).
    *   aKMC searches for saddle points on the potential energy surface to find transition states.
    *   It allows the system to "hop" between stable states, extending the timescale to real-world relevance.

    **Order Parameter ($S$):**
    The plot below shows the simulated rise in the **Long-Range Order (LRO) Parameter**.
    *   **$S=0$**: Disordered (Random alloy). Atoms are randomly distributed.
    *   **$S=1$**: Perfectly Ordered. Fe and Pt atoms form alternating layers (L10 structure).
    """
)

# %%
mo.md(
    """
    ## Conclusion

    In this tutorial, we demonstrated the end-to-end workflow of **PYACEMAKER**:
    1.  **Automation**: The system autonomously improved the potential via Active Learning.
    2.  **Integration**: We saw how the Orchestrator bridges DFT (Oracle), ML (Trainer), and MD (Dynamics).
    3.  **Application**: We applied the potential to a realistic surface deposition scenario.
    """
)

# %%
mo.md(
    """
    ### Cleanup

    The following cell handles the cleanup of temporary directories created during this tutorial session.
    It ensures that no large data files or artifacts are left consuming disk space.
    """
)

# %%
# Explicit cleanup hook
if tutorial_tmp_dir and hasattr(tutorial_tmp_dir, 'cleanup'):
    try:
        tutorial_tmp_dir.cleanup()
        print("Cleanup: Temporary directory removed.")
    except Exception as e:
        print(f"Cleanup warning: {e}")

# %%
mo.md("### Utility: Common Imports\nLoading standard libraries for utility functions.")

# %%
import os
import sys
import importlib.util
from pathlib import Path

# %%
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.build import surface, bulk
from ase.io import write

# Locate src directory
cwd = Path.cwd()
possible_src_paths = [
    cwd / "src",
    cwd.parent / "src",
]

src_path = None
for p in possible_src_paths:
    if (p / "pyacemaker" / "__init__.py").exists():
        src_path = p
        break

if src_path:
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
        print(f"Added {src_path} to sys.path")
else:
    mo.md(
        f"""
        ::: warning
        **Warning:** 'src/pyacemaker' not found in {possible_src_paths}.
        Relying on installed package. If not installed, subsequent cells will fail.
        :::
        """
    )

# NOTE: We do NOT enable PYACEMAKER_SKIP_FILE_CHECKS.
# We will ensure all temporary files are created within the project root to satisfy strict security.

# Default to CI mode (Mock) if not specified
if "CI" not in os.environ:
    os.environ["CI"] = "true"

# Pyacemaker imports with error handling
HAS_PYACEMAKER = False
pyacemaker = None
PYACEMAKERConfig = None
CONSTANTS = None
Orchestrator = None
Potential = None
StructureMetadata = None
PotentialHelper = None
metadata_to_atoms = None

# Check for pyacemaker package existence using importlib
spec = importlib.util.find_spec("pyacemaker")
if spec is None and not src_path:
    mo.md(
        f"""
        ::: error
        **ERROR: PYACEMAKER package not found.**
        Please install it using `uv sync` or `pip install -e .`.
        :::
        """
    )
    HAS_PYACEMAKER = False
else:
    try:
        # Verify src path is active if we are relying on it
        if src_path and str(src_path) not in sys.path:
             raise ImportError(f"Source directory {src_path} found but not in sys.path")

        import pyacemaker
        from pyacemaker.core.config import PYACEMAKERConfig, CONSTANTS
        from pyacemaker.orchestrator import Orchestrator
        from pyacemaker.domain_models.models import Potential, StructureMetadata
        from pyacemaker.modules.dynamics_engine import PotentialHelper
        from pyacemaker.core.utils import metadata_to_atoms
        HAS_PYACEMAKER = True
        print(f"Successfully imported pyacemaker from {pyacemaker.__file__}")

    except ImportError as e:
        mo.md(
            f"""
            ::: error
            **ERROR: Import failed.**
            Even though the package was detected, the import raised an error.
            Details: {e}
            :::
            """
        )
        HAS_PYACEMAKER = False
        print(f"Import Error details: {e}")
    except Exception as e:
        mo.md(f"::: error\n**Unexpected Error during import:** {e}\n:::")
        HAS_PYACEMAKER = False


# %%
if not HAS_PYACEMAKER:
    mo.md(
        """
        # 🚨 CRITICAL ERROR: PYACEMAKER Not Found

        This tutorial **cannot proceed** without the `pyacemaker` package.

        **Action Required:**
        Please install the package before continuing.

        ```bash
        uv sync
        # OR
        pip install -e .
        ```
        """
    )

# %%
# Detect Mode
# Input Sanitization: Strictly parse boolean string from env var
raw_ci = os.environ.get("CI", "false").strip().lower()

# Whitelist valid boolean strings
valid_true = ["true", "1", "yes", "on"]
valid_false = ["false", "0", "no", "off"]

if raw_ci in valid_true:
    IS_CI = True
elif raw_ci in valid_false:
    IS_CI = False
else:
    print(f"Warning: Invalid CI environment variable '{raw_ci}'. Defaulting to Mock Mode (CI=True).")
    IS_CI = True

mode_name = "Mock Mode (CI)" if IS_CI else "Real Mode (Production)"

mo.md(f"### Current Mode: **{mode_name}**")

# %%
orchestrator = None
if HAS_PYACEMAKER:
    try:
        orchestrator = Orchestrator(config)
        print("Orchestrator Initialized.")
    except Exception as e:
        print(f"Error initializing Orchestrator: {e}")

# %%
print("Phase 3: Analysis of Long-Term Ordering (aKMC)")

# Mock Data: Order Parameter vs Time
time_steps = np.linspace(0, 1e6, 50)

# Sigmoid function to simulate ordering transition
order_param = 1.0 / (1.0 + np.exp(-1e-5 * (time_steps - 3e5)))

plt.figure(figsize=(8, 4))
plt.plot(time_steps, order_param, 'r-', linewidth=2)
plt.axhline(1.0, color='k', linestyle='--', alpha=0.3)
plt.title("L10 Ordering Kinetic Monte Carlo (Mock)")
plt.xlabel("Time (microseconds)")
plt.ylabel("Order Parameter (0-1)")
plt.grid(True, alpha=0.3)
plt.show()
