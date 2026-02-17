
__generated_with = "0.19.11"

# %%
import marimo as mo

# %%
mo.md(
    r"""
    # PYACEMAKER Tutorial: Fe/Pt Deposition on MgO

    This interactive notebook demonstrates the **PYACEMAKER** automated MLIP construction system.

    **Goal**: Simulate the deposition of Iron (Fe) and Platinum (Pt) atoms onto a Magnesium Oxide (MgO) (001) substrate, observe the nucleation of clusters, and visualize the L10 ordering process.

    **How to Run:**
    Execute this notebook using Marimo:
    ```bash
    uv run marimo run tutorials/UAT_AND_TUTORIAL.py
    ```

    **Scenario:**
    1.  **Phase 1 (Active Learning):** Train a hybrid ACE potential for Fe-Pt-Mg-O.
    2.  **Phase 2 (MD Deposition):** Use the trained potential to simulate deposition.
    3.  **Phase 3 (Analysis):** Analyze long-term ordering (mocked aKMC results).
    """
)

# %%
mo.md(
    """
    ### Step 1: Environment Setup

    We import the necessary libraries.
    *   **Standard Library**: For path manipulation and system operations.
    *   **Scientific Stack**: **NumPy** for calculations, **Matplotlib** for plotting, and **ASE (Atomic Simulation Environment)** for atomistic manipulation.
    We also set up the random seed for reproducibility.
    """
)

# %%
import os
import sys
import shutil
import tempfile
import atexit
import importlib.util
from pathlib import Path
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# %%
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, Atom
from ase.visualize.plot import plot_atoms
from ase.build import surface, bulk
from ase.io import write

# Set random seed for reproducibility
np.random.seed(42)

# %%
mo.md(
    """
    Now we locate the `pyacemaker` source code. This logic allows the tutorial to run even if the package is installed in editable mode or located in a parent directory.
    """
)

# %%
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

# %%
mo.md(
    """
    Finally, we attempt to import the **PYACEMAKER** core modules.

    If this fails, you likely need to install the package dependencies.
    Required packages include:
    *   `ase`
    *   `numpy`
    *   `matplotlib`
    *   `marimo`
    *   `pyyaml`
    *   `pydantic`

    Install them using:
    ```bash
    uv sync
    # OR
    pip install -e .[dev]
    ```
    """
)

# %%
HAS_PYACEMAKER = False
pyacemaker = None
PYACEMAKERConfig = None
CONSTANTS = None
Orchestrator = None
Potential = None
StructureMetadata = None
PotentialHelper = None
metadata_to_atoms = None

spec = importlib.util.find_spec("pyacemaker")
if spec is None:
    mo.md(
        """
        ::: error
        **ERROR: PYACEMAKER package not found.**

        Please install dependencies:
        ```bash
        uv sync
        ```
        :::
        """
    )
else:
    try:
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
            **Import Error:** {e}

            The package was found but failed to import. Check dependencies.
            :::
            """
        )
    except Exception as e:
         mo.md(
            f"""
            ::: error
            **Unexpected Error:** {e}
            :::
            """
        )

# %%
mo.md(
    """
    ### Step 2: Mode Detection & Dependency Check

    We detect whether to run in **Mock Mode** (CI) or **Real Mode** (Production) based on the `CI` environment variable.

    We also verify the presence of critical external binaries.
    """
)

# %%
# Dependency Check
required_binaries = ["pw.x", "lmp", "pace_train"]
found_binaries = {}
missing_binaries = []

for binary in required_binaries:
    bin_path = shutil.which(binary)
    if bin_path:
        found_binaries[binary] = bin_path
    else:
        missing_binaries.append(binary)

# Detect Mode
# Default to CI/Mock mode if not explicitly set to false/0/no/off
raw_ci = os.environ.get("CI", "true").strip().lower()
valid_true = ["true", "1", "yes", "on"]
valid_false = ["false", "0", "no", "off"]

if raw_ci in valid_true:
    IS_CI = True
elif raw_ci in valid_false:
    IS_CI = False
else:
    IS_CI = True # Default safe

# Force Mock Mode if binaries are missing
if not IS_CI and missing_binaries:
    mo.md(
        f"""
        ::: warning
        **Missing Binaries:** {', '.join(missing_binaries)}

        Falling back to **Mock Mode** despite `CI={raw_ci}` because required tools are not in PATH.
        :::
        """
    )
    IS_CI = True

mode_name = "Mock Mode (CI)" if IS_CI else "Real Mode (Production)"

# Render Status Table
status_md = f"""
### System Status: **{mode_name}**

| Binary | Status | Path |
| :--- | :--- | :--- |
"""
for binary in required_binaries:
    if binary in found_binaries:
        status_md += f"| `{binary}` | ✅ Found | `{found_binaries[binary]}` |\n"
    else:
        status_md += f"| `{binary}` | ❌ Missing | - |\n"

mo.md(status_md)

# %%
mo.md(
    r"""
    ### Step 3: Configuration Setup

    We configure the **PYACEMAKER** system. This involves:
    1.  Creating a temporary workspace.
    2.  Setting up Pseudopotentials (using dummies in Mock mode).
    3.  Defining the `PYACEMAKERConfig` object.
    """
)

# %%
mo.md(
    r"""
    #### Understanding Active Learning & Extrapolation Grade ($\gamma$)

    The core of PYACEMAKER is its **Active Learning Loop**. Traditional potentials are trained on a static dataset, often failing when encountering unseen configurations. PYACEMAKER uses an iterative approach:

    1.  **Train**: Build an initial potential.
    2.  **Explore**: Run Molecular Dynamics (MD) simulations.
    3.  **Detect Uncertainty**: At every MD step, we calculate the **Extrapolation Grade ($\gamma$)**.
        *   $\gamma$ is calculated as the **distance of the current atomic environment from the training set in feature space** (using the ACE basis).
        *   If $\gamma < \text{threshold}$ (e.g., 0.5): The simulation continues (Safe, low uncertainty).
        *   If $\gamma > \text{threshold}$ (e.g., 0.5): The simulation **halts**. This means the atomic environment is more than 0.5 units away from the training data, indicating high uncertainty.
    4.  **Label**: The "uncertain" structure is sent to the Oracle (DFT) for accurate energy/force calculation.
    5.  **Retrain**: The new data is added, and the potential is retrained.

    This ensures the potential learns exactly what it needs to know, minimizing expensive DFT calculations.
    """
)

# %%
config = None
config_dict = None
pseudos = None
tutorial_dir = None
tutorial_tmp_dir = None

if HAS_PYACEMAKER:
    # Create temporary directory in CWD for security compliance
    tutorial_tmp_dir = tempfile.TemporaryDirectory(prefix="pyacemaker_tutorial_", dir=Path.cwd())
    tutorial_dir = Path(tutorial_tmp_dir.name)

    # Register cleanup on exit to ensure directory is removed even on crash
    def _cleanup_handler():
        try:
            if tutorial_tmp_dir:
                tutorial_tmp_dir.cleanup()
                print(f"Cleanup: Removed {tutorial_dir}")
        except Exception:
            pass
    atexit.register(_cleanup_handler)

    mo.md(f"Initializing Tutorial Workspace at: `{tutorial_dir}`")

    pseudos = {"Fe": "Fe.pbe.UPF", "Pt": "Pt.pbe.UPF", "Mg": "Mg.pbe.UPF", "O": "O.pbe.UPF"}

    if IS_CI:
        mo.md("::: danger\n**MOCK MODE: Creating DUMMY `.UPF` files.**\n:::")
        # Security: Ensure content is static and harmless
        safe_dummy_content = '<UPF version="2.0.1"><PP_INFO>MOCK_DATA</PP_INFO></UPF>'
        for element, filename in pseudos.items():
            pseudo_path = tutorial_dir / filename
            if not pseudo_path.exists():
                with open(pseudo_path, "w") as f:
                    f.write(safe_dummy_content)

    # Define configuration
    config_dict = {
        "version": "0.1.0",
        "project": {"name": "FePt_MgO", "root_dir": str(tutorial_dir)},
        "logging": {"level": "INFO"},
        "orchestrator": {"max_cycles": 2 if IS_CI else 10},
        "oracle": {"dft": {"pseudopotentials": {k: str(tutorial_dir / v) if IS_CI else v for k, v in pseudos.items()}}, "mock": IS_CI},
        "trainer": {"potential_type": "pace", "mock": IS_CI, "max_epochs": 1},
        "dynamics_engine": {"engine": "lammps", "mock": IS_CI, "gamma_threshold": 0.5, "timestep": 0.001, "n_steps": 100},
        "structure_generator": {"strategy": "random"},
        "validator": {"test_set_ratio": 0.1},
    }
    config = PYACEMAKERConfig(**config_dict)
    (tutorial_dir / "data").mkdir(exist_ok=True, parents=True)

# %%
mo.md(
    r"""
    ## Step 4: Phase 1 - Active Learning Loop

    The `Orchestrator` manages the loop: Generation -> Oracle -> Training -> Exploration -> Validation.

    1.  **Cold Start**: Generate initial random structures and label them.
    2.  **Training**: Fit an ACE potential.
    3.  **Exploration**: Run MD with `fix halt`.
    4.  **Refinement**: If MD halts, label the bad structure and retrain.
    """
)

# %%
orchestrator = None
if HAS_PYACEMAKER:
    try:
        orchestrator = Orchestrator(config)
        print("Orchestrator Initialized.")
    except Exception as e:
        mo.md(f"::: error\n**Init Error:** {e}\n:::")

# %%
mo.md(
    """
    ### Data Conversion: `metadata_to_atoms`

    The `metadata_to_atoms` function is a crucial utility that bridges the gap between PYACEMAKER's internal data model and the ASE (Atomic Simulation Environment) ecosystem.

    *   **Internal Model (`StructureMetadata`)**: PYACEMAKER uses a rich Pydantic model to store structures along with their full provenance (origin, calculation status, tags) and calculated features (energy, forces, stress, uncertainty).
    *   **External Tool (`ase.Atoms`)**: Most simulation engines (like Pacemaker, LAMMPS via ASE) operate on standard `ase.Atoms` objects.

    `metadata_to_atoms` extracts the atomic positions, cell, and numbers from `StructureMetadata` and packages them into an `ase.Atoms` object, attaching energy and forces as properties if they exist. This allows seamless data exchange between the orchestrator and the training/simulation modules.
    """
)

# %%
mo.md(
    """
    ### Visualization

    We plot the Root Mean Square Error (RMSE) of the energy predictions on the validation set for each cycle.
    A downward trend indicates the potential is improving.
    """
)

# %%
if HAS_PYACEMAKER and results:
    rmse_values = []
    for r in results:
        v = getattr(r.metrics, "energy_rmse", 0.0)
        if v == 0.0: v = r.metrics.model_dump().get("energy_rmse", 0.0)
        rmse_values.append(v)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(results)+1), rmse_values, 'b-o')
    plt.title("Training Convergence")
    plt.xlabel("Cycle")
    plt.ylabel("RMSE (eV/atom)")
    plt.grid(True)
    plt.show()

# %%
mo.md(
    """
    ## Step 7: Phase 2 - Dynamic Deposition (MD)

    Using the trained potential, we now simulate the physical process of depositing Fe/Pt atoms onto the MgO substrate.

    *   **Real Mode**: This would use LAMMPS with the `fix deposit` command to physically simulate atoms landing on the surface.
    *   **Mock Mode**: We simulate the deposition by randomly placing atoms above the surface to visualize the initial state.
    """
)

# %%
mo.md(
    """
    ### Deposition Simulation

    The `run_deposition` function performs the following tasks:

    1.  **Environment Check**: Determines if we are in Mock Mode or Real Mode.
    2.  **Substrate Setup**: Creates an MgO (001) slab using ASE `bulk` and `surface` tools.
    3.  **Real Mode Logic**: If a trained potential exists, it generates the necessary `in.lammps` input files to run a physical MD simulation using `PotentialHelper`.
    4.  **Mock/Visualization Logic**: Regardless of mode, it performs a Python-based stochastic placement of Fe and Pt atoms above the surface. This allows us to visualize the *expected* geometry of the deposition process immediately in the notebook, without waiting for a potentially long-running MD job.
    5.  **Output**: Saves the structure to `deposition_md/final.xyz` for analysis.
    """
)

# %%
output_path = None
deposited_structure = None

# Logic: Validate symbols against system configuration to ensure consistency.
valid_symbols = ["Fe", "Pt"]

if HAS_PYACEMAKER and orchestrator:
    # Dependency Usage: Acknowledge the 'results' to maintain topological order semantics
    print(f"Starting deposition after {len(results)} active learning cycles.")

    # Robust attribute check
    potential = getattr(orchestrator, 'current_potential', None)
    if potential is None:
         print("Warning: No potential available from orchestrator. Deposition simulation might fail in Real Mode.")

    md_work_dir = tutorial_dir / "deposition_md"
    md_work_dir.mkdir(exist_ok=True)

    # Setup Substrate
    substrate = surface(bulk("MgO", "rocksalt", a=4.21), (0, 0, 1), 2)
    substrate.center(vacuum=10.0, axis=2)
    deposited_structure = substrate.copy()

    # Real Mode Generation
    if not IS_CI and potential and potential.path.exists():
        helper = PotentialHelper()
        cmds = helper.get_lammps_commands(potential.path, "zbl", ["Mg", "O", "Fe", "Pt"])
        print("Generated LAMMPS commands.")

    # Simulation (Mock Logic for visual)
    # Using np.random for consistency
    # Dynamic atom count based on mode
    n_atoms = 5 if IS_CI else 50

    print(f"Simulating deposition of {n_atoms} atoms (Mode: {'CI' if IS_CI else 'Real'})...")

    for _ in range(n_atoms):
        x = np.random.uniform(0, substrate.cell[0,0])
        y = np.random.uniform(0, substrate.cell[1,1])
        z = substrate.positions[:,2].max() + np.random.uniform(2.0, 3.0)

        # Use proper Atom object
        symbol = np.random.choice(valid_symbols)
        atom = Atom(symbol=symbol, position=[x, y, z])
        deposited_structure.append(atom)

    plt.figure(figsize=(6, 6))
    plot_atoms(deposited_structure, rotation="-80x, 20y, 0z")
    plt.title(f"Deposition Result ({n_atoms} atoms)")
    plt.axis("off")
    plt.show()

    output_path = md_work_dir / "final.xyz"
    write(output_path, deposited_structure)

# %%
mo.md(
    """
    ## Step 8: Phase 3 - Analysis (L10 Ordering)

    After deposition, we are interested in whether the Fe and Pt atoms arrange themselves into the chemically ordered L10 phase. This process happens over long timescales (microseconds to seconds), which is too slow for standard MD.

    We use **Adaptive Kinetic Monte Carlo (aKMC)** (via EON) to accelerate time.

    The plot below shows the **Order Parameter** vs Time.
    *   **0**: Disordered (Random alloy)
    *   **1**: Perfectly Ordered (L10 layers)
    """
)

# %%
# Mock data for visualization
time_steps = np.linspace(0, 1e6, 50)
# Sigmoid function to simulate ordering transition
order_param = 1.0 / (1.0 + np.exp(-1e-5 * (time_steps - 3e5)))

plt.figure(figsize=(8, 4))
plt.plot(time_steps, order_param, 'r-')
plt.title("L10 Ordering (Mock)")
plt.xlabel("Time (us)")
plt.ylabel("Order Parameter")
plt.grid(True)
plt.show()

# %%
# Dependency on output_path and order_param ensures this runs LAST
# Constitution: Check if tutorial_tmp_dir is valid before cleanup to prevent crashes.
# Safety Check: Ensure the path is a safe temporary directory created by this tutorial
if tutorial_tmp_dir is not None and hasattr(tutorial_tmp_dir, 'cleanup'):
    try:
        # Resolve paths using pathlib for robust checking
        tmp_path = Path(tutorial_tmp_dir.name).resolve()

        # Allow:
        # 1. System temp directory
        # 2. Current working directory (since setup_config creates it there with dir=Path.cwd())

        allowed_bases = [
            Path(tempfile.gettempdir()).resolve(),
            Path.cwd().resolve()
        ]

        is_safe_location = any(tmp_path.is_relative_to(base) for base in allowed_bases)
        has_prefix = "pyacemaker_tutorial_" in tmp_path.name

        if has_prefix and is_safe_location:
            tutorial_tmp_dir.cleanup()
            print("Cleanup: Done.")
        else:
            logging.warning(f"Skipping cleanup: {tmp_path} is not a verified safe path.")
    except Exception as e:
        # Use logging to report cleanup errors without crashing
        logging.error(f"Cleanup failed: {e}")
        print(f"Cleanup warning: {e}")
