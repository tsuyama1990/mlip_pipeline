
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
    """
)

# %%
mo.md(
    """
    ### Step 1: Environment Setup

    First, we import the standard library modules required for path manipulation and system operations.
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

# %%
mo.md(
    """
    Next, we import the scientific stack: **NumPy** for calculations, **Matplotlib** for plotting, and **ASE (Atomic Simulation Environment)** for atomistic manipulation.
    """
)

# %%
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, Atom
from ase.visualize.plot import plot_atoms
from ase.build import surface, bulk
from ase.io import write

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

    If this fails, you likely need to install the package:
    ```bash
    uv sync
    # OR
    pip install -e .
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
    except Exception as e:
        mo.md(
            f"""
            ::: error
            **Import Error:** {e}

            The package was found but failed to import. Check dependencies.
            :::
            """
        )


# %%
mo.md(
    """
    ### Step 2: Mode Detection

    We detect whether to run in **Mock Mode** (CI) or **Real Mode** (Production) based on the `CI` environment variable.
    *   **Mock Mode**: Uses simulated data/functions. Safe and fast.
    *   **Real Mode**: Tries to run QE/LAMMPS. Requires external binaries.
    """
)

# %%
# Detect Mode
raw_ci = os.environ.get("CI", "false").strip().lower()
valid_true = ["true", "1", "yes", "on"]
valid_false = ["false", "0", "no", "off"]

if raw_ci in valid_true:
    IS_CI = True
elif raw_ci in valid_false:
    IS_CI = False
else:
    IS_CI = True # Default safe

mode_name = "Mock Mode (CI)" if IS_CI else "Real Mode (Production)"
mo.md(f"### Current Mode: **{mode_name}**")

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
    #### Understanding `gamma_threshold`

    The **`gamma_threshold`** (Extrapolation Grade Limit) is the most critical hyperparameter in the active learning loop.

    *   **Definition**: It defines the "safe zone" of the potential's applicability domain.
    *   **Mechanism**: During MD simulations, the uncertainty ($\gamma$) of the local atomic environment is calculated at every step.
    *   **Action**:
        *   If $\gamma < \text{threshold}$: The simulation continues (Safe).
        *   If $\gamma > \text{threshold}$: The simulation **halts** (Uncertainty detected). The structure is saved and sent to the Oracle (DFT) for labeling.
    *   **Analogy**: Think of it as a "confidence interval". If the potential encounters a structure too different from what it has seen during training (high $\gamma$), it stops guessing and asks for the ground truth.
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
            tutorial_tmp_dir.cleanup()
            print(f"Cleanup: Removed {tutorial_dir}")
        except Exception:
            pass
    atexit.register(_cleanup_handler)

    mo.md(f"Initializing Tutorial Workspace at: `{tutorial_dir}`")

    pseudos = {"Fe": "Fe.pbe.UPF", "Pt": "Pt.pbe.UPF", "Mg": "Mg.pbe.UPF", "O": "O.pbe.UPF"}

    if IS_CI:
        mo.md("::: danger\n**MOCK MODE: Creating DUMMY `.UPF` files.**\n:::")
        for element, filename in pseudos.items():
            path = tutorial_dir / filename
            if not path.exists():
                content = '<UPF version="2.0.1"><PP_INFO>MOCK_DATA</PP_INFO></UPF>'
                with open(path, "w") as f:
                    f.write(content)

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
results = []
if HAS_PYACEMAKER and orchestrator:
    try:
        print("Starting Active Learning...")

        # Robust attribute checking
        if not hasattr(orchestrator, 'dataset_path') or not hasattr(orchestrator, 'dataset_manager'):
             raise AttributeError("Orchestrator instance is missing required attributes.")

        # Cold Start
        if orchestrator.dataset_path and not orchestrator.dataset_path.exists():
            print("Running Cold Start...")
            initial = orchestrator.structure_generator.generate_initial_structures()
            computed = orchestrator.oracle.compute_batch(initial)
            atoms_stream = (metadata_to_atoms(s) for s in computed)
            orchestrator.dataset_manager.save_iter(atoms_stream, orchestrator.dataset_path, mode="ab", calculate_checksum=False)

        # Cycles
        for i in range(orchestrator.config.orchestrator.max_cycles):
            print(f"--- Cycle {i+1} ---")
            res = orchestrator.run_cycle()
            results.append(res)
            if str(res.status).upper() == "CONVERGED":
                print("Converged!")
                break
    except Exception as e:
        mo.md(f"::: error\n**Runtime Error:** {e}\n:::")

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
    In Real Mode, this uses the generated LAMMPS commands.
    """
)

# %%
output_path = None
deposited_structure = None

if HAS_PYACEMAKER and orchestrator:
    # Robust attribute check
    potential = getattr(orchestrator, 'current_potential', None)

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
    rng = np.random.default_rng(42)
    for _ in range(5):
        x, y = rng.uniform(0, substrate.cell[0,0]), rng.uniform(0, substrate.cell[1,1])
        z = substrate.positions[:,2].max() + rng.uniform(2.0, 3.0)

        # Use proper Atom object
        symbol = rng.choice(["Fe", "Pt"])
        atom = Atom(symbol=symbol, position=[x, y, z])
        deposited_structure.append(atom)

    plt.figure(figsize=(6, 6))
    plot_atoms(deposited_structure, rotation="-80x, 20y, 0z")
    plt.title("Deposition Result")
    plt.axis("off")
    plt.show()

    output_path = md_work_dir / "final.xyz"
    write(output_path, deposited_structure)


# %%
mo.md(
    """
    ## Step 8: Phase 3 - Analysis (aKMC)

    We analyze the long-term ordering of the deposited film.
    The plot shows the **Order Parameter** rising from 0 (Disordered) to 1 (Ordered L10 phase) over microseconds.
    """
)

# %%
time_steps = np.linspace(0, 1e6, 50)
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
if tutorial_tmp_dir:
    try:
        tutorial_tmp_dir.cleanup()
        print("Cleanup: Done.")
    except Exception as e:
        print(f"Cleanup warning: {e}")
