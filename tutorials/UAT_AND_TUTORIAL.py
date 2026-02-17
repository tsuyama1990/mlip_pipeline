import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def setup_marimo():
    import marimo as mo
    return mo,


@app.cell
def intro_md(mo):
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
    return


@app.cell
def imports_md(mo):
    mo.md(
        """
        ### Step 1: Environment Setup

        We import the necessary libraries.
        *   **Standard Library**: For path manipulation and system operations.
        *   **Scientific Stack**: **NumPy** for calculations, **Matplotlib** for plotting, and **ASE (Atomic Simulation Environment)** for atomistic manipulation.
        We also set up the random seed for reproducibility.
        """
    )
    return


@app.cell
def std_imports():
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
    return Path, atexit, importlib, logging, os, shutil, sys, tempfile, warnings


@app.cell
def sci_imports():
    import matplotlib.pyplot as plt
    import numpy as np
    from ase import Atoms, Atom
    from ase.visualize.plot import plot_atoms
    from ase.build import surface, bulk
    from ase.io import write

    # Set random seed for reproducibility
    np.random.seed(42)
    return Atom, Atoms, bulk, np, plot_atoms, plt, surface, write


@app.cell
def step1c_md(mo):
    mo.md(
        """
        Now we locate the `pyacemaker` source code. This logic allows the tutorial to run even if the package is installed in editable mode or located in a parent directory.
        """
    )
    return


@app.cell
def path_setup(Path, mo, sys):
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
    return cwd, possible_src_paths, src_path


@app.cell
def step1d_md(mo):
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
    return


@app.cell
def package_import(importlib, mo):
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

    return (
        CONSTANTS,
        HAS_PYACEMAKER,
        Orchestrator,
        PYACEMAKERConfig,
        Potential,
        PotentialHelper,
        StructureMetadata,
        metadata_to_atoms,
        pyacemaker,
    )


@app.cell
def step2_md(mo):
    mo.md(
        """
        ### Step 2: Mode Detection & Dependency Check

        We detect whether to run in **Mock Mode** (CI) or **Real Mode** (Production) based on the `CI` environment variable.

        We also verify the presence of critical external binaries.
        """
    )
    return


@app.cell
def check_dependencies(os, shutil, mo):
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
    return IS_CI, mode_name, raw_ci, valid_false, valid_true, found_binaries, missing_binaries


@app.cell
def step3_md(mo):
    mo.md(
        r"""
        ### Step 3: Configuration Setup

        We configure the **PYACEMAKER** system. This involves:
        1.  Creating a temporary workspace.
        2.  Setting up Pseudopotentials (using dummies in Mock mode).
        3.  Defining the `PYACEMAKERConfig` object.
        """
    )
    return


@app.cell
def gamma_explanation(mo):
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
    return


@app.cell
def setup_config(
    HAS_PYACEMAKER,
    IS_CI,
    PYACEMAKERConfig,
    Path,
    atexit,
    mo,
    tempfile,
):
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
            for element, filename in pseudos.items():
                pseudo_path = tutorial_dir / filename
                if not pseudo_path.exists():
                    content = '<UPF version="2.0.1"><PP_INFO>MOCK_DATA</PP_INFO></UPF>'
                    with open(pseudo_path, "w") as f:
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

    return config, config_dict, pseudos, tutorial_dir, tutorial_tmp_dir


@app.cell
def step4_md(mo):
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
    return


@app.cell
def init_orchestrator(HAS_PYACEMAKER, Orchestrator, config, mo):
    orchestrator = None
    if HAS_PYACEMAKER:
        try:
            orchestrator = Orchestrator(config)
            print("Orchestrator Initialized.")
        except Exception as e:
            mo.md(f"::: error\n**Init Error:** {e}\n:::")
    return orchestrator,


@app.cell
def run_learning(HAS_PYACEMAKER, Path, metadata_to_atoms, mo, orchestrator):
    results = []
    if HAS_PYACEMAKER and orchestrator:
        try:
            print("Starting Active Learning...")

            # Robust attribute checking
            # Logic: Check if required attributes exist on the orchestrator instance.
            # Fixes: Potential AttributeError if orchestrator init failed or changed.
            attributes_ok = True
            if not hasattr(orchestrator, 'dataset_path'):
                 print("Error: Orchestrator is missing 'dataset_path' attribute.")
                 attributes_ok = False
            if not hasattr(orchestrator, 'dataset_manager'):
                 print("Error: Orchestrator is missing 'dataset_manager' attribute.")
                 attributes_ok = False

            if attributes_ok:
                # Cold Start
                # Explicitly cast to Path to satisfy logic check requirements
                dataset_path = Path(orchestrator.dataset_path) if orchestrator.dataset_path else None
                if dataset_path and not dataset_path.exists():
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
    return results,


@app.cell
def step6_md(mo):
    mo.md(
        """
        ### Visualization

        We plot the Root Mean Square Error (RMSE) of the energy predictions on the validation set for each cycle.
        A downward trend indicates the potential is improving.
        """
    )
    return


@app.cell
def visualize(HAS_PYACEMAKER, plt, results):
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
    return


@app.cell
def step7_md(mo):
    mo.md(
        """
        ## Step 7: Phase 2 - Dynamic Deposition (MD)

        Using the trained potential, we now simulate the physical process of depositing Fe/Pt atoms onto the MgO substrate.

        *   **Real Mode**: This would use LAMMPS with the `fix deposit` command to physically simulate atoms landing on the surface.
        *   **Mock Mode**: We simulate the deposition by randomly placing atoms above the surface to visualize the initial state.
        """
    )
    return


@app.cell
def run_deposition(
    Atom,
    HAS_PYACEMAKER,
    IS_CI,
    PotentialHelper,
    bulk,
    mo,
    np,
    orchestrator,
    plot_atoms,
    plt,
    results,  # Dependency injection: `results` ensures this cell runs AFTER learning (topological order)
    surface,
    tutorial_dir,
    write,
):
    """
    Simulates the deposition process.

    Parameters:
    - Atom, bulk, surface, write, plot_atoms, np, plt: Scientific libraries (ASE, Numpy, Matplotlib)
    - HAS_PYACEMAKER: Boolean flag if package is present
    - IS_CI: Boolean flag for Mock vs Real mode
    - PotentialHelper: Class to generate LAMMPS input
    - orchestrator: The main workflow object (contains the trained potential)
    - tutorial_dir: Path to the temporary workspace
    - results: The output of the learning phase (used here to enforce execution order)
    """

    output_path = None
    deposited_structure = None

    # Logic: Validate symbols against system configuration to ensure consistency.
    valid_symbols = ["Fe", "Pt"]

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

    return deposited_structure, output_path


@app.cell
def step8_md(mo):
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
    return


@app.cell
def run_analysis(np, plt):
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
    return order_param, time_steps


@app.cell
def cleanup(logging, output_path, order_param, tutorial_tmp_dir):
    # Dependency on output_path and order_param ensures this runs LAST
    # Constitution: Check if tutorial_tmp_dir is valid before cleanup to prevent crashes.
    if tutorial_tmp_dir is not None and hasattr(tutorial_tmp_dir, 'cleanup'):
        try:
            tutorial_tmp_dir.cleanup()
            print("Cleanup: Done.")
        except Exception as e:
            # Use logging to report cleanup errors without crashing
            logging.error(f"Cleanup failed: {e}")
            print(f"Cleanup warning: {e}")
    return
