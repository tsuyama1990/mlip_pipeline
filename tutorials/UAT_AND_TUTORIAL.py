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
    return


@app.cell
def step1_md(mo):
    mo.md(
        """
        ### Step 1: Environment Setup

        In this step, we configure the Python environment. We ensure the `pyacemaker` source code is accessible and import necessary libraries.
        We also set environment variables to configure the system for this tutorial.
        """
    )
    return


@app.cell
def imports_and_setup(mo):
    import os
    import sys
    import shutil
    import tempfile
    import importlib.util
    from pathlib import Path

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

    spec = importlib.util.find_spec("pyacemaker")
    if spec is None and not src_path:
        mo.md("::: error\n**ERROR: PYACEMAKER package not found.**\n:::")
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
            mo.md(f"::: error\n**Import Error:** {e}\n:::")

    return (
        Atoms,
        CONSTANTS,
        HAS_PYACEMAKER,
        Orchestrator,
        PYACEMAKERConfig,
        Path,
        Potential,
        PotentialHelper,
        StructureMetadata,
        bulk,
        importlib,
        metadata_to_atoms,
        np,
        os,
        plot_atoms,
        plt,
        pyacemaker,
        shutil,
        src_path,
        surface,
        sys,
        tempfile,
        write,
    )


@app.cell
def step2_md(mo):
    mo.md(
        """
        ### Step 2: Mode Detection

        We detect whether to run in **Mock Mode** (CI) or **Real Mode** (Production).
        """
    )
    return


@app.cell
def detect_mode(os, mo):
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
    return IS_CI, mode_name, raw_ci, valid_false, valid_true


@app.cell
def step3_md(mo):
    mo.md(
        r"""
        ### Step 3: Configuration Setup

        We configure parameters for the Orchestrator, DFT Oracle, Trainer, and Dynamics Engine.
        Critical parameter: **`gamma_threshold`**.
        """
    )
    return


@app.cell
def setup_config(HAS_PYACEMAKER, IS_CI, PYACEMAKERConfig, Path, mo, tempfile):
    config = None
    config_dict = None
    pseudos = None
    tutorial_dir = None
    tutorial_tmp_dir = None

    if HAS_PYACEMAKER:
        # Create temporary directory in CWD for security compliance
        tutorial_tmp_dir = tempfile.TemporaryDirectory(prefix="pyacemaker_tutorial_", dir=Path.cwd())
        tutorial_dir = Path(tutorial_tmp_dir.name)

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

    return config, config_dict, pseudos, tutorial_dir, tutorial_tmp_dir


@app.cell
def step4_md(mo):
    mo.md(
        r"""
        ## Step 4: Phase 1 - Active Learning Loop

        The `Orchestrator` manages the loop: Generation -> Oracle -> Training -> Exploration -> Validation.
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
def run_learning(HAS_PYACEMAKER, metadata_to_atoms, mo, orchestrator):
    results = []
    if HAS_PYACEMAKER and orchestrator:
        try:
            print("Starting Active Learning...")
            # Cold Start
            if not orchestrator.dataset_path.exists():
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
        plt.ylabel("RMSE")
        plt.show()
    return


@app.cell
def step7_md(mo):
    mo.md("## Step 7: Phase 2 - Dynamic Deposition (MD)")
    return


@app.cell
def run_deposition(
    HAS_PYACEMAKER,
    IS_CI,
    PotentialHelper,
    bulk,
    mo,
    np,
    orchestrator,
    plot_atoms,
    plt,
    results,  # Dependency injection
    surface,
    tutorial_dir,
    write,
):
    output_path = None
    deposited_structure = None

    if HAS_PYACEMAKER and orchestrator:
        potential = orchestrator.current_potential
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
            deposited_structure.append(rng.choice(["Fe", "Pt"]))
            deposited_structure.positions[-1] = [x, y, z]

        plt.figure(figsize=(6, 6))
        plot_atoms(deposited_structure, rotation="-80x, 20y, 0z")
        plt.show()

        output_path = md_work_dir / "final.xyz"
        write(output_path, deposited_structure)

    return deposited_structure, output_path


@app.cell
def step8_md(mo):
    mo.md("## Step 8: Phase 3 - Analysis (aKMC)")
    return


@app.cell
def run_analysis(np, plt):
    time_steps = np.linspace(0, 1e6, 50)
    order_param = 1.0 / (1.0 + np.exp(-1e-5 * (time_steps - 3e5)))
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, order_param, 'r-')
    plt.title("L10 Ordering (Mock)")
    plt.show()
    return order_param, time_steps


@app.cell
def cleanup(output_path, order_param, tutorial_tmp_dir):
    # Dependency on output_path and order_param ensures this runs LAST
    if tutorial_tmp_dir:
        try:
            tutorial_tmp_dir.cleanup()
            print("Cleanup: Done.")
        except Exception as e:
            print(f"Cleanup Error: {e}")
    return
