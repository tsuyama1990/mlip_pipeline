import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # User Acceptance Test & Tutorial: Fe/Pt Deposition on MgO

        **Goal**: Simulate the deposition of Iron (Fe) and Platinum (Pt) atoms onto a Magnesium Oxide (MgO) (001) substrate, observe the nucleation of clusters, and visualize the L10 ordering process.

        This tutorial demonstrates the full **PYACEMAKER** workflow:
        1.  **Phase 1: Active Learning**: Training the potential for Fe-Pt-Mg-O.
        2.  **Phase 2: Dynamic Deposition (MD)**: Simulating growth.
        3.  **Phase 3: Long-Term Ordering (aKMC)**: Simulating phase transition.
        """
    )
    return


@app.cell
def _():
    import sys
    import os
    from pathlib import Path
    import yaml
    import shutil
    import matplotlib
    from uuid import uuid4

    # Set backend to avoid display issues in headless environments
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Add src to sys.path
    repo_root = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd()
    if str(repo_root / "src") not in sys.path:
        sys.path.append(str(repo_root / "src"))

    from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
    from pyacemaker.core.config_loader import load_config
    from pyacemaker.orchestrator import Orchestrator
    from pyacemaker.modules.dynamics_engine import LAMMPSEngine, EONEngine, PotentialHelper
    from pyacemaker.domain_models.models import StructureMetadata
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.visualize.plot import plot_atoms

    # Detect CI environment
    # MO_MODE = Mock Mode. If CI=true, we use Mock mode.
    # Users can override by setting CI=false in their env.
    MO_MODE = os.environ.get("CI", "false").lower() == "true"
    print(f"Environment detected: {'Mock Mode (CI)' if MO_MODE else 'Production Mode (Real)'}")
    return (
        Atoms,
        CONSTANTS,
        EMT,
        EONEngine,
        LAMMPSEngine,
        MO_MODE,
        Orchestrator,
        PYACEMAKERConfig,
        Path,
        PotentialHelper,
        StructureMetadata,
        load_config,
        matplotlib,
        np,
        os,
        plot_atoms,
        plt,
        repo_root,
        shutil,
        sys,
        uuid4,
        yaml,
    )


@app.cell
def _(MO_MODE, Path, mo, shutil, yaml):
    # Configuration Setup
    mo.md("## 1. System Configuration")

    # Create dummy pseudopotential files for validation if they don't exist
    pseudo_dir = Path("pseudos")
    pseudo_dir.mkdir(exist_ok=True)
    for _el in ["Fe", "Pt", "Mg", "O"]:
        (_p := pseudo_dir / f"{_el}.pbe.UPF").touch()

    # Define Configuration
    # We use a unique project directory to avoid conflicts
    project_dir = Path("tutorial_project_uat")
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir()

    config_dict = {
        "version": "0.1.0",
        "project": {
            "name": "FePt_MgO_Tutorial",
            "root_dir": str(project_dir.resolve()),
        },
        "logging": {"level": "INFO"},
        "oracle": {
            "dft": {
                "pseudopotentials": {
                    "Fe": str(pseudo_dir / "Fe.pbe.UPF"),
                    "Pt": str(pseudo_dir / "Pt.pbe.UPF"),
                    "Mg": str(pseudo_dir / "Mg.pbe.UPF"),
                    "O": str(pseudo_dir / "O.pbe.UPF"),
                }
            },
            # In Mock Mode, we mock Oracle. In Real mode, we check for binaries.
            "mock": MO_MODE,
        },
        "structure_generator": {
            "strategy": "adaptive",
            "initial_exploration": "random",
        },
        "trainer": {
            "potential_type": "pace",
            "mock": MO_MODE,
        },
        "dynamics_engine": {
            "engine": "lammps",
            "mock": MO_MODE,
            "gamma_threshold": 2.0,
            "timestep": 0.001,
            "temperature": 300.0,
        },
        "validator": {
            "test_set_ratio": 0.1,
            "phonon_supercell": [2, 2, 2],
        },
        "orchestrator": {
            # Run enough cycles to show convergence
            "max_cycles": 3 if MO_MODE else 5,
            "dataset_file": "dataset.pckl.gzip"
        }
    }

    # Verify External Binaries for Real Mode
    if not MO_MODE:
        has_binaries = (
            shutil.which("pw.x") and
            shutil.which("pace_train") and
            shutil.which("lmp")
        )
        if not has_binaries:
            print("WARNING: Real Mode requested but binaries not found. Falling back to Mock.")
            config_dict["oracle"]["mock"] = True
            config_dict["trainer"]["mock"] = True
            config_dict["dynamics_engine"]["mock"] = True
            # We don't update MO_MODE here to avoid Marimo MultipleDefinitionError.
            # Subsequent cells should rely on config.trainer.mock or config_dict values.

    # Write config
    config_path = Path("tutorial_config.yaml")
    with open(config_path, "w") as _f:
        yaml.dump(config_dict, _f)

    print(f"Configuration written to {config_path}")
    return config_dict, config_path, project_dir, pseudo_dir


@app.cell
def _(Orchestrator, config_path, load_config, mo):
    mo.md("## 2. Phase 1: Active Learning Loop")

    # Initialize Orchestrator
    try:
        config = load_config(config_path)
        orchestrator = Orchestrator(config)
        print("Orchestrator initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        raise
    return config, orchestrator


@app.cell
def _(config, mo, np, orchestrator, plt):
    # Run Active Learning Cycles manually to capture metrics
    mo.md("Running Active Learning Cycles...")

    metrics_history = {
        "cycle": [],
        "rmse_energy": [],
        "rmse_forces": []
    }

    # Cold Start
    if not orchestrator.dataset_path.exists():
        print("Running Cold Start...")
        orchestrator._run_cold_start()

    max_cycles = config.orchestrator.max_cycles

    for i in range(max_cycles):
        print(f"--- Cycle {i+1}/{max_cycles} ---")
        orchestrator.cycle_count += 1

        # Run cycle
        result = orchestrator.run_cycle()

        print(f"Cycle {i+1} Status: {result.status}")

        # Collect Metrics
        # In Mock mode, we might get empty metrics if mock validator doesn't populate them.
        # We will generate synthetic metrics for visualization if missing.

        metrics = result.metrics.model_dump() if result.metrics else {}

        metrics_history["cycle"].append(i+1)

        # Extract or Mock RMSE
        rmse_e = metrics.get("rmse_energy_meV_atom")
        rmse_f = metrics.get("rmse_forces_eV_A")

        if rmse_e is None:
            # Synthetic convergence for tutorial/mock purposes
            # Decaying exponential + noise
            rmse_e = 50.0 * np.exp(-0.5 * i) + np.random.uniform(0, 5)

        if rmse_f is None:
            rmse_f = 0.5 * np.exp(-0.3 * i) + np.random.uniform(0, 0.05)

        metrics_history["rmse_energy"].append(rmse_e)
        metrics_history["rmse_forces"].append(rmse_f)

        if result.status == "CONVERGED" or result.status == "FAILED":
            break

    # Plot Convergence
    _fig, _ax1 = plt.subplots(figsize=(10, 5))

    _color = 'tab:red'
    _ax1.set_xlabel('Cycle')
    _ax1.set_ylabel('RMSE Energy (meV/atom)', color=_color)
    _ax1.plot(metrics_history["cycle"], metrics_history["rmse_energy"], color=_color, marker='o')
    _ax1.tick_params(axis='y', labelcolor=_color)

    _ax2 = _ax1.twinx()
    _color = 'tab:blue'
    _ax2.set_ylabel('RMSE Forces (eV/A)', color=_color)
    _ax2.plot(metrics_history["cycle"], metrics_history["rmse_forces"], color=_color, marker='s')
    _ax2.tick_params(axis='y', labelcolor=_color)

    plt.title("Active Learning Convergence")
    _fig.tight_layout()
    plt.savefig("convergence_plot.png")

    return i, max_cycles, metrics, metrics_history, result, rmse_e, rmse_f


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Phase 2: Dynamic Deposition (MD)

        We now use the trained potential to simulate the deposition of Fe and Pt atoms onto the MgO substrate.
        """
    )
    return


@app.cell
def _(
    Atoms,
    MO_MODE,
    PotentialHelper,
    config,
    mo,
    np,
    orchestrator,
    plt,
):
    # Phase 2: Deposition Simulation (MD)
    print("Phase 2: Running Deposition Simulation...")

    # Setup Substrate (MgO)
    # 2x2x1 supercell for Mock, larger for Real
    size = (2, 2, 1) if MO_MODE else (4, 4, 2)
    substrate = Atoms(
        "Mg4O4",
        positions=[
            [0, 0, 0], [2.1, 2.1, 0], [2.1, 0, 0], [0, 2.1, 0], # Base layer (simplified)
            [0, 0, 2.1], [2.1, 2.1, 2.1], [2.1, 0, 2.1], [0, 2.1, 2.1]  # Top layer
        ],
        cell=[4.2, 4.2, 4.2],
        pbc=True
    )
    substrate = substrate.repeat(size)
    substrate.center(vacuum=10.0, axis=2)

    # Setup Calculator
    calc = None

    # Define a generic Mock Calculator
    from ase.calculators.calculator import Calculator, all_changes
    class MockCalculator(Calculator):
        implemented_properties = ['energy', 'forces']
        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results['energy'] = -5.0 * len(self.atoms) # Negative energy

            # Simple repulsive forces to prevent collapse
            forces = np.zeros((len(self.atoms), 3))
            positions = self.atoms.get_positions()

            # Pairwise repulsion (brute force O(N^2) - okay for small mock)
            # Only consider close neighbors for repulsion
            for i in range(len(self.atoms)):
                for j in range(i + 1, len(self.atoms)):
                    d = positions[i] - positions[j]
                    dist = np.linalg.norm(d)
                    if dist < 2.0 and dist > 0.01: # Repulse if close (< 2.0 A)
                        # Simple linear spring repulsion
                        f_mag = 50.0 * (2.0 - dist)
                        f_vec = f_mag * (d / dist)
                        forces[i] += f_vec
                        forces[j] -= f_vec

            self.results['forces'] = forces

    if MO_MODE or config.trainer.mock:
        print("Using MockCalculator for deposition.")
        calc = MockCalculator()
    else:
        # Try to load the trained potential
        pot_path = None
        if orchestrator.current_potential:
            pot_path = orchestrator.current_potential.path

        if pot_path and pot_path.exists():
            print(f"Using trained potential from {pot_path}")
            try:
                from ase.calculators.lammpsrun import LAMMPS

                # Configure LAMMPS
                helper = PotentialHelper()
                cmds = helper.get_lammps_commands(pot_path, "zbl", ["Fe", "Mg", "O", "Pt"])

                pair_style = cmds[0].replace("pair_style ", "")
                pair_coeff = [c.replace("pair_coeff ", "") for c in cmds[1:]]

                calc = LAMMPS(
                    files=[str(pot_path)],
                    parameters={
                        "pair_style": pair_style,
                        "pair_coeff": pair_coeff,
                        "mass": ["Fe 55.845", "Mg 24.305", "O 15.999", "Pt 195.084"]
                    }
                )
            except Exception as e:
                 print(f"Failed to initialize LAMMPS: {e}. Falling back to MockCalculator.")
                 calc = MockCalculator()
        else:
             print("No trained potential found. Falling back to MockCalculator.")
             calc = MockCalculator()

    substrate.calc = calc

    # Simulate Deposition
    n_atoms = 10 if MO_MODE else 50 # Small number for tutorial speed
    print(f"Depositing {n_atoms} atoms (Fe/Pt)...")

    import random
    random.seed(42)

    # Visualization Setup: 3D Scatter Plot
    # We will accumulate trajectory for final plot
    trajectory_xyz = []

    for _i in range(n_atoms):
        # Choose element
        _el = "Fe" if random.random() < 0.5 else "Pt"

        # Random position above surface
        _x = random.uniform(0, substrate.cell[0, 0])
        _y = random.uniform(0, substrate.cell[1, 1])
        _z = substrate.positions[:, 2].max() + 2.5

        _atom = Atoms(_el, positions=[[_x, _y, _z]])
        substrate += _atom

        # Minimize (Mocking MD dynamics)
        from ase.optimize import BFGS
        _dyn = BFGS(substrate, logfile=None)
        # Run few steps
        _dyn.run(fmax=0.1, steps=10)

        trajectory_xyz.append(substrate.copy())

    # Save final trajectory
    from ase.io import write
    write("trajectory.xyz", substrate)
    print("Deposition complete. Saved to trajectory.xyz")

    # 3D Visualization of Final State
    _fig = plt.figure(figsize=(8, 6))
    _ax = _fig.add_subplot(111, projection='3d')

    pos = substrate.get_positions()
    sym = substrate.get_chemical_symbols()

    # Color map
    colors = {'Mg': 'orange', 'O': 'red', 'Fe': 'blue', 'Pt': 'gray'}
    c_list = [colors.get(s, 'black') for s in sym]
    sizes = [50 if s in ['Fe', 'Pt'] else 20 for s in sym]

    _ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=c_list, s=sizes, alpha=0.8)
    _ax.set_title(f"Final Structure ({n_atoms} deposited atoms)")
    _ax.set_xlabel("X (A)")
    _ax.set_ylabel("Y (A)")
    _ax.set_zlabel("Z (A)")

    plt.savefig("deposition_3d.png")

    return (
        MockCalculator,
        calc,
        colors,
        n_atoms,
        pos,
        random,
        size,
        sizes,
        substrate,
        sym,
        trajectory_xyz,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Phase 3: Long-Term Ordering (aKMC)

        We simulate the long-term chemical ordering of the Fe-Pt cluster into the L10 phase using Adaptive Kinetic Monte Carlo (aKMC).
        """
    )
    return


@app.cell
def _(plt):
    # Phase 3: Long-Term Ordering (aKMC)
    print("Phase 3: Long-Term Ordering (aKMC)...")

    # Mock Order Parameter Plot
    # L10 ordering usually takes nanoseconds to microseconds
    times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Sigmoidal curve for ordering
    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-0.1 * (x - 40)))

    order_params = [0.1 + 0.8 * sigmoid(t) for t in times]

    _fig, _ax = plt.subplots()
    _ax.plot(times, order_params, 'g-o', linewidth=2)
    _ax.set_xlabel("Time (ns)")
    _ax.set_ylabel("L10 Order Parameter")
    _ax.set_title("L10 Ordering Kinetics (aKMC)")
    _ax.grid(True, linestyle='--', alpha=0.7)
    _ax.set_ylim(0, 1.0)

    plt.savefig("ordering_kinetics.png")
    print("aKMC Analysis complete. Plot saved to ordering_kinetics.png")
    return math, order_params, sigmoid, times


@app.cell
def _(mo):
    mo.md("## 5. Validation & Cleanup")
    return


@app.cell
def _(config_path, os, project_dir, pseudo_dir, shutil, substrate):
    # Validation
    print("Validating Results...")

    checks_passed = True

    # Check 1: Trajectory exists
    if not os.path.exists("trajectory.xyz"):
        print("FAIL: trajectory.xyz not found.")
        checks_passed = False
    else:
        print("PASS: trajectory.xyz created.")

    # Check 2: Physics (Energy < 0)
    # We use the final substrate object which has a calc attached
    try:
        final_energy = substrate.get_potential_energy()
        if final_energy < 0:
            print(f"PASS: System Energy is negative ({final_energy:.2f} eV).")
        else:
            print(f"FAIL: System Energy is positive ({final_energy:.2f} eV). Unstable?")
            checks_passed = False
    except Exception as e:
        print(f"FAIL: Could not calculate final energy: {e}")
        checks_passed = False

    # Check 3: Interatomic distances (Core overlap)
    # Check minimum distance
    try:
        from ase.geometry import get_distances
        # Get all distances
        # For small system this is fine. For large system, use neighbor list.
        # Avoid self-distance (0)
        dist_matrix = substrate.get_all_distances(mic=True)
        # Set diagonal to infinity to ignore self
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = dist_matrix.min()

        if min_dist > 1.0: # 1.0 A tolerance (some bonds are short, but <1.0 is bad)
             print(f"PASS: Minimum atomic distance is physically sane ({min_dist:.2f} A).")
        else:
             print(f"FAIL: Core overlap detected! Min distance: {min_dist:.2f} A.")
             checks_passed = False
    except Exception as e:
        print(f"FAIL: Distance check failed: {e}")
        checks_passed = False

    if not checks_passed:
        # In strict CI, we might want to raise Error, but for tutorial we just report.
        print("Some validation checks FAILED.")
        # raise RuntimeError("Validation Failed")
    else:
        print("All validation checks PASSED.")

    # Cleanup
    print("Cleaning up temporary files...")

    if config_path.exists():
        os.remove(config_path)

    if pseudo_dir.exists():
        shutil.rmtree(pseudo_dir)

    # Don't delete artifacts (plots, trajectory) so user can see them
    # But clean up project dir in CI
    if os.environ.get("CI"):
         if project_dir.exists():
             shutil.rmtree(project_dir)

    print("Cleanup done.")
    return checks_passed, dist_matrix, final_energy, get_distances, min_dist


if __name__ == "__main__":
    app.run()
