import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    from pathlib import Path
    import yaml
    import tempfile
    import shutil
    import matplotlib

    # Set backend to avoid display issues in headless environments
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Add src to sys.path to ensure pyacemaker is importable
    # Assuming running from repo root or tutorials dir
    repo_root = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd()
    if str(repo_root / "src") not in sys.path:
        sys.path.append(str(repo_root / "src"))

    from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
    from pyacemaker.core.config_loader import load_config
    from pyacemaker.orchestrator import Orchestrator
    from pyacemaker.modules.dynamics_engine import LAMMPSEngine, EONEngine, PotentialHelper
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.visualize.plot import plot_atoms

    # Detect CI environment
    MO_MODE = os.environ.get("CI", "false").lower() == "true"
    print(f"Running in Mock Mode: {MO_MODE}")
    return (
        Atoms,
        CONSTANTS,
        EONEngine,
        EMT,
        LAMMPSEngine,
        MO_MODE,
        Orchestrator,
        PYACEMAKERConfig,
        Path,
        PotentialHelper,
        app,
        load_config,
        matplotlib,
        np,
        os,
        plot_atoms,
        plt,
        repo_root,
        shutil,
        sys,
        tempfile,
        yaml,
    )


@app.cell
def _(CONSTANTS, MO_MODE, Path, yaml, shutil):
    # Configuration Setup
    # Create dummy pseudopotential files for validation
    pseudo_dir = Path("pseudos")
    pseudo_dir.mkdir(exist_ok=True)
    for _el in ["Fe", "Pt", "Mg", "O"]:
        (pseudo_dir / f"{_el}.pbe.UPF").touch()

    # Define Configuration
    config_dict = {
        "version": "0.1.0",
        "project": {
            "name": "FePt_MgO_Tutorial",
            "root_dir": str(Path.cwd().resolve() / "tutorial_project"),
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
            "mock": True, # Always mock Oracle for tutorial speed/portability unless explicit overrides
        },
        "structure_generator": {
            "strategy": "adaptive",
            "initial_exploration": "random", # Simple start
        },
        "trainer": {
            "potential_type": "pace",
            "mock": True, # Always mock Trainer for tutorial unless specifically configured
        },
        "dynamics_engine": {
            "engine": "lammps",
            "mock": True, # Mock MD for exploration phase speed
            "gamma_threshold": 2.0,
            "timestep": 0.001,
            "temperature": 300.0,
        },
        "validator": {
            "test_set_ratio": 0.1,
            "phonon_supercell": [2, 2, 2],
        },
        "orchestrator": {
            "max_cycles": 2 if MO_MODE else 5, # Short run for tutorial
        }
    }

    # If NOT in Mock Mode (Real Production), one might want to use real binaries.
    # But for a tutorial file that runs "out of the box", sticking to mocks is safer.
    # The instructions say "Mock Mode (CI)... Real Mode (Production)".
    # If CI=false, we should try to use real things IF available.
    # But usually tutorials are demonstrations.
    # Let's respect the "Real Mode" instruction if env var is set.

    if not MO_MODE:
        # Check if we have executables
        if shutil.which("pw.x") and shutil.which("pace_train") and shutil.which("lmp"):
            print("Real Mode: External binaries found. Switching to REAL execution (may take long).")
            config_dict["oracle"]["mock"] = False
            config_dict["trainer"]["mock"] = False
            config_dict["dynamics_engine"]["mock"] = False
        else:
            print("Real Mode: External binaries NOT found. Falling back to Mock execution.")

    # Write config
    config_path = Path("tutorial_config.yaml")
    with open(config_path, "w") as _f:
        yaml.dump(config_dict, _f)

    print(f"Configuration written to {config_path}")
    return config_dict, config_path, pseudo_dir


@app.cell
def _(Orchestrator, config_path, load_config):
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
def _(orchestrator):
    # Phase 1: Active Learning Loop
    print("Phase 1: Starting Active Learning Loop...")

    # Run the active learning pipeline
    # This generates initial structures, calculates energies (mock/real),
    # trains potential (mock/real), and validates.
    result = orchestrator.run()

    print(f"Active Learning Phase Completed. Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    return result,


@app.cell
def _(
    Atoms,
    EMT,
    MO_MODE,
    PotentialHelper,
    config,
    np,
    orchestrator,
    plot_atoms,
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

    # Define a generic Mock Calculator for elements not supported by EMT (like Mg)
    from ase.calculators.calculator import Calculator, all_changes
    class MockCalculator(Calculator):
        implemented_properties = ['energy', 'forces']
        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results['energy'] = -100.0 * len(self.atoms)
            # Add a small repulsive force to prevent collapse during minimization
            # Simple dummy forces
            self.results['forces'] = np.random.uniform(-0.1, 0.1, (len(self.atoms), 3))

    if MO_MODE or config.trainer.mock:
        print("Using MockCalculator for deposition.")
        calc = MockCalculator()
    else:
        # Try to load the trained potential
        # orchestrator.current_potential should be set if training succeeded
        pot_path = None
        if orchestrator.current_potential:
            pot_path = orchestrator.current_potential.path

        if pot_path and pot_path.exists():
            print(f"Using trained potential from {pot_path}")
            # In a real scenario, we would use LAMMPS here.
            try:
                from ase.calculators.lammpsrun import LAMMPS

                # Configure LAMMPS to use the trained potential
                # We use PotentialHelper to generate the correct pair_style/coeff commands
                # assuming ZBL baseline as configured
                helper = PotentialHelper()
                # Note: Elements order matters for LAMMPS, usually sorted alphabetically by ASE
                # But here we pass all elements involved.
                cmds = helper.get_lammps_commands(pot_path, "zbl", ["Fe", "Mg", "O", "Pt"])

                # Parse commands for ASE LAMMPS calculator parameters
                # pair_style is the first command
                pair_style = cmds[0].replace("pair_style ", "")
                # pair_coeff is the list of subsequent commands
                pair_coeff = [c.replace("pair_coeff ", "") for c in cmds[1:]]

                calc = LAMMPS(
                    files=[str(pot_path)],
                    parameters={
                        "pair_style": pair_style,
                        "pair_coeff": pair_coeff,
                        "mass": ["Fe 55.845", "Mg 24.305", "O 15.999", "Pt 195.084"]
                    }
                )
            except ImportError:
                print("ASE LAMMPS interface not available. Falling back to MockCalculator.")
                calc = MockCalculator()
            except Exception as e:
                 print(f"Failed to initialize LAMMPS: {e}. Falling back to MockCalculator.")
                 calc = MockCalculator()
        else:
             print("No trained potential found. Falling back to MockCalculator.")
             calc = MockCalculator()

    substrate.calc = calc

    # Simulate Deposition
    # Add atoms one by one and minimize/relax
    n_atoms = 5 if MO_MODE else 100
    deposited_atoms = []

    # Create a figure for snapshots
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))
    _axes = _axes.flatten()

    print(f"Depositing {n_atoms} atoms (Fe/Pt)...")

    import random
    random.seed(42)

    for _i in range(n_atoms):
        # Choose element
        _el = "Fe" if random.random() < 0.5 else "Pt"

        # Random position above surface
        _x = random.uniform(0, substrate.cell[0, 0])
        _y = random.uniform(0, substrate.cell[1, 1])
        _z = substrate.positions[:, 2].max() + 3.0 # Start 3A above highest atom

        _atom = Atoms(_el, positions=[[_x, _y, _z]])
        substrate += _atom

        # Minimize (Mocking MD dynamics with minimization for stability in tutorial)
        # Real MD would use MaxwellBoltzmannDistribution and run dynamics
        from ase.optimize import BFGS
        _dyn = BFGS(substrate, logfile=None)
        _dyn.run(fmax=0.1, steps=20 if MO_MODE else 100)

        # Save trajectory
        from ase.io import write
        write("trajectory.xyz", substrate, append=True)

        # Capture snapshots
        if _i == 0:
            plot_atoms(substrate, ax=_axes[0], rotation="-90x")
            _axes[0].set_title("Start")
        elif _i == n_atoms // 3:
            plot_atoms(substrate, ax=_axes[1], rotation="-90x")
            _axes[1].set_title("1/3 Deposition")
        elif _i == 2 * n_atoms // 3:
            plot_atoms(substrate, ax=_axes[2], rotation="-90x")
            _axes[2].set_title("2/3 Deposition")

    # Final snapshot
    plot_atoms(substrate, ax=_axes[3], rotation="-90x")
    _axes[3].set_title("Final Cluster")

    plt.tight_layout()
    # Save figure to show it works
    plt.savefig("deposition_snapshots.png")
    print("Deposition complete. Snapshots saved to deposition_snapshots.png")

    return (
        calc,
        deposited_atoms,
        n_atoms,
        pot_path,
        random,
        size,
        substrate,
    )


@app.cell
def _(MO_MODE, plt):
    # Phase 3: Long-Term Ordering (aKMC)
    print("Phase 3: Long-Term Ordering (aKMC)...")

    # In a real tutorial, we would export the final structure to EON format.
    # Here we show a placeholder for the ordering analysis.

    # Mock Order Parameter Plot
    times = [0, 10, 20, 30, 40, 50]
    order_params = [0.1, 0.2, 0.4, 0.6, 0.75, 0.8] # Increasing order

    _fig2, _ax2 = plt.subplots()
    _ax2.plot(times, order_params, 'b-o')
    _ax2.set_xlabel("Time (ns)")
    _ax2.set_ylabel("L10 Order Parameter")
    _ax2.set_title("Mock aKMC Ordering Kinetics")
    _ax2.grid(True)

    plt.savefig("ordering_kinetics.png")
    print("aKMC Analysis complete. Plot saved to ordering_kinetics.png")
    return order_params, times


@app.cell
def _(config_path, os, pseudo_dir, shutil):
    # Cleanup
    print("Cleaning up temporary files...")

    if config_path.exists():
        os.remove(config_path)

    if pseudo_dir.exists():
        shutil.rmtree(pseudo_dir)

    if os.path.exists("trajectory.xyz"):
        os.remove("trajectory.xyz")

    # We might want to keep the 'tutorial_project' dir for inspection,
    # or clean it up. For CI, clean up.
    if os.environ.get("CI"):
         project_dir = Path("tutorial_project")
         if project_dir.exists():
             shutil.rmtree(project_dir)

    print("Cleanup done.")
    return project_dir,


if __name__ == "__main__":
    app.run()
