import os
import sys
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read


def get_calculator(potential_path: str) -> Calculator:
    """
    Factory to get the calculator.
    In production, this would load the .yace potential using pyace or similar.
    For this mock, we might need a fallback or real implementation.
    """
    try:
        from pyace import PyACECalculator  # type: ignore

        return PyACECalculator(potential_path)
    except ImportError:
        # Fallback to MACE if yace not available (unlikely in production env)
        try:
            from mace.calculators import mace_mp  # type: ignore

            return mace_mp(model=potential_path, device="cpu")
        except ImportError:
            msg = "pypacemaker not found"
            raise ImportError(msg) from None  # Use explicit None to satisfy B904


def run_driver() -> None:
    """
    Main driver loop for EON communication.
    Reads atomic configuration from input files (pos.con), calculates E/F,
    and writes to stdout.
    """
    try:
        # 1. Read Arguments / Environment
        # EON usually executes the potential script with arguments or expects reading from files
        # Standard EON potential interface:
        # Execution is managed by client. usually it writes `pos.con` and calls this script.

        if not Path("pos.con").exists():
            # If called without pos.con, maybe just check status?
            sys.exit(0)

        atoms_res = read("pos.con", format="eon")
        # type: ignore
        atoms: Atoms = atoms_res[0] if isinstance(atoms_res, list) else atoms_res

        if not isinstance(atoms, Atoms):
             # Should not happen given logic above
             sys.exit(1)

        # 2. Load Potential
        pot_path = os.environ.get("PACE_POTENTIAL_PATH")
        if not pot_path:
            print("Error: PACE_POTENTIAL_PATH not set", file=sys.stderr)  # noqa: T201
            sys.exit(1)

        calc = get_calculator(pot_path)
        atoms.calc = calc

        # 3. Calculate
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # 4. Check Uncertainty (if supported by calculator)
        # Assuming calculator might return extra info or we use a separate UQ model
        # For now, we simulate or use a threshold on max force as a proxy if needed,
        # but EON expects specific output format.
        # Format:
        # Energy
        # Fx Fy Fz (for each atom)

        # Gamma check (Extrapolation Grade)
        gamma = 0.0
        # If calculator supports gamma (like MACE with variance, or ACE with gamma)
        if hasattr(calc, "get_gamma"):
            gamma = calc.get_gamma(atoms)

        # Check threshold
        threshold_str = os.environ.get("PACE_GAMMA_THRESHOLD")
        halt = False
        if threshold_str:
            threshold = float(threshold_str)
            if gamma > threshold:
                halt = True

        result = {
            "energy": energy,
            "forces": forces.tolist(),
            "gamma": gamma,
            "halt": halt,
        }

        # 5. Output
        if result["halt"]:
            print(f"Halt: Gamma {result['gamma']} > {threshold}", file=sys.stderr)  # noqa: T201
            sys.exit(100)  # EON specific exit code for "bad structure/re-train"

        # Output Energy
        print(f"{result['energy']:.6f}")  # noqa: T201
        # Output Forces
        for force in result["forces"]:
            print(f"{force[0]:.6f} {force[1]:.6f} {force[2]:.6f}")  # noqa: T201

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    run_driver()
