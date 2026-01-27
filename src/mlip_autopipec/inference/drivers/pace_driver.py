#!/usr/bin/env python3
import io
import os
import sys
from typing import Any

from ase.io import read


def get_potential_calculator(potential_path: str) -> Any:
    try:
        from pypacemaker import Calculator
        return Calculator(potential_path)
    except ImportError:
        msg = "pypacemaker not found"
        raise ImportError(msg)

def process_structure(atoms: Any, calculator: Any, threshold: float | None = None) -> dict[str, Any]:
    """
    Calculates energy, forces and uncertainty (gamma).
    Returns dict with results and halt flag.
    """
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    gamma = 0.0

    # Extract gamma from calculator results
    # Adjust based on exact pypacemaker API
    results = getattr(calculator, "results", {})
    if "gamma" in results:
         gamma = results["gamma"]

    halt = False
    if threshold is not None and gamma > float(threshold):
        halt = True

    return {
        "energy": energy,
        "forces": forces,
        "gamma": gamma,
        "halt": halt
    }

def main() -> None:
    # Read environment variables
    pot_path = os.environ.get("PACE_POTENTIAL_PATH")
    if not pot_path:
        sys.exit(1)

    threshold_str = os.environ.get("PACE_GAMMA_THRESHOLD")
    threshold = float(threshold_str) if threshold_str else 5.0

    try:
        # Read from stdin
        # We assume EON format which ASE can read (format='eon')
        # or we try to detect.
        input_data = sys.stdin.read()
        f = io.StringIO(input_data)

        # Try 'eon' format first, then auto
        try:
            atoms = read(f, format='eon')
        except Exception:
            f.seek(0)
            atoms = read(f)

        calc = get_potential_calculator(pot_path)

        result = process_structure(atoms, calc, threshold)

        if result['halt']:
            sys.exit(100)

        # Output Energy
        # Output Forces
        for _force in result['forces']:
            pass

    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
