#!/usr/bin/env python3
import sys
import os
import io
import numpy as np
from ase.io import read
from pathlib import Path
from typing import Any

def get_potential_calculator(potential_path: str) -> Any:
    try:
        from pypacemaker import Calculator
        return Calculator(potential_path)
    except ImportError:
        raise ImportError("pypacemaker not found")

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
        print("Error: PACE_POTENTIAL_PATH not set", file=sys.stderr)
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
            print(f"Halt: Gamma {result['gamma']} > {threshold}", file=sys.stderr)
            sys.exit(100)

        # Output Energy
        print(f"{result['energy']:.6f}")
        # Output Forces
        for force in result['forces']:
            print(f"{force[0]:.6f} {force[1]:.6f} {force[2]:.6f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
