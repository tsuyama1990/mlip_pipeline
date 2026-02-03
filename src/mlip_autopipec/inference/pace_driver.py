import os
import sys
from typing import TextIO

from ase import Atoms


def read_geometry(input_stream: TextIO) -> Atoms:
    """Reads geometry from EON client format (stdin)."""
    lines = input_stream.readlines()
    if not lines:
        msg = "Empty input"
        raise ValueError(msg)

    try:
        # Line 1: Number of atoms
        n_atoms = int(lines[0].strip())

        # Line 2: Box vectors (9 floats)
        box_line = lines[1].strip().split()
        if len(box_line) != 9:
             _raise_box_error(len(box_line))

        box = [float(x) for x in box_line]
        cell = [box[0:3], box[3:6], box[6:9]]

        symbols = []
        positions = []

        for i in range(n_atoms):
            line = lines[2+i].strip().split()
            symbols.append(line[0])
            positions.append([float(x) for x in line[1:4]])

        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    except Exception as e:
        msg = f"Failed to parse EON geometry: {e}"
        raise ValueError(msg) from e

def _raise_box_error(count: int) -> None:
    msg = f"Expected 9 box components, got {count}"
    raise ValueError(msg)


def print_results(atoms: Atoms, output_stream) -> None:  # type: ignore[no-untyped-def]
    """Prints energy and forces in EON client format."""
    energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
    forces = atoms.get_forces()  # type: ignore[no-untyped-call]

    output_stream.write(f"{energy:.16e}\n")
    for f in forces:
        output_stream.write(f"{f[0]:.16e} {f[1]:.16e} {f[2]:.16e}\n")


def main() -> None:
    try:
        from pyacemaker.calculator import PaceCalculator
    except ImportError:
        sys.stderr.write("Error: pyacemaker not found\n")
        sys.exit(1)

    potential_path = "potential.yace"

    try:
        atoms = read_geometry(sys.stdin)

        calc = PaceCalculator(potential_path)
        atoms.calc = calc

        # Trigger calculation
        _ = atoms.get_potential_energy()  # type: ignore[no-untyped-call]

        # Check extrapolation grade / gamma if available
        # Threshold from environment or default 10.0
        threshold = float(os.environ.get("MLIP_GAMMA_THRESHOLD", "10.0"))

        if hasattr(calc, "results") and "c_gamma_val" in calc.results:
            gamma = calc.results["c_gamma_val"]
            # If gamma is list/array, take max
            if hasattr(gamma, "__iter__"):
                gamma = max(gamma)

            if gamma > threshold:
                sys.stderr.write(f"Halted due to gamma {gamma} > {threshold}\n")
                sys.exit(100)

        print_results(atoms, sys.stdout)

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
