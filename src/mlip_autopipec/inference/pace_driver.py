import os
import sys

from ase import Atoms


def read_geometry(input_stream) -> Atoms:
    """Reads geometry from EON client format (stdin)."""
    lines = input_stream.readlines()
    if not lines:
        raise ValueError("Empty input")

    try:
        # Line 1: Number of atoms
        n_atoms = int(lines[0].strip())

        # Line 2: Box vectors (9 floats)
        box_line = lines[1].strip().split()
        if len(box_line) != 9:
             raise ValueError(f"Expected 9 box components, got {len(box_line)}")

        box = [float(x) for x in box_line]
        cell = [box[0:3], box[3:6], box[6:9]]

        symbols = []
        positions = []

        for i in range(n_atoms):
            line = lines[2+i].strip().split()
            symbols.append(line[0])
            positions.append([float(x) for x in line[1:4]])

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        return atoms
    except Exception as e:
        raise ValueError(f"Failed to parse EON geometry: {e}")

def print_results(atoms: Atoms, output_stream):
    """Prints energy and forces in EON client format."""
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    output_stream.write(f"{energy:.16e}\n")
    for f in forces:
        output_stream.write(f"{f[0]:.16e} {f[1]:.16e} {f[2]:.16e}\n")

def main():
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
        _ = atoms.get_potential_energy()

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
