import sys
from typing import IO, cast

from ase import Atoms
from ase.io import read

# We use conditional import to avoid ImportErrors during dev/testing if pyacemaker is missing
try:
    from pyacemaker.calculator import PaceCalculator
except ImportError:
    PaceCalculator = None  # type: ignore

THRESHOLD = 5.0 # Max extrapolation grade

def read_eon_geometry(stream: IO[str]) -> Atoms:
    """
    Reads geometry from EON client format (stdin).
    Format:
    N
    ax ay az bx by bz cx cy cz
    Type X Y Z
    ...
    """
    lines = stream.readlines()
    if not lines:
        msg = "Empty input"
        raise ValueError(msg)

    try:
        n_atoms = int(lines[0].strip())

        # Line 2: Box Matrix (9 floats)
        box_line = lines[1].strip()
        box_parts = [float(x) for x in box_line.split()]
        if len(box_parts) != 9:
             # Try falling back to ASE read if format is not as expected
             # But here we assume strict EON format
             msg = "Expected 9 floats for box matrix"
             raise ValueError(msg)  # noqa: TRY301

        cell = [box_parts[0:3], box_parts[3:6], box_parts[6:9]]

        symbols = []
        positions = []
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])

        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    except Exception:
        # If parsing fails, rewind (if possible) or try passing raw content?
        # Stream might be consumed.
        # Since we read lines, we can try to make a new stream or just fallback to ASE read on string.
        # Let's try ASE read on the content string as fallback (e.g. for extxyz)
        stream.seek(0)
        res = read(stream, format="extxyz")
        if isinstance(res, list):
            return res[-1] # type: ignore
        return cast(Atoms, res)

def run_driver() -> int:
    """
    Main driver logic for EON interface.
    Reads geometry from stdin, calculates forces/energy using PaceCalculator,
    and prints to stdout.
    Returns exit code: 0 for success, 100 for halt (high uncertainty).
    """
    try:
         # Read all stdin content first to handle non-seekable streams
         content = sys.stdin.read()
         if not content:
             # It might be that sys.stdin.read() returned empty because it was already consumed?
             # No, we start here.
             msg = "Empty input"
             raise ValueError(msg)  # noqa: TRY301

         from io import StringIO
         s = StringIO(content)

         try:
             atoms = read_eon_geometry(s)
         except Exception:
             s.seek(0)
             try:
                res = read(s, format="extxyz")
                if isinstance(res, list):
                    atoms = res[-1] # type: ignore
                else:
                    atoms = cast(Atoms, res)
             except Exception:
                s.seek(0)
                res = read(s) # Auto detect
                if isinstance(res, list):
                    atoms = res[-1] # type: ignore
                else:
                    atoms = cast(Atoms, res)

    except Exception as e:
        sys.stderr.write(f"Error reading atoms: {e}\n")
        return 1

    if PaceCalculator is None:
        sys.stderr.write("pyacemaker not installed.\n")
        return 1

    # Calculate
    # We assume the potential file is named 'potential.yace' in the current directory
    try:
        calc = PaceCalculator("potential.yace")
        results = calc.calculate(atoms)
    except Exception as e:
        sys.stderr.write(f"Calculation failed: {e}\n")
        return 1

    # Check Uncertainty
    # We assume 'gamma' is available in results
    gamma_val = getattr(results, "gamma", 0.0)

    if gamma_val > THRESHOLD:
        # Signal Halt
        sys.stderr.write(f"Halt: Max gamma {gamma_val} > {THRESHOLD}\n")
        return 100

    # Print Energy/Forces to stdout
    # EON Expects:
    # Energy
    # Fx Fy Fz
    # ...

    energy = getattr(results, "energy", 0.0)
    forces = getattr(results, "forces", [])

    sys.stdout.write(f"{energy:.16e}\n")
    for f in forces:
        sys.stdout.write(f"{f[0]:.16e} {f[1]:.16e} {f[2]:.16e}\n")

    return 0

if __name__ == "__main__":
    sys.exit(run_driver())
