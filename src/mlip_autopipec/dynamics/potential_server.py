import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import numpy as np
from ase import Atoms
from ase.io import read, write

# Import from package - assumed installed
try:
    from mlip_autopipec.core.logger import setup_logging
    from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory
except ImportError:
    # Fallback for standalone testing without package installed
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("mlip_autopipec package not found. Using standalone fallback.")
    setup_logging = None # type: ignore[assignment]
    MLIPCalculatorFactory = None # type: ignore[assignment, misc]

logger = logging.getLogger("potential_server")

def _raise_error(msg: str) -> None:
    raise ValueError(msg)

def _read_input_lines(stream: TextIO) -> Iterator[str]:
    """Reads input stream line by line."""
    for line in stream:
        yield line.strip()

def _parse_header(stream: Iterator[str]) -> int:
    try:
        line = next(stream)
    except StopIteration:
         _raise_error("Empty input received from EON.")

    if not line:
         _raise_error("Empty header")
    n_atoms = int(line)

    if n_atoms < 0:
        _raise_error(f"Negative atom count: {n_atoms}")

    return n_atoms

def _parse_energy(stream: Iterator[str]) -> str:
    # Check if next line is energy (1 float) or Box (3 floats)
    try:
        line = next(stream)
    except StopIteration:
         # Just N provided?
         raise StopIteration from None

    parts = line.split()
    if len(parts) == 1:
        # It was energy, consume and return next line for Box A
        try:
            line = next(stream)
        except StopIteration:
            _raise_error("Unexpected end of input after energy")

    return line

def _parse_cell(stream: Iterator[str], first_line: str) -> list[list[float]]:
    cell = []

    # First vector is already read
    parts = first_line.split()
    if len(parts) != 3:
         _raise_error(f"Invalid cell vector format: {first_line}")
    cell.append([float(x) for x in parts])

    # Next 2 vectors
    for _ in range(2):
        try:
            line = next(stream)
            parts = line.split()
            if len(parts) != 3:
                _raise_error(f"Invalid cell vector format: {line}")
            cell.append([float(x) for x in parts])
        except StopIteration:
            _raise_error("Unexpected end of input while parsing cell")

    return cell

def _parse_positions(stream: Iterator[str], n_atoms: int) -> list[list[float]]:
    positions = []
    for i in range(n_atoms):
        try:
            line = next(stream)
            parts = line.split()
            if len(parts) != 3:
                _raise_error(f"Invalid position format at atom {i}: {line}")
            positions.append([float(x) for x in parts])
        except StopIteration:
            _raise_error(f"Unexpected end of input while parsing positions. Expected {n_atoms}, got {i}")
    return positions

def parse_eon_input(stream: Iterator[str], symbols: list[str]) -> Atoms:
    """
    Parses EON client input format from an iterator of lines.
    """
    try:
        n_atoms = _parse_header(stream)

        if n_atoms == 0:
             # Try peek next line to ensure stream consumption logic consistency or return immediately
             # But we need to handle energy line if present? Usually empty system doesn't have much.
             return Atoms(pbc=True)

        try:
            first_box_line = _parse_energy(stream)
        except StopIteration:
             if n_atoms == 0:
                 return Atoms(pbc=True) # Redundant safe guard
             _raise_error("Unexpected end of input after header")

        cell = _parse_cell(stream, first_box_line)
        positions = _parse_positions(stream, n_atoms)

        # Validation
        if len(symbols) != n_atoms:
             if len(symbols) == 0 and n_atoms > 0:
                  symbols = ["H"] * n_atoms
             elif len(symbols) < n_atoms:
                  logger.warning("Symbol count mismatch: %d vs %d. Extending.", len(symbols), n_atoms)
                  symbols.extend([symbols[-1]] * (n_atoms - len(symbols)))
             else:
                  symbols = symbols[:n_atoms]

        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    except ValueError as e:
        logger.exception("Parsing Error")
        _raise_error(f"Invalid EON input format: {e}")
    except Exception as e:
        logger.exception("Unexpected parsing error")
        _raise_error(f"Parsing failed: {e}")

    return Atoms() # unreachable

def format_eon_output(energy: float, forces: np.ndarray, gamma: float | None = None) -> str:
    lines = [f"{energy:.6f}"]
    for f in forces:
        lines.append(f"{f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
    return "\n".join(lines)

def load_symbols() -> list[str]:
    atoms = None
    try:
        if Path("pos.con").exists():
            atoms = read("pos.con", format="eon")
    except Exception as e:
        logger.debug("Failed to read pos.con: %s", e)

    if isinstance(atoms, Atoms):
         return atoms.get_chemical_symbols() # type: ignore[no-any-return, no-untyped-call]

    return ["H"] * 100

def process_structure(atoms: Atoms, calculator: object, threshold: float = 5.0) -> tuple[float, np.ndarray, float | None]:
    atoms.calc = calculator
    try:
        # Standard ASE methods trigger calculation
        energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
        forces = atoms.get_forces() # type: ignore[no-untyped-call]

        # Uncertainty check
        gamma = None
        results = atoms.calc.results
        for key in ['uncertainty', 'gamma', 'max_gamma', 'c_pace_gamma']:
            if key in results:
                gamma = results[key]
                if hasattr(gamma, "__len__"):
                    gamma = np.max(gamma)
                break

        if gamma is not None and gamma > threshold:
            logger.warning("High uncertainty detected: %s > %s", gamma, threshold)
            write("bad_structure.xyz", atoms)
            with Path("halt_info.txt").open("w") as f:
                f.write(f"reason: uncertainty\nmax_gamma: {gamma}\n")
            sys.exit(100)

        return float(energy), forces, gamma

    except SystemExit:
        raise
    except Exception:
        logger.exception("Calculation failed")
        raise

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", required=True, help="Path to potential file")
    parser.add_argument("--threshold", type=float, default=5.0, help="Uncertainty threshold")
    args = parser.parse_args()

    if setup_logging is not None:
        setup_logging(Path(), "potential_server.log")

    try:
        # Use generator to read stdin lazily
        input_stream = (line.strip() for line in sys.stdin)

        symbols = load_symbols()

        atoms = parse_eon_input(input_stream, symbols)

        if MLIPCalculatorFactory is not None:
            factory = MLIPCalculatorFactory()
            calc = factory.create(Path(args.potential))
        else:
            from ase.calculators.emt import EMT
            calc = EMT()

        energy, forces, gamma = process_structure(atoms, calc, args.threshold)
        sys.stdout.write(format_eon_output(energy, forces, gamma) + '\n')

    except Exception:
        logger.exception("Server Error")
        sys.exit(1)

if __name__ == "__main__":
    main()
