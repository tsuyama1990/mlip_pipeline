import argparse
import logging
import sys
from pathlib import Path

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
    setup_logging = None # type: ignore
    MLIPCalculatorFactory = None # type: ignore

logger = logging.getLogger("potential_server")

def _raise_error(msg: str) -> None:
    raise ValueError(msg)

def _parse_header(lines: list[str]) -> tuple[int, int]:
    if not lines:
        _raise_error("Empty input")

    header = lines[0].strip()
    if not header:
         _raise_error("Empty header")
    n_atoms = int(header)

    if n_atoms < 0:
        _raise_error(f"Negative atom count: {n_atoms}")

    current_line = 1
    if current_line < len(lines):
        parts = lines[current_line].split()
        if len(parts) == 1:
            current_line += 1

    return n_atoms, current_line

def _parse_box(lines: list[str], start_line: int) -> tuple[list[list[float]], int]:
    cell = []
    current = start_line
    for _ in range(3):
        if current >= len(lines):
            _raise_error("Unexpected end of input while parsing cell.")
        parts = lines[current].split()
        if len(parts) != 3:
            _raise_error(f"Invalid cell vector format at line {current+1}")
        cell.append([float(x) for x in parts])
        current += 1
    return cell, current

def _parse_positions(lines: list[str], start_line: int, n_atoms: int) -> tuple[list[list[float]], int]:
    positions = []
    current = start_line
    for i in range(n_atoms):
        if current >= len(lines):
            _raise_error(f"Unexpected end of input while parsing positions. Expected {n_atoms}, got {i}")
        parts = lines[current].split()
        if len(parts) != 3:
            _raise_error(f"Invalid position format at line {current+1}")
        positions.append([float(x) for x in parts])
        current += 1
    return positions, current

def parse_eon_input(input_str: str, symbols: list[str]) -> Atoms:
    """
    Parses EON client input format.
    """
    if not input_str.strip():
        _raise_error("Empty input received from EON.")

    lines = input_str.strip().split('\n')

    try:
        n_atoms, current = _parse_header(lines)
        cell, current = _parse_box(lines, current)
        positions, current = _parse_positions(lines, current, n_atoms)

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

    # Should be unreachable
    return Atoms()

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
    except Exception:
        logger.debug("Failed to read pos.con, falling back to dummy symbols.")

    if isinstance(atoms, Atoms):
         return atoms.get_chemical_symbols() # type: ignore[no-any-return, no-untyped-call]

    return ["H"] * 100

def process_structure(atoms: Atoms, calculator, threshold: float = 5.0) -> tuple[float, np.ndarray, float | None]:
    atoms.calc = calculator
    try:
        # Standard ASE methods trigger calculation
        energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
        forces = atoms.get_forces() # type: ignore[no-untyped-call]

        # Uncertainty check
        gamma = None
        results = atoms.calc.results # type: ignore
        for key in ['uncertainty', 'gamma', 'max_gamma', 'c_pace_gamma']:
            if key in results:
                gamma = results[key]
                if hasattr(gamma, "__len__"):
                    gamma = np.max(gamma)
                break

        if gamma is not None and gamma > threshold:
            logger.warning("High uncertainty detected: %s > %s", gamma, threshold)
            write("bad_structure.xyz", atoms) # type: ignore[no-untyped-call]
            with Path("halt_info.txt").open("w") as f:
                f.write(f"reason: uncertainty\nmax_gamma: {gamma}\n")
            sys.exit(100)

        return float(energy), forces, gamma # type: ignore

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

    if setup_logging is not None: # type: ignore[truthy-function]
        setup_logging(Path(), "potential_server.log")

    try:
        input_str = sys.stdin.read()
        if not input_str:
            return

        symbols = load_symbols()

        try:
            atoms = parse_eon_input(input_str, symbols)
        except Exception:
            lines = input_str.strip().split()
            if lines and lines[0].isdigit():
                n_atoms = int(lines[0])
                atoms = parse_eon_input(input_str, ["H"]*n_atoms)
            else:
                raise

        if MLIPCalculatorFactory is not None: # type: ignore[truthy-function]
            factory = MLIPCalculatorFactory()
            calc = factory.create(Path(args.potential))
        else:
            from ase.calculators.emt import EMT  # type: ignore
            calc = EMT()

        energy, forces, gamma = process_structure(atoms, calc, args.threshold)
        sys.stdout.write(format_eon_output(energy, forces, gamma) + '\n')

    except Exception:
        logger.exception("Server Error")
        sys.exit(1)

if __name__ == "__main__":
    main()
