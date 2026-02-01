#!/usr/bin/env python3
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", required=True, help="Path to .yace file")
    parser.add_argument("--elements", nargs="+", required=True, help="Element symbols mapping (index 0 -> Elem 0, etc.)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Gamma threshold")
    parser.add_argument("--zbl-inner", type=float, default=1.0, help="ZBL inner cutoff")
    parser.add_argument("--zbl-outer", type=float, default=2.0, help="ZBL outer cutoff")
    parser.add_argument("--lammps-cmd", default="lmp", help="LAMMPS executable")
    return parser.parse_args()

def read_input() -> Tuple[int, List[List[float]], List[Dict[str, Any]]]:
    """
    Reads EON input format from stdin.
    Format assumed:
    Line 0: N_atoms
    Line 1-3: Box vectors (3x3)
    Line 4-(4+N): TypeIndex X Y Z
    """
    input_str = sys.stdin.read()
    if not input_str.strip():
        sys.stderr.write("Error: Empty input\n")
        sys.exit(1)

    lines = input_str.strip().splitlines()
    try:
        n_atoms = int(lines[0].strip())

        box = []
        for i in range(1, 4):
            box.append([float(x) for x in lines[i].split()])

        atoms_data = []
        for i in range(4, 4 + n_atoms):
            parts = lines[i].split()
            # EON usually sends index. Assuming 0-based index from EON matching --elements list order
            atoms_data.append({
                "type": int(parts[0]),
                "pos": [float(x) for x in parts[1:4]]
            })

        return n_atoms, box, atoms_data
    except Exception as e:
        sys.stderr.write(f"Error parsing input: {e}\nInput was:\n{input_str}\n")
        sys.exit(1)

def write_lammps_data(path: Path, n_atoms: int, box: List[List[float]], atoms_data: List[Dict[str, Any]], elements: List[str]) -> None:
    """Writes data.lammps"""
    # Use ASE for robust structure IO
    try:
        import ase.io
        from ase import Atoms
    except ImportError:
        sys.stderr.write("ASE not found\n")
        sys.exit(1)

    positions = [d["pos"] for d in atoms_data]
    # Map type index to symbols
    # atoms_data['type'] is integer. check bounds.
    symbols = []
    for d in atoms_data:
        idx = d["type"]
        if idx < 0 or idx >= len(elements):
             sys.stderr.write(f"Type index {idx} out of bounds for elements {elements}\n")
             sys.exit(1)
        symbols.append(elements[idx])

    atoms = Atoms(symbols=symbols, positions=positions, cell=box, pbc=True)

    ase.io.write(path, atoms, format="lammps-data", atom_style="atomic") # type: ignore

def run_lammps(args: argparse.Namespace, data_file: Path, work_dir: Path) -> Tuple[Path, Path]:
    """Runs LAMMPS to compute E, F, Gamma"""
    input_file = work_dir / "in.lammps"
    dump_file = work_dir / "dump.lammpstrj"
    log_file = work_dir / "log.lammps"

    # Absolute path for potential
    pot_path = Path(args.potential).resolve()

    # Prepare interaction commands
    # Get Z numbers
    from ase.data import atomic_numbers

    elem_str = " ".join(args.elements)

    pair_cmds = f"""
    pair_style      hybrid/overlay pace zbl {args.zbl_inner} {args.zbl_outer}
    pair_coeff      * * pace {pot_path} {elem_str}
    """

    # ZBL pairs
    for i, el1 in enumerate(args.elements):
        z1 = atomic_numbers[el1]
        for j, el2 in enumerate(args.elements):
            if j < i:
                continue
            z2 = atomic_numbers[el2]
            pair_cmds += f"    pair_coeff      {i+1} {j+1} zbl {z1} {z2}\n"

    input_script = f"""
    units           metal
    atom_style      atomic
    boundary        p p p
    read_data       {data_file}

    {pair_cmds}

    compute         pace_gamma all pace {pot_path}
    variable        max_gamma equal max(c_pace_gamma)

    thermo_style    custom step temp pe v_max_gamma
    thermo          1

    # Dump forces
    dump            1 all custom 1 {dump_file} id fx fy fz
    dump_modify     1 sort id

    run             0
    """

    with open(input_file, "w") as f:
        f.write(input_script)

    cmd = [args.lammps_cmd, "-in", "in.lammps"]

    # Run
    try:
        with open(log_file, "w") as log:
            subprocess.run(cmd, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        sys.stderr.write("LAMMPS execution failed\n")
        # Print log for debug
        if log_file.exists():
            sys.stderr.write(log_file.read_text())
        sys.exit(1)

    return dump_file, log_file

def parse_output(dump_file: Path, log_file: Path) -> Tuple[float, List[List[float]], float]:
    """
    Parses LAMMPS output.
    Returns: Energy, Forces (list of [fx, fy, fz]), MaxGamma
    """
    # Parse Energy and Gamma from log
    energy: Optional[float] = None
    max_gamma: Optional[float] = None

    with open(log_file, "r") as f:
        # Stream lines
        for line in f:
            if "Step Temp PotEng v_max_gamma" in line:
                # Next line has values
                try:
                    val_line = next(f)
                    parts = val_line.split()
                    energy = float(parts[2])
                    max_gamma = float(parts[3])
                except (ValueError, IndexError, StopIteration):
                    pass
                break

    if energy is None or max_gamma is None:
        sys.stderr.write("Failed to parse Energy or Gamma from LAMMPS log\n")
        sys.exit(1)

    # Parse Forces from dump
    forces = []
    # Dump format: id fx fy fz
    # Skip header (9 lines usually)
    with open(dump_file, "r") as f:
        # Skip header
        for _ in range(9):
            next(f)

        data = []
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                uid = int(parts[0])
                fx = float(parts[1])
                fy = float(parts[2])
                fz = float(parts[3])
                data.append((uid, [fx, fy, fz]))

    data.sort(key=lambda x: x[0])
    forces = [x[1] for x in data]

    return energy, forces, max_gamma

def main() -> None:
    args = parse_args()

    n_atoms, box, atoms_data = read_input()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        data_file = work_dir / "data.lammps"

        write_lammps_data(data_file, n_atoms, box, atoms_data, args.elements)

        dump_file, log_file = run_lammps(args, data_file, work_dir)

        energy, forces, max_gamma = parse_output(dump_file, log_file)

        # Check Gamma
        if max_gamma > args.threshold:
            sys.stderr.write(f"High Gamma Detected: {max_gamma} > {args.threshold}\n")
            sys.exit(100)

        # Output result
        print(f"{energy:.16f}")
        for f in forces:
            print(f"{f[0]:.16f} {f[1]:.16f} {f[2]:.16f}")

if __name__ == "__main__":
    main()
