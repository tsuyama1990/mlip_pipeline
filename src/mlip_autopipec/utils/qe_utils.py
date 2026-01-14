from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets

SSSP_PSEUDOPOTENTIALS = {
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
}


def get_sssp_recommendations(atoms: Atoms) -> dict[str, str]:
    """
    Returns the SSSP pseudopotential for each element in the Atoms object.
    """
    symbols = set(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
    return {symbol: SSSP_PSEUDOPOTENTIALS[symbol] for symbol in symbols}


def get_kpoints(atoms: Atoms, kpoint_density: float = 6.0) -> tuple[int, int, int]:
    """
    Calculates the k-points grid for a given Atoms object.
    """
    kpts, _ = kpts2sizeandoffsets(atoms=atoms, density=kpoint_density)  # type: ignore[no-untyped-call]
    return tuple(int(k) for k in kpts)
