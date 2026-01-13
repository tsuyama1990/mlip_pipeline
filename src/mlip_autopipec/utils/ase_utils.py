from ase import Atoms


def example_ase_util(atoms: Atoms) -> Atoms:
    """
    An example utility function for ASE Atoms objects.
    """
    atoms.info['example_key'] = 'example_value'
    return atoms
