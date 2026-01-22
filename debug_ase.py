from ase import Atoms
import numpy as np

try:
    atoms = Atoms('H2')
    # try direct assignment
    atoms.positions = np.array([[0,0,0]])
    print(f"Direct assignment worked. Shape: {atoms.positions.shape}, Len: {len(atoms)}")
except Exception as e:
    print(f"Direct assignment failed: {e}")

try:
    atoms = Atoms('H2')
    # try array hacking
    atoms.arrays['positions'] = np.array([[0,0,0]])
    print(f"Array hacking worked. Shape: {atoms.positions.shape}, Len: {len(atoms)}")
except Exception as e:
    print(f"Array hacking failed: {e}")
