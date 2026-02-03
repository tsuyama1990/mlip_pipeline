import numpy as np
from ase.build import bulk

from mlip_autopipec.physics.structure_gen.embedding import extract_periodic_box

prim = bulk("Cu", "fcc", a=3.6, cubic=True)
supercell = prim * (3, 3, 3) # 3x3x3 cubic cells

cutoff = 1.8
box = extract_periodic_box(supercell, 0, cutoff)
print(f"Box atoms: {len(box)}")
print(f"Box volume: {box.get_volume()}")
print(f"Expected: {box.get_volume() / (prim.get_volume()/len(prim))}")
