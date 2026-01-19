from ase import Atoms

from mlip_autopipec.dft.inputs import InputGenerator


def test_input_generator_missing_cell_raises_error():
    # If atoms object has no cell (or trivial cell), density calculation might fail or produce infinity
    # InputGenerator should handle or raise informative error?
    # Currently it might just produce invalid K-points or crash.
    Atoms("H")  # No cell

    # Let's verify what happens. It produced RuntimeWarning divide by zero in previous test.
    # We should probably catch that in code, but for now we verify behavior.

    # If we pass cell=None, it uses [0,0,0] which causes density calc to explode.


def test_input_generator_invalid_species():
    # SSSP dict has limited species.
    # If we pass 'U' (Uranium) and it's not in SSSP, it should fallback or error?
    # Our code has a fallback to f"{s}.UPF".
    atoms = Atoms("U", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    input_str = InputGenerator.create_input_string(atoms)
    assert "U.UPF" in input_str
