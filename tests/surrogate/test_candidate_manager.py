from ase import Atoms

from mlip_autopipec.surrogate.candidate_manager import CandidateManager


def test_tag_candidates():
    atoms1 = Atoms("H")
    atoms2 = Atoms("He")
    candidates = [atoms1, atoms2]

    tagged = CandidateManager.tag_candidates(candidates)

    assert tagged[0].info["_original_index"] == 0
    assert tagged[1].info["_original_index"] == 1
    assert tagged is candidates  # In-place modification check


def test_resolve_selection():
    atoms1 = Atoms("H", info={"_original_index": 0})
    atoms2 = Atoms("He", info={"_original_index": 1})
    atoms3 = Atoms("Li", info={"_original_index": 2})

    pool = [atoms1, atoms3]  # index 1 filtered out
    local_indices = [1, 0]  # Select Li then H

    selected, original_indices = CandidateManager.resolve_selection(pool, local_indices)

    assert len(selected) == 2
    assert selected[0].symbols == "Li"
    assert selected[1].symbols == "H"

    assert original_indices == [2, 0]
