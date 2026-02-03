from pathlib import Path

from mlip_autopipec.config.config_model import SelectionConfig
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.physics.selection.selector import ActiveSetSelector


def test_active_set_selector_logic() -> None:
    # 1. Config
    config = SelectionConfig(method="uncertainty", max_structures=2)
    selector = ActiveSetSelector(config)

    # 2. Candidates
    c1 = CandidateStructure(structure_path=Path("s1"), metadata=StructureMetadata(uncertainty=0.1))
    c2 = CandidateStructure(structure_path=Path("s2"), metadata=StructureMetadata(uncertainty=0.9))
    c3 = CandidateStructure(structure_path=Path("s3"), metadata=StructureMetadata(uncertainty=0.5))

    candidates = [c1, c2, c3]

    # 3. Select
    selected = selector.select(candidates, None, Path("work_dir"))

    # 4. Verify
    assert len(selected) == 2
    # Should be sorted by uncertainty descending: c2 (0.9), c3 (0.5)
    assert selected[0].structure_path.name == "s2"
    assert selected[1].structure_path.name == "s3"


def test_active_set_selector_no_candidates() -> None:
    config = SelectionConfig()
    selector = ActiveSetSelector(config)
    selected = selector.select([], None, Path("work_dir"))
    assert selected == []


def test_active_set_selector_less_than_max() -> None:
    config = SelectionConfig(max_structures=10)
    selector = ActiveSetSelector(config)
    c1 = CandidateStructure(structure_path=Path("s1"))
    selected = selector.select([c1], None, Path("work_dir"))
    assert len(selected) == 1
