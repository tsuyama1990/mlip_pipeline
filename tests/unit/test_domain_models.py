from pathlib import Path

from ase import Atoms

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Dataset, Structure
from mlip_autopipec.domain_models.validation import ValidationResult


def test_structure_model() -> None:
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    structure = Structure(atoms=atoms, metadata={"source": "test"})
    assert structure.atoms == atoms
    assert structure.metadata["source"] == "test"

def test_dataset_model() -> None:
    atoms1 = Atoms('H')
    atoms2 = Atoms('He')
    s1 = Structure(atoms=atoms1)
    s2 = Structure(atoms=atoms2)
    dataset = Dataset(structures=[s1, s2], name="test_dataset")

    assert len(dataset) == 2
    assert len(dataset.to_atoms_list()) == 2
    assert dataset.structures[0].atoms == atoms1

def test_potential_model(tmp_path: Path) -> None:
    pot_path = tmp_path / "model.yace"
    pot = Potential(path=pot_path)
    assert pot.path == pot_path
    assert pot.type == "yace"

def test_validation_result_model() -> None:
    res = ValidationResult(passed=True, metrics={"rmse": 0.01})
    assert res.passed
    assert res.metrics["rmse"] == 0.01
    assert "passed=True" in str(res)
