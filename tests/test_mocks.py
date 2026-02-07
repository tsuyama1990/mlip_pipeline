from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.infrastructure import MockDynamics, MockGenerator, MockOracle, MockTrainer


def test_mock_generator():
    gen = MockGenerator()
    structures = list(gen.generate(2))
    assert len(structures) == 2
    assert isinstance(structures[0], Structure)

def test_mock_oracle(valid_structure):
    oracle = MockOracle()
    labeled = list(oracle.compute([valid_structure]))
    assert len(labeled) == 1
    assert labeled[0].energy is not None
    assert labeled[0].forces is not None

def test_mock_trainer(valid_structure, tmp_path):
    trainer = MockTrainer()
    potential = trainer.train([valid_structure], tmp_path)
    assert isinstance(potential, Potential)
    assert potential.path.exists()

def test_mock_dynamics(tmp_path):
    dyn = MockDynamics()
    pot = Potential(path=tmp_path / "dummy.yace")
    structures = list(dyn.explore(pot))
    assert len(structures) > 0
    assert isinstance(structures[0], Structure)
