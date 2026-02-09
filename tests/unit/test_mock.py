from mlip_autopipec.components.mock import (
    MockGenerator, MockOracle, MockTrainer, MockDynamics, MockValidator, MockStructure
)
from mlip_autopipec.domain_models.config import BaseConfig
from mlip_autopipec.domain_models.datastructures import InMemoryDataset
import pytest

def test_mock_components(tmp_path):
    config = BaseConfig()
    work_dir = tmp_path

    # Generator
    gen = MockGenerator(config, work_dir)
    structs = list(gen.generate(5))
    assert len(structs) == 5
    assert isinstance(structs[0], MockStructure)
    assert structs[0].atoms == 0

    enhanced = list(gen.enhance(structs[0]))
    assert len(enhanced) == 1
    assert isinstance(enhanced[0], MockStructure)

    # Oracle
    oracle = MockOracle(config, work_dir)
    res = oracle.compute(structs[0])
    assert isinstance(res, MockStructure)
    assert res == structs[0]

    batch = list(oracle.compute_batch(iter(structs)))
    assert len(batch) == 5
    assert batch == structs

    # Trainer
    trainer = MockTrainer(config, work_dir)
    dataset = InMemoryDataset(batch)
    pot = trainer.train(dataset)
    assert pot == "mock_potential.yace"

    # Dynamics
    dyn = MockDynamics(config, work_dir)
    metrics = dyn.explore(pot, structs[0])
    assert metrics == {"halted": False}

    # Validator
    val = MockValidator(config, work_dir)
    metrics = val.validate(pot)
    assert metrics == {"passed": True}

def test_mock_components_type_validation(tmp_path):
    config = BaseConfig()
    work_dir = tmp_path

    # Generator
    gen = MockGenerator(config, work_dir)
    with pytest.raises(TypeError):
        list(gen.enhance("invalid"))

    # Oracle
    oracle = MockOracle(config, work_dir)
    with pytest.raises(TypeError):
        oracle.compute("invalid")

    # Trainer
    trainer = MockTrainer(config, work_dir)
    with pytest.raises(TypeError):
        trainer.train("invalid_dataset")

    # Dynamics
    dyn = MockDynamics(config, work_dir)
    with pytest.raises(TypeError):
        dyn.explore("pot", "invalid_structure")
