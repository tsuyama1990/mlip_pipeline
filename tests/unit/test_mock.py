from mlip_autopipec.components.mock import (
    MockGenerator, MockOracle, MockTrainer, MockDynamics, MockValidator, MockStructure
)
from mlip_autopipec.domain_models.config import BaseConfig
from mlip_autopipec.domain_models.datastructures import StreamingDataset
import pytest

class InMemoryDataset(StreamingDataset):
    """Mock dataset for testing that behaves like StreamingDataset but uses memory/temp file."""
    def __init__(self, structures=None, work_dir=None):
        self.work_dir = work_dir
        self.filepath = work_dir / "test.jsonl"
        super().__init__(self.filepath, mode="w")
        if structures:
            for s in structures:
                self.append(s)

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
    # Equality check for Structure might need __eq__ but identity/repr check is okay for mock
    assert res.data == structs[0].data

    batch = list(oracle.compute_batch(iter(structs)))
    assert len(batch) == 5

    # Trainer
    trainer = MockTrainer(config, work_dir)
    dataset = InMemoryDataset(batch, work_dir=tmp_path)
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

    # Trainer
    trainer = MockTrainer(config, work_dir)
    with pytest.raises(TypeError):
        trainer.train("invalid_dataset")
