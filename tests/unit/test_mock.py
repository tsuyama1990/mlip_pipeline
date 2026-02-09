from mlip_autopipec.components.mock import (
    MockGenerator, MockOracle, MockTrainer, MockDynamics, MockValidator, Structure
)
from mlip_autopipec.domain_models.config import BaseConfig
from pathlib import Path

def test_mock_components(tmp_path):
    config = BaseConfig()
    work_dir = tmp_path

    # Generator
    gen = MockGenerator(config, work_dir)
    structs = list(gen.generate(5))
    assert len(structs) == 5
    enhanced = list(gen.enhance(structs[0]))
    assert len(enhanced) == 1

    # Oracle
    oracle = MockOracle(config, work_dir)
    res = oracle.compute(structs[0])
    assert isinstance(res, Structure)
    batch = list(oracle.compute_batch(structs))
    assert len(batch) == 5

    # Trainer
    trainer = MockTrainer(config, work_dir)
    pot = trainer.train(batch)
    assert pot == "mock_potential.yace"

    # Dynamics
    dyn = MockDynamics(config, work_dir)
    metrics = dyn.explore(pot, structs[0])
    assert metrics == {"halted": False}

    # Validator
    val = MockValidator(config, work_dir)
    metrics = val.validate(pot)
    assert metrics == {"passed": True}
