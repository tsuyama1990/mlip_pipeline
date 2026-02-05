from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.config.config_model import GlobalConfig, ExplorationConfig, DFTConfig, TrainingConfig
from mlip_autopipec.domain_models.structures import StructureMetadata

def test_mock_explorer() -> None:
    explorer = MockExplorer()
    config = ExplorationConfig(strategy="random")
    candidates = explorer.generate_candidates(config, 2)
    assert len(candidates) == 2
    assert isinstance(candidates[0], StructureMetadata)
    assert candidates[0].source == "mock_explorer"

def test_mock_oracle() -> None:
    oracle = MockOracle()
    config = DFTConfig()
    # Create dummy input
    from ase import Atoms
    meta = StructureMetadata(structure=Atoms('H'), source="test", generation_method="test")
    results = oracle.calculate([meta], config)
    assert len(results) == 1
    assert results[0].structure.calc is not None # type: ignore

def test_mock_trainer() -> None:
    trainer = MockTrainer()
    config = TrainingConfig()
    path = trainer.train([], config)
    assert path.endswith(".yace")

def test_mock_validator() -> None:
    validator = MockValidator()
    config = GlobalConfig()
    result = validator.validate("dummy.yace", config)
    assert result.passed is True
    assert len(result.metrics) > 0
