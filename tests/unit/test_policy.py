import pytest
from unittest.mock import MagicMock

from mlip_autopipec.domain_models.config import Config, PolicyConfig
from mlip_autopipec.domain_models.exploration import ExplorationTask
# The module does not exist yet, so this import will fail if I run pytest now.
# But in TDD I write the test first.
from mlip_autopipec.physics.structure_gen.policy import AdaptivePolicy

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.policy = MagicMock(spec=PolicyConfig)
    config.policy.is_metal = False
    config.md = MagicMock()
    config.md.temperature = 300.0
    config.md.n_steps = 1000
    return config

def test_policy_cycle_0_cold_start(mock_config):
    """Cycle 0 should return a Cold Start strategy (e.g. Random Strain)."""
    policy = AdaptivePolicy()
    task = policy.decide(cycle=0, config=mock_config)

    assert isinstance(task, ExplorationTask)
    # Strategy says RandomStrain + M3GNet. Since we don't have M3GNet, maybe just Static/Strain?
    # Or maybe we stick to what existing ExplorationPhase does if policy is disabled?
    # But here we test the policy itself.
    # Let's assume it returns Static with "strain" modifier.
    assert task.method == "Static"
    assert "strain" in task.modifiers

def test_policy_metal_cycle_gt_0(mock_config):
    """Metals should use HybridMDMC (MD + Swap)."""
    mock_config.policy.is_metal = True
    policy = AdaptivePolicy()
    task = policy.decide(cycle=1, config=mock_config)

    assert task.method == "MD"
    assert "swap" in task.modifiers

def test_policy_insulator_cycle_gt_0(mock_config):
    """Insulators should use DefectSampling."""
    mock_config.policy.is_metal = False
    policy = AdaptivePolicy()
    task = policy.decide(cycle=1, config=mock_config)

    # SPEC says "If Insulator: Return DefectSampling"
    assert task.method == "Static"
    assert "defect" in task.modifiers
