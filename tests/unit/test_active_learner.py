"""Unit tests for Active Learner."""

import pytest

from pyacemaker.core.config import Step2ActiveLearningConfig
from pyacemaker.domain_models.models import StructureMetadata, UncertaintyState

# Import from expected location
try:
    from pyacemaker.modules.active_learner import ActiveLearner
except ImportError:
    ActiveLearner = None


@pytest.mark.skipif(ActiveLearner is None, reason="ActiveLearner not implemented")
def test_active_learner_select_batch() -> None:
    """Test selection logic of ActiveLearner."""
    config = Step2ActiveLearningConfig(n_select=2, uncertainty_threshold=0.5)
    learner = ActiveLearner(config)

    from uuid import uuid4
    # Create dummy candidates with varying uncertainty
    # s1: low (0.1), s2: high (0.9), s3: medium (0.6), s4: low (0.0)
    s1 = StructureMetadata(id=uuid4(), tags=["s1"], uncertainty_state=UncertaintyState(gamma_max=0.1))
    s2 = StructureMetadata(id=uuid4(), tags=["s2"], uncertainty_state=UncertaintyState(gamma_max=0.9))
    s3 = StructureMetadata(id=uuid4(), tags=["s3"], uncertainty_state=UncertaintyState(gamma_max=0.6))
    s4 = StructureMetadata(id=uuid4(), tags=["s4"], uncertainty_state=UncertaintyState(gamma_max=0.0))

    candidates = [s1, s2, s3, s4]

    # Select batch
    selected = list(learner.select_batch(candidates))

    # Expect 2 selected (n_select=2)
    assert len(selected) == 2
    # Expect s2 (0.9) and s3 (0.6) as they are the highest
    ids = {s.id for s in selected}
    assert s2.id in ids
    assert s3.id in ids

    # Verify order (descending uncertainty)
    assert selected[0].id == s2.id
    assert selected[1].id == s3.id
