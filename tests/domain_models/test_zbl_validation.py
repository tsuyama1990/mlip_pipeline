import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.config import PotentialConfig

def test_zbl_cutoffs_valid():
    config = PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        zbl_inner_cutoff=1.0,
        zbl_outer_cutoff=2.0
    )
    assert config.zbl_inner_cutoff == 1.0
    assert config.zbl_outer_cutoff == 2.0

def test_zbl_cutoffs_invalid_order():
    with pytest.raises(ValidationError) as exc:
        PotentialConfig(
            elements=["Si"],
            cutoff=5.0,
            pair_style="hybrid/overlay",
            zbl_inner_cutoff=2.0,
            zbl_outer_cutoff=1.0
        )
    assert "ZBL outer cutoff must be > inner cutoff" in str(exc.value)

def test_zbl_cutoffs_invalid_negative():
    with pytest.raises(ValidationError) as exc:
        PotentialConfig(
            elements=["Si"],
            cutoff=5.0,
            pair_style="hybrid/overlay",
            zbl_inner_cutoff=-0.5,
            zbl_outer_cutoff=1.0
        )
    assert "ZBL inner cutoff must be > 0" in str(exc.value)

def test_zbl_cutoffs_ignored_for_pace():
    # Should not raise validation error if pair_style is pace, even if cutoffs look wrong
    # (Though defaults are valid).
    config = PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="pace",
        zbl_inner_cutoff=5.0,
        zbl_outer_cutoff=1.0
    )
    # Validator checks if pair_style == hybrid/overlay.
    assert config.pair_style == "pace"
