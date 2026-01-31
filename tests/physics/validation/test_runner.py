from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from mlip_autopipec.physics.validation.runner import ValidationRunner
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric
import numpy as np

def test_validation_runner():
    # Setup
    val_config = ValidationConfig()
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0, pair_style="hybrid/overlay")
    runner = ValidationRunner(val_config, pot_config, Path("pot.yace"))

    structure = Structure(
        symbols=["Si"],
        positions=np.zeros((1,3)),
        cell=np.eye(3),
        pbc=(True,True,True)
    )

    # Mock validators
    with patch.object(runner.eos_validator, 'validate', return_value=ValidationMetric(name="EOS", value=0.1, passed=True)), \
         patch.object(runner.elastic_validator, 'validate', return_value=ValidationMetric(name="Elastic", value=0.1, passed=True)), \
         patch.object(runner.phonon_validator, 'validate', return_value=(ValidationMetric(name="Phonon", value=0.1, passed=True), None)), \
         patch.object(runner.report_generator, 'generate'):

         result = runner.validate(structure)

         assert result.overall_status == "PASS"
         assert len(result.metrics) == 3
         runner.report_generator.generate.assert_called_once()
