from pathlib import Path

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.validation import ElasticConfig
from mlip_autopipec.validation.elasticity import ElasticityValidator


def test_elastic_validator_security(tmp_path):
    config = ElasticConfig(command="calc; rm -rf /")  # Unsafe
    validator = ElasticityValidator(config, tmp_path)

    with pytest.raises(ValueError):
        validator.validate(Atoms("Al"), Path("pot.yace"))


def test_elastic_validator_run(tmp_path):
    config = ElasticConfig(command="calc")
    validator = ElasticityValidator(config, tmp_path)

    res = validator.validate(Atoms("Al"), Path("pot.yace"))
    assert res.metric == "C11"
