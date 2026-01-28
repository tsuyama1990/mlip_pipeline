from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.validation import ElasticConfig
from mlip_autopipec.validation.elasticity import ElasticityValidator


def test_elastic_validator_run(tmp_path):
    config = ElasticConfig()
    validator = ElasticityValidator(config, tmp_path)

    res = validator.validate(Atoms("Al"), Path("pot.yace"))
    assert res.module == "elastic"
    assert len(res.metrics) > 0
    assert res.metrics[0].name == "C11"
