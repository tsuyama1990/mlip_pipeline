from unittest.mock import MagicMock, patch
from mlip_autopipec.domain_models.config import Config, LammpsConfig, PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.orchestration.workflow import run_one_shot

def test_run_one_shot_success():
    config = MagicMock(spec=Config)
    config.lammps = LammpsConfig(command="lmp", cores=1, timeout=10)
    config.potential = PotentialConfig(elements=["Si"], cutoff=3.0)
    # Mocking exploration as it might be accessed
    config.exploration = MagicMock()
    config.exploration.lattice_constant = 5.43
    config.exploration.rattle_amplitude = 0.0
    config.exploration.composition = "Si"
    config.exploration.md_params = None

    with patch("mlip_autopipec.orchestration.workflow.StructureBuilder") as MockBuilder, \
         patch("mlip_autopipec.orchestration.workflow.LammpsRunner") as MockRunner:

         MockBuilder.return_value.build_bulk.return_value = MagicMock()
         MockRunner.return_value.run.return_value.status = JobStatus.COMPLETED

         result = run_one_shot(config)
         assert result.status == JobStatus.COMPLETED
