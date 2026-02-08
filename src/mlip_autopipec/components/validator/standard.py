from typing import Any

from mlip_autopipec.components.validator.base import BaseValidator
from mlip_autopipec.domain_models.config import StandardValidatorConfig
from mlip_autopipec.domain_models.potential import Potential


class StandardValidator(BaseValidator):
    def __init__(self, config: StandardValidatorConfig) -> None:
        super().__init__(config)
        self.config: StandardValidatorConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def validate(self, potential: Potential) -> dict[str, Any]:
        msg = "Standard Validator (Phonon/EOS/Elastic) is not yet implemented."
        raise NotImplementedError(msg)
