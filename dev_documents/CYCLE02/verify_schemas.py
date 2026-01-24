from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig

gc = GeneratorConfig()
print("GeneratorConfig default:", gc.model_dump())

sc = SystemConfig()
print("SystemConfig default generator:", sc.generator_config.model_dump())

assert sc.generator_config == gc
print("Verification successful.")
