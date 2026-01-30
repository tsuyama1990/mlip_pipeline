from mlip_autopipec.domain_models.calculation import DFTConfig, DFTError, SCFError


class RecoveryHandler:
    def apply_fix(self, config: DFTConfig, error: DFTError, attempt: int) -> DFTConfig:
        """
        Returns a modified configuration based on the error.
        Raises DFTError if no fix is available or max retries exceeded.
        """
        if isinstance(error, SCFError):
            if attempt == 1:
                # Fix 1: Reduce mixing beta
                return config.model_copy(update={"mixing_beta": 0.3})
            elif attempt == 2:
                # Fix 2: Increase smearing and change type
                # Ensure degauss is at least increased
                new_degauss = max(0.05, config.degauss * 2.0)
                return config.model_copy(
                    update={"smearing": "mv", "degauss": new_degauss}
                )
            elif attempt == 3:
                # Fix 3: Very soft mixing
                return config.model_copy(update={"mixing_beta": 0.1})

        # If we don't know how to fix it, re-raise
        raise error
