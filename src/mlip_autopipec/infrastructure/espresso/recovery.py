from collections.abc import Generator
from typing import Any


class RecoveryStrategy:
    """
    Defines a sequence of parameter overrides (recipes) to try when a calculation fails.
    """

    def get_recipes(self) -> Generator[dict[str, Any], None, None]:
        """
        Yields a sequence of parameter dictionaries to merge with the base configuration.
        """
        # 1. First attempt: Use original parameters (no overrides)
        yield {}

        # 2. Second attempt: Reduce mixing beta (0.7 -> 0.3)
        yield {"mixing_beta": 0.3}

        # 3. Third attempt: Increase smearing
        # Assuming cold smearing is default or similar
        yield {"smearing": "methfessel-paxton", "sigma": 0.02}

        # 4. Fourth attempt: Combine both and be more aggressive
        yield {"mixing_beta": 0.1, "smearing": "methfessel-paxton", "sigma": 0.05}
