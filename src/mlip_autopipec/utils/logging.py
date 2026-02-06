import logging
import sys

def setup_logging(log_level: str = "INFO") -> None:
    """
    Sets up the global logging configuration.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # force=True reconfigures logging even if already configured
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
