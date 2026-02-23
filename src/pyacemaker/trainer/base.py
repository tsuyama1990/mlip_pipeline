"""Base Trainer implementation."""

from abc import ABC

from loguru import logger

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import Trainer


class BaseTrainer(Trainer, ABC):
    """Base class for all trainers."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the BaseTrainer."""
        super().__init__(config)
        self.trainer_config = config.trainer
        self.logger = logger.bind(name=self.__class__.__name__)
