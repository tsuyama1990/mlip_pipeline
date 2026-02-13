"""
This module defines project-wide constants.
"""

# Potential Formats
POTENTIAL_FORMAT_YACE: str = "yace"
"""Format string for ACE potentials (.yace)."""

# Provenance Tags
PROVENANCE_MD_HALT: str = "md_halt"
"""Provenance tag for structures halted during MD simulation due to high uncertainty."""
PROVENANCE_RANDOM: str = "random"
"""Provenance tag for randomly generated structures."""
PROVENANCE_M3GNET: str = "m3gnet"
"""Provenance tag for structures generated via M3GNet pre-exploration."""

# Generator Modes
MODE_SEED: str = "seed"
"""Generator mode for producing seed structures for MD."""

# Context Keys
KEY_CYCLE: str = "cycle"
"""Key for cycle number in context dictionary."""
KEY_TEMPERATURE: str = "temperature"
"""Key for temperature in context dictionary."""
