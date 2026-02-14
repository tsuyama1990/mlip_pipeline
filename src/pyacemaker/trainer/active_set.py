"""Active Set Selection Logic."""

from pathlib import Path

from pyacemaker.trainer.wrapper import PacemakerWrapper


class ActiveSetSelector:
    """Selects optimal structures for active learning."""

    def __init__(self, wrapper: PacemakerWrapper | None = None) -> None:
        self.wrapper = wrapper or PacemakerWrapper()

    def select(self, candidates_path: Path, num_select: int) -> Path:
        """Select N structures from candidates dataset.

        Args:
            candidates_path: Path to the full candidates dataset (.pckl.gzip)
            num_select: Number of structures to select

        Returns:
            Path to the new dataset containing only selected structures.

        """
        # Determine output path based on input path
        # e.g., candidates.pckl.gzip -> candidates_selected.pckl.gzip
        # Ensure we handle multiple extensions correctly
        name = candidates_path.name
        if name.endswith(".pckl.gzip"):
            stem = name[:-10]  # Remove .pckl.gzip
        elif name.endswith((".gzip", ".pckl")):
            stem = name[:-5]
        else:
            stem = candidates_path.stem

        output_path = candidates_path.parent / f"{stem}_selected.pckl.gzip"

        return self.wrapper.select_active_set(candidates_path, num_select, output_path)
