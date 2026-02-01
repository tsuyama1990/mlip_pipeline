from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.embedding import EmbeddingHandler


class CandidateManager:
    """
    Manages the processing of candidate structures.
    Delegates to EmbeddingHandler for physics logic.
    """

    def __init__(self) -> None:
        self.embedding_handler = EmbeddingHandler()

    def extract_cluster(self, supercell: Structure, center_atom_index: int, radius: float) -> Structure:
        """
        Extracts a periodic box around the center atom.
        Wraps `extract_periodic_box` with box_length = 2 * radius.
        """
        return self.extract_periodic_box(supercell, center_atom_index, box_length=2.0 * radius)

    def extract_periodic_box(self, supercell: Structure, center_index: int, box_length: float) -> Structure:
        """
        Extracts a cubic periodic box centered at a specific atom.
        Delegates to EmbeddingHandler.
        """
        # Audit: Bounds checking for center_index
        n_atoms = len(supercell.positions)
        if center_index < 0 or center_index >= n_atoms:
            raise ValueError(f"center_index {center_index} is out of bounds for structure with {n_atoms} atoms.")

        return self.embedding_handler.extract_periodic_box(supercell, center_index, box_length)

    def embed_cluster(self, cluster: Structure, vacuum: float = 10.0) -> Structure:
        """
        Ensures the structure is periodic.
        Now a pass-through since extraction creates a periodic box.
        Kept for compatibility with CalculationPhase.
        """
        if not all(cluster.pbc):
            # Fallback for legacy non-periodic clusters
            atoms = cluster.to_ase()
            atoms.center(vacuum=vacuum)  # type: ignore[no-untyped-call]
            atoms.set_pbc((True, True, True))  # type: ignore[no-untyped-call]
            return Structure.from_ase(atoms)
        return cluster
