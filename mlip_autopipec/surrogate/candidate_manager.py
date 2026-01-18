
from ase import Atoms


class CandidateManager:
    """
    Manages a pool of candidate structures, preserving their identity via indices.
    """

    @staticmethod
    def tag_candidates(candidates: list[Atoms]) -> list[Atoms]:
        """
        Tags each atom with its original index in the list.
        This modifies the atoms in-place (or returns the modified list).
        """
        for i, atom in enumerate(candidates):
            if 'info' not in dir(atom):
                atom.info = {}
            atom.info['_original_index'] = i
        return candidates

    @staticmethod
    def resolve_selection(pool: list[Atoms], local_indices: list[int]) -> tuple[list[Atoms], list[int]]:
        """
        Resolves the selected subset and maps back to original indices.

        Args:
            pool: The filtered pool of candidates.
            local_indices: Indices within the filtered pool.

        Returns:
            (selected_structures, original_indices)
        """
        selected_structures = [pool[i] for i in local_indices]
        original_indices = [atom.info.get('_original_index', -1) for atom in selected_structures]
        return selected_structures, original_indices
