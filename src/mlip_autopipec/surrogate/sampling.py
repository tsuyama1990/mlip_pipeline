import numpy as np
from scipy.spatial.distance import cdist  # type: ignore


class FarthestPointSampling:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def select(self, descriptors: np.ndarray) -> list[int]:
        """
        Selects indices using FPS.

        Args:
            descriptors: (N, D) array.

        Returns:
            List of selected indices.
        """
        N = descriptors.shape[0]
        if N == 0 or self.n_samples == 0:
            return []
        if self.n_samples >= N:
            return list(range(N))

        # Initialize with index 0
        # The Spec suggests picking the one with lowest energy, but here we just take the first one
        # (assuming the caller sorts them if needed).
        selected_indices = [0]

        # Initial distances from point 0 to all other points
        # cdist returns (1, N) matrix, flatten to (N,)
        min_dists = cdist(descriptors[0:1], descriptors, metric="euclidean").flatten()

        for _ in range(1, self.n_samples):
            # Find point with maximum minimum distance to the current set
            farthest_idx = np.argmax(min_dists)
            selected_indices.append(int(farthest_idx))

            # Update min_dists with distances from the new selected point
            new_dists = cdist(
                descriptors[farthest_idx : farthest_idx + 1], descriptors, metric="euclidean"
            ).flatten()
            min_dists = np.minimum(min_dists, new_dists)

        return selected_indices
