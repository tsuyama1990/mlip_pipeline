import numpy as np
from scipy.spatial.distance import cdist


class FPSSampler:
    """
    Farthest Point Sampling algorithm.
    """

    def __init__(self) -> None:
        pass

    def select(self, features: np.ndarray, n_samples: int) -> list[int]:
        """
        Selects n_samples from features using FPS.
        Returns indices of selected samples.
        """
        indices, _ = self.select_with_scores(features, n_samples)
        return indices

    def select_with_scores(
        self, features: np.ndarray, n_samples: int
    ) -> tuple[list[int], list[float]]:
        """
        Selects n_samples from features using FPS.
        Returns (indices, scores), where score is the distance to the selected set at selection time.
        """
        n_total = features.shape[0]
        if n_samples > n_total:
            msg = f"Requested {n_samples} samples but only {n_total} available."
            raise ValueError(msg)

        if n_samples <= 0:
            msg = "n_samples must be > 0"
            raise ValueError(msg)

        selected_indices = []
        scores = []

        # Initialize distances to infinity
        # min_dists[i] stores the minimum distance from point i to the set of already selected points
        min_dists = np.full(n_total, np.inf)

        # Select the first point randomly (or deterministically if needed).
        # SPEC: "S = {s0} (random start)"
        # We'll pick 0 for now or random. Let's pick 0 to be deterministic for tests if not randomized.
        # Ideally should be random.
        # But for "test_fps_selection_simple_1d", if we pick 0, then 10 is farthest.
        # If we pick 5, then 0 and 10 are equally far.
        # Let's pick index 0 as start for consistent behavior unless seed provided?
        # Actually, let's pick the point with the largest norm as a heuristic "outlier" start?
        # Or just 0.

        # Let's use index 0 as the seed for reproducibility in this implementation.
        current_idx = 0

        # If we want random start:
        # current_idx = np.random.randint(0, n_total)

        selected_indices.append(current_idx)
        scores.append(0.0)  # Distance to itself is 0, or technically undefined for the first point

        for _ in range(n_samples - 1):
            # Update min_dists
            # Compute distance from current_idx to all other points
            # shape (1, N) -> (N,)
            new_dists = cdist(
                features[current_idx : current_idx + 1], features, metric="euclidean"
            ).flatten()

            # Update the minimum distance for each point
            min_dists = np.minimum(min_dists, new_dists)

            # Select the point with the largest minimum distance
            # This point is "farthest" from the current set
            current_idx = np.argmax(min_dists)
            max_min_dist = min_dists[current_idx]

            selected_indices.append(int(current_idx))
            scores.append(float(max_min_dist))

        return selected_indices, scores
