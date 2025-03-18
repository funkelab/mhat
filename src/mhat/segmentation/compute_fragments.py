import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.ndimage.filters import maximum_filter
from skimage.segmentation import watershed


def compute_fragments(
    affs: np.ndarray,
    aff_threshold: float = 0.5,
    id_offset: int = 0,
    min_seed_distance: int = 3,
) -> tuple[np.ndarray, int]:
    """Get fragments from seed points using the by creating a fg/bg
    mask from the affinities. A point is foreground if the mean
    of the y and x affinities is greater than aff_threshold.
    Then computes the distance transform, finds local maxima for seed points,
    and uses watershed to get the final fragments.

    Args:
        affs (np.ndarray): An array of affinities with shape (3, [z], y, x).
            Only channels 1, 2 are used, the first channel is ignored
        aff_threshold (float, optional): Determines the fg/bg mask - if the
            mean of affs[1] and affs[2] is greater than this threshold, it is
            fg. Defaults to 0.5.
        id_offset (int, optional): Offset all non-zero labels by this amount.
            Useful for ensuring unique IDs across time points. Defaults to 0.
        min_seed_distance (int, optional): The minimum distance between seeds
            used for watershed. Defaults to 3.

    Returns:
        np.ndarray, int: An array of extracted fragment labels with shape
            ([z], y, x), and the max label id in the fragments
    """
    mean_affs = 0.5 * (affs[1] + affs[2])

    boundary_mask = mean_affs > aff_threshold
    boundary_distances = distance_transform_edt(boundary_mask)

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=boundary_mask
    )

    return fragments.astype(np.uint64), n + id_offset
