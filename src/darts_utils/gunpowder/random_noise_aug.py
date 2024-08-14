import random

import numpy as np
import skimage
from gunpowder.batch_request import BatchRequest
from gunpowder.nodes import BatchFilter


class NoiseAugment(BatchFilter):
    """Add a random amount of gaussian noise to an array. The variance of the
    noise gaussian will be uniformly sampled between the provided min and max.
    Uses the scikit-image function skimage.util.random_noise.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within
            range [-1, 1] or [0, 1].

        noise_min (``float``):

            The minimum variance of noise to add

        noise_max(``float``):

            The maximum variance of noise to add

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by
            clipping values in the end, see scikit-image documentation
    """

    def __init__(self, array, noise_min=0.0, noise_max=0.2, clip=True):
        self.array = array
        self.mode = "gaussian"
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.clip = clip

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        if raw.data.dtype not in [np.float32, np.float64]:
            raise ValueError(
                "Noise augmentation requires float types for the raw array (not "
                + str(raw.data.dtype)
                + "). Consider using Normalize before."
            )
        if self.clip and (raw.data.min() >= -1 or raw.data.max() <= 1):
            raise ValueError(
                "Noise augmentation expects raw values in [-1,1]"
                " or [0,1]. Consider using Normalize before."
            )

        seed = request.random_seed

        try:
            raw.data = skimage.util.random_noise(
                raw.data,
                mode=self.mode,
                rng=seed,
                clip=self.clip,
                var=random.uniform(self.noise_min, self.noise_max),
            ).astype(raw.data.dtype)

        except ValueError:
            # legacy version of skimage random_noise
            raw.data = skimage.util.random_noise(
                raw.data, mode=self.mode, seed=seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)
