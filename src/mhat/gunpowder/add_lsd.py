import logging

import numpy as np
from gunpowder import Array, Batch, BatchFilter, BatchRequest, Roi
from lsd.train import LsdExtractor

logger = logging.getLogger(__name__)


class AddLocalShapeDescriptor(BatchFilter):
    """Create a local segmentation shape discriptor to each voxel.
    Takes in a 2D + time input and computes 2D LSDs for the middle
    time point.

    Args:

        segmentation (:class:`ArrayKey`): The array storing the segmentation
            to use.

        descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            generate.

        lsds_mask (:class:`ArrayKey`, optional): The array to store a binary mask
            the size of the descriptors. Background voxels, which do not have a
            descriptor, will be set to 0. This can be used as a loss scale
            during training, such that background is ignored.

        labels_mask (:class:`ArrayKey`, optional): The array to use as a mask
            for labels. Lsds connecting at least one masked out label will be
            masked out in lsds_mask.

        unlabelled (:class:`ArrayKey`, optional): A binary array to indicate
            unlabelled areas with 0. Lsds from labelled to unlabelled voxels are set
            to 0, lsds between unlabelled voxels are masked out (they will not be
            used for training).

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel or the radius of the sphere.

        mode (string): Either ``gaussian`` or ``sphere``. Specifies how to
            accumulate local statistics: ``gaussian`` uses Gaussian convolution
            to compute a weighed average of statistics inside an object.
            ``sphere`` accumulates values in a sphere.

        components (string, optional): The components of the local shape descriptors to
            compute and return. Should be a string of integers chosen from 0 through 9
            (if 3D) or 6 (if 2D), in order. Example: "0129" or "345".
            Defaults to all components.

            Component string lookup, where example component : "3D axes", "2D axes"

                mean offset (mean) : "012", "01"
                orthogonal covariance (ortho) : "345", "23"
                diagonal covariance (diag) : "678", "4"
                size : "9", "5"

            Example combinations:

                diag + size : "6789", "45"
                mean + diag + size : "0126789", "0145"
                mean + ortho + diag : "012345678", "01234"
                ortho + diag : "345678", "234"

        downsample (int, optional): Downsample the segmentation mask to extract
            the statistics with the given factore. Default is 1 (no
            downsampling).
    """

    def __init__(
        self,
        segmentation,
        descriptor,
        lsds_mask=None,
        labels_mask=None,
        unlabelled=None,
        sigma=5.0,
        mode="gaussian",
        components=None,
        downsample=1,
    ):
        self.segmentation = segmentation
        self.descriptor = descriptor
        self.lsds_mask = lsds_mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled
        self.components = components

        if isinstance(sigma, (tuple, list)):
            self.sigma = tuple(sigma)
        else:
            self.sigma = (sigma,) * 3

        self.mode = mode
        self.downsample = downsample
        self.voxel_size = None
        self.context = None
        self.skip = False
        # the LSD extractor should be two dimensional (ignore time)
        self.extractor = LsdExtractor(self.sigma[1:], self.mode, self.downsample)

    def setup(self):
        spec = self.spec[self.segmentation].copy()

        spec.dtype = np.float32

        self.voxel_size = spec.voxel_size
        self.provides(self.descriptor, spec)

        if self.lsds_mask:
            self.provides(self.lsds_mask, spec.copy())

        if self.mode == "gaussian":
            self.context = tuple(s * 3 for s in self.sigma)
        elif self.mode == "sphere":
            self.context = tuple(self.sigma)
        else:
            raise RuntimeError(f"Unknown mode {self.mode}")

    def prepare(self, request):
        deps = BatchRequest()
        if self.descriptor in request:
            dims = len(request[self.descriptor].roi.get_shape())

            if dims == 2:
                self.context = self.context[0:2]

            # increase segmentation ROI to fit Gaussian (in y and x)
            context_roi = request[self.descriptor].roi.grow(
                self.context,
                self.context,
            )

            # ensure context roi is multiple of voxel size
            context_roi = context_roi.snap_to_grid(self.voxel_size, mode="shrink")

            grown_roi = request[self.segmentation].roi.union(context_roi)

            deps[self.segmentation] = request[self.descriptor].copy()
            deps[self.segmentation].roi = grown_roi

        else:
            self.skip = True

        if self.unlabelled:
            deps[self.unlabelled] = deps[self.segmentation].copy()

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.segmentation].copy()

        return deps

    def process(self, batch, request):
        if self.skip:
            return

        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = request[self.descriptor].roi.copy()
        segmentation_array = batch[self.segmentation]
        # get voxel roi of requested descriptors
        # this is the only region in
        # which we have to compute the descriptors
        seg_roi = segmentation_array.spec.roi  # with the context (-60, -60) offset
        offset = seg_roi.get_offset()
        seg_roi = seg_roi - offset  # now starts at (0, 0)
        descriptor_roi = request[self.descriptor].roi  # no context (0, 0) offset
        descriptor_roi = descriptor_roi - offset  # now starts at (60, 60)
        voxel_roi_in_seg = (
            seg_roi.intersect(descriptor_roi) - seg_roi.get_offset()
        ) / self.voxel_size

        crop = voxel_roi_in_seg.get_bounding_box()

        time_roi = descriptor_spec.roi.grow((0, None, None), (0, None, None))

        intersection_roi = time_roi.intersect(segmentation_array.spec.roi)
        seg_data = segmentation_array.crop(intersection_roi).data

        descriptors = []
        for seg_slice in seg_data:
            slice_roi = Roi(voxel_roi_in_seg.offset[1:], voxel_roi_in_seg.shape[1:])

            descriptor_slice = self.extractor.get_descriptors(
                segmentation=seg_slice,
                components=self.components,
                voxel_size=(1, 1),
                roi=slice_roi,
            )
            # add back in time dimension
            descriptors.append(descriptor_slice)

        descriptor = np.stack(descriptors, axis=1)
        # create descriptor array
        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = request[self.descriptor].roi.copy()
        descriptor_array = Array(descriptor, descriptor_spec)

        old_batch = batch
        seg_roi = seg_roi + offset  # now starts at (-60, -60)
        descriptor_roi = descriptor_roi + offset  # back to (0, 0)

        # Create new batch for descriptor:
        batch = Batch()

        # create lsds mask array
        if self.lsds_mask and self.lsds_mask in request:
            if self.labels_mask:
                mask = self._create_mask(old_batch, self.labels_mask, descriptor, crop)

            else:
                mask = (segmentation_array.crop(descriptor_roi).data != 0).astype(
                    np.float32
                )

                mask_shape = len(mask.shape)

                assert mask.shape[-mask_shape:] == descriptor.shape[-mask_shape:]

                mask = np.array([mask] * descriptor.shape[0])

            if self.unlabelled:
                unlabelled_mask = self._create_mask(
                    old_batch, self.unlabelled, descriptor, crop
                )

                mask = mask * unlabelled_mask

            batch[self.lsds_mask] = Array(
                mask.astype(descriptor.dtype), descriptor_spec.copy()
            )

        batch[self.descriptor] = descriptor_array

        return batch

    def _create_mask(self, batch, mask, lsds, crop):
        mask = batch.arrays[mask].data

        mask = np.array([mask] * lsds.shape[0])

        mask = mask[(slice(None), *crop)]

        return mask
