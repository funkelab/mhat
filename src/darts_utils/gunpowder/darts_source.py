import random
from warnings import warn

import gunpowder as gp
import numpy as np
from funlib.geometry.coordinate import Coordinate
from funlib.geometry.roi import Roi

from darts_utils.data import DartsZarr


class DartsSource(gp.BatchProvider):
    """A gunpowder node for providing Darts data. Must specify channel, can specify FOV
    or have it randomly selected (or iterated over).
    Within the fov and channel, data is stored t, y, x
    """

    def __init__(
        self,
        darts_zarr: DartsZarr,
        channels: dict[gp.ArrayKey, str],  # array keys to channels
        array_specs: dict[gp.ArrayKey, gp.ArraySpec] | None = None,
        fov: int | None = None,
    ):
        super().__init__()
        self.darts_zarr = darts_zarr
        valid_channels = self.darts_zarr.get_channels(
            random.choice(self.darts_zarr.get_fovs())
        )
        for channel in channels.values():
            if channel not in valid_channels:
                raise ValueError(
                    f"Channel {channel} not in darts zarr {darts_zarr.dataset_name}"
                    f"channels ({valid_channels})"
                )

        self.channels = channels
        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs
        self.fov = fov
        self.ndims = 3

    def setup(self):
        for key, channel in self.channels.items():
            print(key, channel)
            spec = self.__read_spec(key, channel)
            print(spec)
            self.provides(key, spec)

    def provide(self, request):
        batch = gp.Batch()
        for key, request_spec in request.array_specs.items():
            voxel_size = self.spec[key].voxel_size
            dataset_roi = request_spec.roi / voxel_size
            # dataset_roi = dataset_roi - self.spec[key].roi.offset / voxel_size
            array_spec = self.spec[key].copy()
            array_spec.roi = request_spec.roi
            if self.fov:
                fov = self.fov
            else:
                fov = random.choice(self.darts_zarr.get_fovs())
                print(f"Randomly chose fov {fov}")
            ds = self.darts_zarr.get_data(channel=self.channels[key], fov=fov)
            batch.arrays[key] = gp.Array(
                self.__read(ds, dataset_roi),
                array_spec,
            )
        return batch

    def __read(self, dataset, roi):
        # t c y x -> channels neither first nor last
        c = len(dataset.shape) - self.ndims
        slices = roi.to_slices()
        slices = (slices[0], slice(None), *slices[1:])
        array = np.asarray(dataset[slices]).squeeze(1)  # remove channel dimension

        return array

    def __read_spec(self, array_key, channel):
        if self.fov:
            fov = self.fov
        else:
            fov = random.choice(self.darts_zarr.get_fovs())
            print(f"Randomly chose fov {fov} when reading spec")
            # TODO: deal with fovs of different sizes
        dataset = self.darts_zarr.get_data(channel=channel, fov=fov)
        print(dataset.shape)

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = gp.ArraySpec()

        if spec.voxel_size is None:
            voxel_size = Coordinate((1,) * self.ndims)
            spec.voxel_size = voxel_size

        if spec.roi is None:
            offset = Coordinate((0,) * self.ndims)
            shape = Coordinate(dataset.shape)  # t c y x
            shape = Coordinate(shape[0], *shape[2:])
            print(shape)
            spec.roi = Roi(offset, shape * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, (
                "dtype %s provided in array_specs for %s, "
                "but differs from dataset %s dtype %s"
                % (self.array_specs[array_key].dtype, array_key, channel, dataset.dtype)
            )
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float32,
                np.float64,
                # np.float128,
                np.uint8,  # assuming this is not used for labels
            ]
            warn(
                "WARNING: You didn't set 'interpolatable' for %s "
                "(dataset %s). Based on the dtype %s, it has been "
                "set to %s. This might not be what you want."
                % (
                    array_key,
                    channel,
                    spec.dtype,
                    spec.interpolatable,
                )
            )

        return spec
