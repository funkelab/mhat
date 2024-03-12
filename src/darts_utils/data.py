import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zarr


class DartsZarr:
    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        zarr_name: str,
        mode: str = "r",
        store_type="nested",
        zarr_base_group: str = "",
    ):
        """A wrapper class for zarrs that maintains DARTS data structure, with fovs and
        channels stored in separate groups. Can be used to read existing data or
        create a new DARTS structured zarr depending on the mode.

        Args:
            base_path (str | Path): Path to directory containing all datasets
            dataset_name (str): Dataset name, used to access correct datset directory
            zarr_name (str): Name of zarr within the dataset dir we are interested in.
            mode (str, optional): Zarr modes. Defaults to "r".
            store_type (str, optional): Type of zarr store to use. Defaults to "nested".
                Will be removed when all data is converted from flat to nested
                in the near future.
            zarr_base_group (str, optional): The path within the zarr to the "base"
                group where the fovs are located. Useful for storing multiple
                experiments in the same zarr.

        Raises:
            ValueError: If base_path or dataset dir within base path don't exist.
            Will raise other zarr errors if other sub-groups don't exist and mode isn't
            appropriate.
        """
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.mode = mode
        self.zarr_name = zarr_name
        self.zarr_base_group = zarr_base_group
        self.store_type = store_type

        self.dataset_path = self.base_path / dataset_name
        self.zarr_path = self.dataset_path / self.zarr_name

        # verify passed in values
        if not self.base_path.exists():
            raise ValueError(f"Base path {base_path} does not exist")
        if not self.zarr_path.exists() and self.mode.startswith("r"):
            raise ValueError(
                f"Zarr at {self.zarr_path} does not exist and mode is {mode}"
            )
        # create (if necessary) and store root as attribute
        if self.store_type == "nested":
            self.store = zarr.NestedDirectoryStore(self.zarr_path)
        else:
            self.store = zarr.DirectoryStore(self.zarr_path)

    def get_fovs(self) -> list[int]:
        """Get a list of all fovs present in the zarr.

        Returns:
            list[int]: A list of all fovs present in zarr.
        """
        root = zarr.open_group(self.store, mode=self.mode, path=self.zarr_base_group)
        return [DartsZarr._fov_from_key(k) for k in root.group_keys()]

    def get_channels(self, fov: int) -> list[str]:
        """Get a list of all channels for that fov in the zarr

        Args:
            fov (int): Field of view to get channels for

        Returns:
            list[str]: List of all channel names present in the zarr for that fov.
        """
        fov_group_path = self._get_fov_group_path(fov)
        fov_group = zarr.open_group(self.store, mode=self.mode, path=fov_group_path)
        return [DartsZarr._channel_from_key(k) for k in fov_group.array_keys()]

    def get_data(
        self,
        fov: int,
        channel: str,
        shape: Optional[int | tuple[int, ...]] = None,
        dtype: Optional[str | np.dtype] = None,
    ) -> zarr.array:
        """Get the zarr array for the given fov and channel, with behavior determined
        by the mode of the DartsZarr (e.g. overwrite if 'w', fail if not present
        if 'r').

        Args:
            fov (int): Field of view to get data for
            channel (str): Channel to get data for
            shape (int | tuple[int, ...], optional): array shape passed to
                zarr.open_array(). Defaults to None.
            dtype(str | zarr.dtype, optional): dtype passed to zarr.open_array().
                Defaults to None.


        Returns:
            zarr.array: A zarr array for the data at the given fov and channel.
                As usual, it is lazy so no data is loaded in this function.
        """
        data_group_path = self._get_data_group_path(fov, channel)
        return zarr.open_array(
            self.store, mode=self.mode, path=data_group_path, shape=shape, dtype=dtype
        )

    def _get_fov_group_path(self, fov: int) -> str:
        """Get the internal zarr path to the fov group. Prepends the base group
        to the fov string if necessary.

        Args:
            fov (int): Field of view to get group path of

        Returns:
            str: Path within zarr to fov group
        """
        fov_key = DartsZarr._fov_key(fov)
        if self.zarr_base_group:
            return self.zarr_base_group + "/" + fov_key
        else:
            return fov_key

    def _get_data_group_path(self, fov: int, channel: str) -> str:
        """Get the internal zarr path to the channel array of the given fov.

        Args:
            fov (int): Field of view to use in path
            channel (str): Channel name to use in path

        Returns:
            str: Path within zarr to channel array of the given fov
        """
        channel_key = DartsZarr._channel_key(channel)
        return self._get_fov_group_path(fov) + "/" + channel_key

    @staticmethod
    def _fov_key(fov: int) -> str:
        """Static method to turn an int fov into a string key in DARTS format.

        Args:
            fov (int): Field of view

        Returns:
            str: Key used in DARTS format for the field of view, e.g. `fov=1`.
        """
        return f"fov={fov}"

    @staticmethod
    def _fov_from_key(fov_key: str) -> int:
        """Static method to extract the fov int from the fov_key string.
        Args:
            fov_key (str): Field of view key in DARTS format, e.g. `fov=1`

        Returns:
            int: Fov int extracted from fov key
        """
        return int(fov_key.split("=")[1])

    @staticmethod
    def _channel_key(channel: str) -> str:
        """Static method to get the DARTS formatted channel_key string from the
        channel name.
        For example:
            `YFP` -> `channel=YFP`

        Args:
            channel (str): Channel name

        Returns:
            str: Channel key in DARTS format `channel=<name>`
        """
        return f"channel={channel}"

    @staticmethod
    def _channel_from_key(channel_key: str) -> str:
        """Static method to extract the channel name from the channel_key string.
        For example:
            `channel=YFP` -> `YFP`
        Args:
            channel_key (str): Channel key in format `channel=<name>`

        Returns:
            str: Channel name extracted from channel_key.
        """
        return channel_key.split("=")[1]


class RawDataZarr(DartsZarr):
    """A DARTS zarr that assumes the zarr name is "raw.zarr" and the base group is
    empty
    """

    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        mode: str = "r",
        store_type: str = "nested",
    ):
        super().__init__(
            base_path,
            dataset_name,
            "raw.zarr",
            mode=mode,
            zarr_base_group="",
            store_type=store_type,
        )


class SegmentationZarr(DartsZarr):
    """A DARTS zarr that assumes the zarr name is "segmentation.zarr" and the base
    group corresponds to a result name that must be passed to constructor (but can
    be changed later).
    """

    def __init__(
        self,
        base_path: str | Path,
        dataset_name: str,
        result_name: str,
        mode="r",
    ):
        super().__init__(
            base_path,
            dataset_name,
            "segmentation.zarr",
            mode=mode,
            zarr_base_group=result_name,
        )

    def get_result_names(self) -> list[str]:
        """Get a list of results inside the segmentation zarr.

        Returns:
            list[str]: All groups directly inside the segmentation zarr root,
                which are assumed to be names of experimental results.
        """
        root = zarr.open_group(self.store, mode=self.mode)
        return list(root.group_keys())

    def set_result_name(self, result_name: str):
        """Set the experimental result name to use for future operations

        Args:
            result_name (str): Experiment name to use when querying zarr
        """
        self.zarr_base_group = result_name

    def has_segmentation(
        self,
        fov: Optional[int] = None,
        channel: Optional[str] = None,
    ) -> bool:
        """Check if the zarr has a segmentation for the currently set experimental
        result name. Can specify fov and channel or check first fov and/or all channels.
        Existance of an array is considered having a segmentation - it does not check
        if the array is empty.

        Args:
            fov (Optional[int], optional): _description_. Defaults to None.
            channel (Optional[str], optional): _description_. Defaults to None.

        Returns:
            bool: True if there is an array at the given fov or the first fov
                and the given channel or any channel, otherwise False.
        """
        if not fov:
            # pick first one available
            fovs = self.get_fovs()
            if len(fovs) == 0:
                return False
            fov = fovs[0]

        if not channel:
            # check for all channels
            channels = self.get_channels(fov)
        else:
            channels = [channel]

        for channel in channels:
            seg_group_path = self._get_data_group_path(fov, channel)
            print(seg_group_path)
            if zarr.storage.contains_array(self.store, seg_group_path):
                return True
        return False


def add_data_args(
    parser: argparse.ArgumentParser, base_path_default="/Volumes/funke/data/darts"
):
    group = parser.add_argument_group("Data arguments")
    group.add_argument("-d", "--dataset_name", help="Dataset: name")
    group.add_argument(
        "-f",
        "--fov",
        type=int,
        help="Dataset: field of view (currently only supports one)",
    )
    group.add_argument(
        "-c",
        "--channels",
        type=str,
        nargs="+",
        help="Dataset: Channels to view. See spreadsheet 'Channel names' column "
        "for options",
    )
    group.add_argument("-dbp", "--data_base_path", default=base_path_default)


def add_segmentation_args(
    parser: argparse.ArgumentParser,
    seg_path_default="/Volumes/funke/projects/darts/experiments/segmentation",
):
    group = parser.add_argument_group(
        "Segmentation zarr arguments (use data args for general dataset identificaton)"
    )
    group.add_argument("-sbp", "--segmentation_base_path", default=seg_path_default)
    group.add_argument("-r", "--result_name")
