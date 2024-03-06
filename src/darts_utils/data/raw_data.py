import argparse
from pathlib import Path
from typing import Optional

import zarr


def get_raw_data(
    base_path: str | Path,
    dataset_name: str,
    fov: Optional[int] = None,
    channels: Optional[list[str]] = None,
) -> dict[tuple[str, str, str], zarr.Array]:
    """Get the zarr array containing the raw data. Performs error checking if values
    are incorrect. If fov not provided, automatically load first one. If channels
    not provided, return all of them.

    Args:
        base_path (str | Path): Path to raw data zarrs. Probably /nrs/funke/data/darts,
            or /Volumes/funke/data/darts if mounted on a mac.
        dataset_name (str): Name of dataset to access.
        fov (int, optional): Which fov to access. If not provided, automatically
            discovers which fovs are present and takes the first one. Defaults to None.
        channels (list[str], optional): Which channels to access. If not provided,
            automatically discovers which channels are present and returns all of them.
            Defaults to None.

    Raises:
        ValueError: If dataset directory doesn't exist
        ValueError: If raw.zarr doesn't exist in dataset directory
        ValueError: If provided fov not present in zarr
        ValueError: If provided channels not present in zarr

    Returns:
        dict[str, zarr.Array]: A map from data_name
            ({ds_name}_fov={fov}_channel={channel}) to zarr array containing data.
            These are lazy and thus not loaded into memory until accessed.
    """
    base_path = Path(base_path)
    dataset_path = base_path / dataset_name
    if not dataset_path.exists():
        raise ValueError(f"No data found at {dataset_path}.")
    raw_data_path = dataset_path / "raw.zarr"
    if not raw_data_path.exists():
        raise ValueError(f"No data found at {raw_data_path}.")
    root = zarr.open(raw_data_path, "r")

    if not fov:
        # pick first one available
        fovs = list(root.group_keys())
        fov_str = fovs[0]
        fov = fov_str.split("=")[1]
    else:
        fov_str = f"fov={fov}"
        if fov_str not in root.group_keys():
            raise ValueError(f"Fov {fov} not present in {dataset_name}.")
    fov_group = root[fov_str]

    if not channels:
        channel_strs = list(fov_group.array_keys())
        channels = [cs.split("=")[1] for cs in channel_strs]
    else:
        channel_strs = [f"channel={channel}" for channel in channels]
        for channel_str in channel_strs:
            if channel_str not in fov_group.array_keys():
                raise ValueError(f"{channel_str} not present in {dataset_name}")

    return {
        (dataset_name, fov_str, channel_str): fov_group[channel_str]
        for channel_str in channel_strs
    }


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
