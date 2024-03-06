from pathlib import Path
from typing import Optional

import zarr


def get_segmentation(
    base_path: str | Path,
    dataset_name: str,
    result_name: str,
    fov: Optional[int] = None,
    channels: Optional[list[str]] = None,
) -> dict[tuple[str, str, str, str], zarr.Array]:
    base_path = Path(base_path)
    dataset_path = base_path / dataset_name
    if not dataset_path.exists():
        raise ValueError(f"No data found at {dataset_path}.")
    raw_data_path = dataset_path / "segmentation.zarr"
    if not raw_data_path.exists():
        raise ValueError(f"No data found at {raw_data_path}.")
    root = zarr.open(raw_data_path, "r")
    if not result_name:
        # pick first one available
        results = list(root.group_keys())
        result_name = results[0]
    else:
        if result_name not in root.group_keys():
            raise ValueError(f"Result {result_name} not present in {dataset_name}.")
    result_group = root[result_name]

    if not fov:
        # pick first one available
        fovs = list(result_group.group_keys())
        fov_str = fovs[0]
        fov = fov_str.split("=")[1]
    else:
        fov_str = f"fov={fov}"
        if fov_str not in result_group.group_keys():
            raise ValueError(f"Fov {fov} not present in {dataset_name}.")
    fov_group = result_group[fov_str]

    if not channels:
        channel_strs = list(fov_group.array_keys())
        channels = [cs.split("=")[1] for cs in channel_strs]
    else:
        channel_strs = [f"channel={channel}" for channel in channels]
        for channel_str in channel_strs:
            if channel_str not in fov_group.array_keys():
                raise ValueError(f"{channel_str} not present in {dataset_name}")

    return {
        (dataset_name, result_name, fov_str, channel_str): fov_group[channel_str]
        for channel_str in channel_strs
    }
