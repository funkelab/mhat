from darts_experiments.segmentation.voronoi_otsu import voronoi_otsu_labeling
from darts_experiments.utils.data import get_raw_data, add_data_args
import argparse
from pathlib import Path
import zarr
import git
import numpy as np


def segment_data(
    dataset_name,
    data_base_path,
    fov,
    channels,
    output_base_path,
    result_name,
    spot_sigma=2,
    outline_sigma=10,
    overwrite=False,
    exp_metadata=None,
):
    raw_data: dict[tuple[str, str, str], zarr.Array] = get_raw_data(
        data_base_path, dataset_name, fov=fov, channels=channels
    )
    fluorescent_prefixes = [
        "mCherry",
        "RFP",
        "YFP",
        "GFP",
        "CFP",
    ]
    raw_data = {
        metadata: data
        for metadata, data in raw_data.items()
        if any(prefix in metadata[2] for prefix in fluorescent_prefixes)
    }
    print(raw_data.keys())
    # prepare output zarr with structure:
    # <dataset_name> (directory)
    #   segmentation.zarr (root)
    #       <result_name>_<spot_sigma>_<outline_sigma> (group)
    #           fov=<fov> (group)
    #               channel=<channel> (array)
    output_base_path = Path(output_base_path)
    assert (
        output_base_path.exists()
    ), f"Output base path {output_base_path} does not exist."
    output_dataset_path = output_base_path / dataset_name
    if not output_dataset_path.exists():
        print(f"Making output directory at {output_dataset_path}")
        output_dataset_path.mkdir()
    zarr_root_path = output_dataset_path / "segmentation.zarr"
    if zarr_root_path.exists():
        root = zarr.open(zarr_root_path, "r+")
    else:
        print(f"Making zarr at {zarr_root_path}")
        # create nested store
        store = zarr.NestedDirectoryStore(zarr_root_path)
        root = zarr.open_group(store, mode="a")

    # make result group
    result_group = root.create_group(
        f"{result_name}_{spot_sigma}_{outline_sigma}", overwrite=overwrite
    )

    for metadata, data in raw_data.items():
        _, fov_str, channel_str = metadata

        # make fov and channel groups
        if fov_str not in result_group.group_keys():
            fov_group = result_group.create_group(fov_str, overwrite=overwrite)
        else:
            fov_group = result_group[fov_str]
        channel_arr = fov_group.create_dataset(
            channel_str, shape=data.shape, dtype="uint16", overwrite=overwrite
        )
        if exp_metadata:
            channel_arr.attrs.update(**exp_metadata)
        for i, frame in enumerate(data):
            print(f"frame max: {frame.max()}")
            labeling = voronoi_otsu_labeling(
                frame.squeeze(), spot_sigma=spot_sigma, outline_sigma=outline_sigma
            )
            print(labeling.max())
            print(labeling.astype(np.uint16).max())
            channel_arr[i, 0] = labeling.astype(np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument(
        "-obp",
        "--output_base_path",
        default="/Volumes/funke/projects/darts/experiments/segmentation",
    )
    parser.add_argument("--overwrite", type=bool, default=False)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    args = parser.parse_args()
    segment_data(
        args.dataset_name,
        args.data_base_path,
        fov=args.fov,
        channels=args.channels,
        output_base_path=args.output_base_path,
        result_name="voronoi_otsu",
        spot_sigma=2,
        outline_sigma=1,
        overwrite=args.overwrite,
        exp_metadata={"git_hash": sha},
    )
