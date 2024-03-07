import argparse
from pathlib import Path

import git
import numpy as np
import zarr
from darts_utils.data.raw_data import add_data_args, get_raw_data
from darts_utils.experiment_metadata import get_experiment_metadata
from darts_utils.segmentation.voronoi_otsu import voronoi_otsu_labeling


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
    if not raw_data.keys():
        return
    # prepare output zarr with structure:
    # <dataset_name> (directory)
    #   segmentation.zarr (root)
    #       <result_name>_<spot_sigma>_<outline_sigma> (group)
    #           fov=<fov> (group)
    #               channel=<channel> (array)
    output_base_path = Path(output_base_path)
    if not output_base_path.exists():
        raise ValueError(f"Output base path {output_base_path} does not exist.")
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
    result_str = f"{result_name}_{spot_sigma}_{outline_sigma}"
    if result_str not in root.group_keys():
        result_group = root.create_group(result_str, overwrite=overwrite)
    else:
        result_group = root[result_str]

    for metadata, data in raw_data.items():
        _, fov_str, channel_str = metadata

        # make fov and channel groups
        if fov_str not in result_group.group_keys():
            fov_group = result_group.create_group(fov_str, overwrite=overwrite)
        else:
            fov_group = result_group[fov_str]

        if channel_str in fov_group.array_keys() and not overwrite:
            print(f"Result already present for {dataset_name} {fov_str} {channel_str}, skipping.")
            return

        channel_arr = fov_group.create_dataset(
            channel_str, shape=data.shape, dtype="uint16", overwrite=overwrite
        )
        if exp_metadata:
            channel_arr.attrs.update(**exp_metadata)
        for i, frame in enumerate(data):
            labeling = voronoi_otsu_labeling(
                frame.squeeze(), spot_sigma=spot_sigma, outline_sigma=outline_sigma
            )
            channel_arr[i, 0] = labeling.astype(np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument(
        "-sbp",
        "--segmentation_base_path",
        default="/Volumes/funke/projects/darts/experiments/segmentation",
    )
    parser.add_argument("--overwrite", type=bool, default=False)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    args = parser.parse_args()
    base_path = Path(args.data_base_path)
    if not args.dataset_name:
        datasets = [s.name for s in base_path.iterdir() if s.is_dir()]
        print(datasets)
    else:
        datasets = [args.dataset_name]

    for ds_name in datasets:
        print(ds_name)
        segment_data(
            ds_name,
            args.data_base_path,
            fov=args.fov,
            channels=args.channels,
            output_base_path=args.segmentation_base_path,
            result_name="voronoi_otsu",
            spot_sigma=2,
            outline_sigma=1,
            overwrite=args.overwrite,
            exp_metadata=get_experiment_metadata(),
        )
