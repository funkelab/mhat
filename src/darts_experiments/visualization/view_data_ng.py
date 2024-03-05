import neuroglancer as ng
import neuroglancer.cli as ngcli
import argparse
import webbrowser
import zarr
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)

def visualize_data(
        viewer_context, 
        dataset_name: str,
        data_base_path: str,
        fov: int,
        channels: list[str],
        start_time=None,
        end_time=None,
):
    print("visualizing data")
    # open dataset
    dataset_path = data_base_path / dataset_name
    if not dataset_path.exists():
        raise ValueError(f"No data found at {dataset_path}.")
    raw_data_path = dataset_path / "raw.zarr"
    if not raw_data_path.exists():
        raise ValueError(f"No data found at {raw_data_path}.")
    root = zarr.open(raw_data_path, 'r')

    if not fov:
        # pick first one available
        fovs = list(root.group_keys())
        fov_str = fovs[0]
        fov = fov_str.split("=")[1]
    else:
        fov_str = f"fov={fov}"
    fov_group = root[fov_str]

    print(channels)
    if not channels:
        channel_strs = list(fov_group.array_keys())
        channels = [cs.split("=")[1] for cs in channel_strs]
    else:
        channel_strs = [f"channel={channel}" for channel in channels]

    for channel_str, channel in zip(channel_strs, channels):
        channel_ds = fov_group[channel_str]
        print(
            f"DS {dataset_name} fov {fov} channel {channel} shape {channel_ds.shape}"
        )
        layer = ng.LocalVolume(
            data=channel_ds, 
            dimensions = ng.CoordinateSpace(
                names=["t","c^", "y", "x"],
                units=["s","", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="image",
        )
        viewer_context.layers.append(
            name=f"{dataset_name}_fov={fov}_channel={channel}",
            layer=layer
        )
    return viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ngcli.add_server_arguments(parser)
    parser.add_argument("-d", "--dataset_name", help = "name of dataset to visualize")
    parser.add_argument("-f", "--fov", type=int, help="FOV to view")
    parser.add_argument(
        "-c", "--channels", type=str, nargs="+",
        help="Channel to view. Options: BF, YFP-DUAL",
    )
    parser.add_argument("-dbp", "--data_base_path", default="/Volumes/funke/data/darts")
    parser.add_argument("--start_time", type=int)
    parser.add_argument("--end_time", type=int)
    args = parser.parse_args()
    #ngcli.handle_server_arguments(args)
    ng.set_server_bind_address(bind_address='localhost', bind_port=8080)
    viewer = ng.Viewer()
    base_path = Path(args.data_base_path)
    if not args.dataset_name:
        print(list(base_path.iterdir()))
        print(list(base_path.iterdir())[0].is_dir())
        print(list(base_path.iterdir())[0].name)
        datasets = [s.name for s in base_path.iterdir() if s.is_dir()]
        print(datasets)
    else:
        datasets = [args.dataset_name]

    for ds_name in datasets:
        print(ds_name)
        with viewer.txn() as s:
            visualize_data(
                s,
                ds_name,
                data_base_path = base_path,
                fov=args.fov,
                channels=args.channels,
                start_time = args.start_time,
                end_time = args.end_time,
            )
        url = str(viewer)
        print(url)
        webbrowser.open_new(url)

        print("Press ENTER to go to next dataset")
        input()