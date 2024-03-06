import argparse
import logging
import webbrowser
from pathlib import Path

import neuroglancer as ng
import neuroglancer.cli as ngcli
import zarr
from darts_utils.data.raw_data import add_data_args, get_raw_data
from darts_utils.data.segmentation_storage import get_segmentation

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
):
    print("visualizing data")
    raw_data: dict[tuple[str, str, str], zarr.array] = get_raw_data(
        data_base_path, dataset_name, fov=fov, channels=channels
    )
    for metadata, data in raw_data.items():
        layer = ng.LocalVolume(
            data=data,
            dimensions=ng.CoordinateSpace(
                names=["t", "c^", "y", "x"],
                units=["s", "", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="image",
        )
        viewer_context.layers.append(name="_".join(metadata), layer=layer)


def visualize_segmentation(
    viewer_context,
    dataset_name: str,
    data_base_path: str,
    result_name: str,
    fov: int,
    channels: list[str],
):
    print("visualizing data")
    seg_data: dict[tuple[str, str, str], zarr.array] = get_segmentation(
        data_base_path, dataset_name, result_name, fov=fov, channels=channels
    )
    for metadata, data in seg_data.items():
        layer = ng.LocalVolume(
            data=data,
            dimensions=ng.CoordinateSpace(
                names=["t", "c^", "y", "x"],
                units=["s", "", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="segmentation",
        )
        viewer_context.layers.append(name="_".join(metadata), layer=layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ngcli.add_server_arguments(parser)
    add_data_args(parser)
    parser.add_argument("-sr", "--seg_result_name")
    parser.add_argument(
        "-sbp",
        "--seg_base_path",
        default="/Volumes/funke/projects/darts/experiments/segmentation",
    )
    args = parser.parse_args()
    # ngcli.handle_server_arguments(args)
    ng.set_server_bind_address(bind_address="localhost", bind_port=8080)
    viewer = ng.Viewer()
    base_path = Path(args.data_base_path)
    if not args.dataset_name:
        datasets = [s.name for s in base_path.iterdir() if s.is_dir()]
        print(datasets)
    else:
        datasets = [args.dataset_name]

    for ds_name in datasets:
        with viewer.txn() as s:
            visualize_data(
                s,
                ds_name,
                data_base_path=base_path,
                fov=args.fov,
                channels=args.channels,
            )
        with viewer.txn() as s:
            visualize_segmentation(
                s,
                ds_name,
                data_base_path=args.seg_base_path,
                result_name=args.seg_result_name,
                fov=args.fov,
                channels=args.channels,
            )
        url = str(viewer)
        print(url)
        webbrowser.open_new(url)

        print("Press ENTER to go to next dataset")
        input()
