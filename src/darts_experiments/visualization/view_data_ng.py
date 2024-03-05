import neuroglancer as ng
import neuroglancer.cli as ngcli
import argparse
import webbrowser
import zarr
from pathlib import Path
import logging
import numpy as np
from darts_experiments.utils.data import get_raw_data, add_data_args

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
    raw_data: dict[str, zarr.array] = get_raw_data(
        data_base_path, dataset_name, fov=fov, channels=channels
    )
    for name, data in raw_data.items():

        layer = ng.LocalVolume(
            data=data, 
            dimensions = ng.CoordinateSpace(
                names=["t","c^", "y", "x"],
                units=["s","", "nm", "nm"],
                scales=[1, 1, 1, 1],
            ),
            volume_type="image",
        )
        viewer_context.layers.append(
            name=name,
            layer=layer
        )
    return viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ngcli.add_server_arguments(parser)
    add_data_args(parser)
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
            )
        url = str(viewer)
        print(url)
        webbrowser.open_new(url)

        print("Press ENTER to go to next dataset")
        input()