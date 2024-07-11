import argparse
import logging
from pathlib import Path

import neuroglancer as ng
import neuroglancer.cli as ngcli

import zarr
import numpy as np

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)


def visualize_image(
    viewer_context,
    data,
    name,

):
    layer = ng.LocalVolume(
        data=data,
        dimensions=ng.CoordinateSpace(
            names=["y", "x"],
            units=["nm", "nm"],
            scales=[1, 1],
        ),
        volume_type="image",
    )
    # compute shader normalization ranges from one time point
    target_time = data.shape[0] // 2
    shader_min = 0.8 * data[target_time].min()
    shader_max = 1.2 * data[target_time].max()
    viewer_context.layers[name] = ng.ImageLayer(
        source=layer,
        shader_controls={"normalized": {"range": [shader_min, shader_max]}},
    )


def visualize_segmentation(
    viewer_context,
    data,
    name,
):
    data = data.astype(np.uint64)
    layer = ng.LocalVolume(
        data=data,
        dimensions=ng.CoordinateSpace(
            names=["y", "x"],
            units=["nm", "nm"],
            scales=[1, 1],
        ),
        volume_type="segmentation",
    )
    
    viewer_context.layers.append(name=name, layer=layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_zarr",)
    parser.add_argument("-g", "--groups", nargs="+")
    ngcli.add_server_arguments(parser)
    args = parser.parse_args()
    ng.set_server_bind_address(bind_address="0.0.0.0")
    base_path = Path(args.path_to_zarr)

    viewer = ng.Viewer()
    root = zarr.open(base_path)
    for group in args.groups:
        print(group)
        data = root[group]
        print(data.shape)
        if  group == "mask":
            with viewer.txn() as s:
                visualize_segmentation(
                    s,
                    data,
                    group
                )
        else:
            with viewer.txn() as s:
                visualize_image(
                    s,
                    data,
                    group
                )
    url = str(viewer)
    print(url)
    input()
