import argparse
from pathlib import Path

import toml
import zarr
from funlib.geometry.roi import Roi
from mhat.data import (
    DataZarr,
    RawDataZarr,
    Row,
    SegmentationZarr,
    add_data_args,
    add_segmentation_args,
)


def delete_old_attempts(dz: DataZarr):
    meta_group = zarr.open_group(dz.store, mode=dz, path="")
    print(f"meta group attrs before: {meta_group.attrs.asdict()}")
    if "rows" in meta_group.attrs:
        del meta_group.attrs["rows"]
    print(f"meta group attrs after: {meta_group.attrs.asdict()}")
    fov_group = zarr.open_group(
        dz.store, mode=dz.mode, path=dz._get_fov_group_path(fov=3)
    )
    print(f"fov group attrs before: {fov_group.attrs.asdict()}")
    if "rows" in fov_group.attrs:
        del meta_group.attrs["rows"]

    if "roi_offset" in fov_group.attrs:
        del fov_group.attrs["roi_offset"]
    if "roi_size" in fov_group.attrs:
        del fov_group.attrs["roi_size"]
    print(f"fov group attrs after: {fov_group.attrs.asdict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_segmentation_args(parser)
    parser.add_argument("config")
    args = parser.parse_args()

    config_path = Path(args.config)
    assert config_path.is_file()
    config = toml.load(config_path)
    print(config)
    dataset = config["dataset"]
    fov = config["fov"]
    roi = Roi(config["roi_offset"], config["roi_shape"])
    up = config["up"]
    row = Row(None, fov, roi, up)
    print(roi.get_shape())
    data_base_path = args.data_base_path
    seg_base_path = args.segmentation_base_path
    raw_zarr = RawDataZarr(data_base_path, dataset, store_type="flat", mode="a")
    seg_zarr = SegmentationZarr(seg_base_path, dataset, "voronoi_otsu_2_1", mode="a")

    # clean up all old attempts:
    # delete_old_attempts(raw_zarr)
    # delete_old_attempts(seg_zarr)
    raw_zarr.add_row(row)
    seg_zarr.add_row(row)

    print(raw_zarr.get_rows())
    print(seg_zarr.get_rows())

    # raw_zarr.delete_row(0)
    # seg_zarr.delete_row(0)

    # print(raw_zarr.get_rows())
    # print(seg_zarr.get_rows())
