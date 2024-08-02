import argparse
import csv
import logging
from pathlib import Path

import napari
import networkx as nx
import zarr
from darts_utils.tracking import utils
from motile_toolbox.visualization import to_napari_tracks_layer

# _themes["dark"].font_size = "18pt"
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)


def load_prediction(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        graph = nx.DiGraph()
        for row in reader:
            node_id = int(float(row["id"]))
            attrs = {
                "time": int(float(row["time"])),
                "pos": [int(float(row["x"])), int(float(row["y"]))],
            }
            parent_id = int(float(row["parent_id"]))
            graph.add_node(node_id, **attrs)
            if parent_id != -1:
                graph.add_edge(parent_id, node_id)
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarrpath")
    args = parser.parse_args()

    zarr_path = Path(args.zarrpath)
    zarr_root = zarr.open(zarr_path, "r")

    lineage_csv_path = zarr_path.parent / ("gt_tracks.csv")

    prediction_csv_path = zarr_path.parent / ("multihypo_pred_tracks.csv")

    # Initialize Napari viewer
    viewer = napari.Viewer()

    phase_data = zarr_root["phase"][:]
    viewer.add_image(phase_data, name="phase")
    mask_data = zarr_root["mask"][:]
    viewer.add_labels(mask_data, name="gt_mask")

    pred_mask_data = zarr_root["multihypo_pred_mask"][:]

    lineage = utils.read_gt_tracks(zarr_path, lineage_csv_path)

    track_data, track_props, track_edges = to_napari_tracks_layer(lineage)
    tracks_layer = napari.layers.Tracks(
        track_data,
        properties=track_props,
        graph=track_edges,
        name="gt_tracks",
        tail_length=3,
    )
    viewer.add_layer(tracks_layer)

    pred_lineage = load_prediction(prediction_csv_path)
    pred_mask_data = utils.relabel_segmentation(pred_lineage, pred_mask_data)
    viewer.add_labels(pred_mask_data, name="pred_mask")
    track_data, track_props, track_edges = to_napari_tracks_layer(pred_lineage)
    tracks_layer = napari.layers.Tracks(
        track_data,
        properties=track_props,
        graph=track_edges,
        name="pred_tracks",
        tail_length=3,
    )
    viewer.add_layer(tracks_layer)

    print("Done adding images")
    # Start the Napari GUI event loop
    napari.run()
