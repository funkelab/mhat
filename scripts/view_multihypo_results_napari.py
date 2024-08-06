import argparse
import logging
from pathlib import Path

from darts_utils.visualization import napari_utils

# _themes["dark"].font_size = "18pt"
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("experiment")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    experiment = args.experiment
    napari_utils.view_run(data_path, experiment, "multihypo")
