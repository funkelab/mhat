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
    parser.add_argument("exp_path")
    parser.add_argument("-s", "--start_frame", type=int, default=None)
    parser.add_argument("-e", "--end_frame", type=int, default=None)
    args = parser.parse_args()

    dt = "2024-08-08_15-35-31"

    exp_path = Path(args.exp_path)
    data_path = exp_path.parent
    experiment = exp_path.stem
    napari_utils.view_run(
        data_path, experiment, experiment, args.start_frame, args.end_frame
    )
