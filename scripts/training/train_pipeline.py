import argparse
from dataclasses import dataclass
from pathlib import Path

import gunpowder as gp
import numpy as np
import toml
import torch
from darts_utils.gunpowder import AddLocalShapeDescriptor, NoiseAugment
from darts_utils.segmentation import MtlsdModel, WeightedMSELoss
from gunpowder.torch import Train
from tqdm import tqdm


@dataclass
class TrainConfig:
    noise_max: float
    max_iterations: int
    batch_size: int
    input_shape: tuple(int)
    input_voxel_size: tuple(int)
    input_dir: str
    input_zarr_glob: str
    num_workers: int
    tensorboard_dir: str = "./tensorboard/run3"
    snapshot_dir: str = None


def train(config: TrainConfig):
    voxel_size = gp.Coordinate(config.input_voxel_size)
    input_shape = gp.Coordinate(config.input_shape)
    # output is always the same as input but with one in the time dim
    output_shape = gp.Coordinate((1, *config.input_shape[1:]))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey("RAW")
    mask = gp.ArrayKey("MASK")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(mask, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    # model parameters
    num_fmaps = 16
    ds_fact = [(2, 2), (2, 2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3), (3, 3)]] * num_levels
    ksu = [[(3, 3), (3, 3)]] * (num_levels - 1)

    model = MtlsdModel(
        in_channels=3,
        num_fmaps=num_fmaps,
        fmap_inc_factor=2,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu,
        constant_upsample=True,
    )

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    raw_array_specs = gp.ArraySpec(
        voxel_size=voxel_size, dtype=np.uint16, interpolatable=True
    )
    mask_array_specs = gp.ArraySpec(
        voxel_size=voxel_size, dtype=np.uint64, interpolatable=False
    )

    zarr_base_path = Path(config.input_dir)
    pad_amt = config.lsd_sigma * 3
    sources = tuple(
        gp.ZarrSource(
            zarr,
            {raw: "phase", mask: "mask"},
            {raw: raw_array_specs, mask: mask_array_specs},
        )
        + gp.Normalize(raw)
        + gp.Pad(size=(pad_amt,) * 3, key=mask)
        + gp.RandomLocation()
        for zarr in zarr_base_path.glob(config.input_zarr_glob)
    )

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(
        mirror_only=None,
        transpose_only=None,
        mirror_probs=(0, 0.5, 0.5),
        transpose_probs=[0, 0, 0],
    )

    pipeline += NoiseAugment(raw, noise_max=config.noise_max)

    pipeline += gp.IntensityAugment(
        array=raw, scale_min=0.5, scale_max=1.5, shift_min=-0.5, shift_max=0.5
    )

    pipeline += gp.AddAffinities(
        affinity_neighborhood=[[0, -1, 0], [0, 0, -1]],
        labels=mask,
        affinities=gt_affs,
        dtype=np.float32,
        affinities_mask=affs_weights,
    )
    pipeline += AddLocalShapeDescriptor(
        mask,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=(0, config.lsd_sigma, config.lsd_sigma),
    )

    pipeline += gp.Stack(config.batch_size)

    pipeline += gp.PreCache(num_workers=config.num_workers)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        outputs={0: pred_lsds, 1: pred_affs},
        loss_inputs={
            "lsds_prediction": pred_lsds,
            "lsds_target": gt_lsds,
            "lsds_weights": lsds_weights,
            "affs_prediction": pred_affs,
            "affs_target": gt_affs,
            "affs_weights": affs_weights,
        },
        log_dir=config.tensorboard_dir,
        log_every=1,
    )

    dataset_names = {
        raw: "phase",
        mask: "mask",
        pred_lsds: "pred_lsds",
        gt_lsds: "gt_lsds",
        lsds_weights: "lsds_weights",
        pred_affs: "pred_affs",
        gt_affs: "gt_affs",
        affs_weights: "affs_weights",
    }

    output_dir = config.snapshot_dir
    output_filename = "Snapshot_{iteration}.zarr"
    every = 1000
    if output_dir is not None:
        pipeline += gp.Snapshot(
            dataset_names,
            output_dir=output_dir,
            output_filename=output_filename,
            every=every,
            additional_request=None,
            compression_type=None,
            dataset_dtypes=None,
            store_value_range=False,
        )

    with gp.build(pipeline):
        progress = tqdm(range(config.iterations))
        for i in progress:
            print(f"Training iteration {i}")
            pipeline.request_batch(request)
            progress.set_description(f"Training iteration {i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = TrainConfig(**toml.load(args.config))
    train(config)
