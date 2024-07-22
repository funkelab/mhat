import gunpowder as gp
import numpy as np
from gunpowder.torch import Train
import torch
import zarr
from darts_utils.segmentation import MtlsdModel, WeightedMSELoss
from darts_utils.gunpowder import AddLocalShapeDescriptor, NoiseAugment
from tqdm import tqdm


voxel_size = gp.Coordinate((1, 1, 1))
input_shape = gp.Coordinate((3, 400, 36))
output_shape = gp.Coordinate((1, 400, 36))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

def train(iterations, batch_size):
    
    phase = gp.ArrayKey('PHASE')
    mask = gp.ArrayKey('MASK')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    
    request = gp.BatchRequest()

    request.add(phase, input_size)
    request.add(mask, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    num_samples = 1000
    num_fmaps = 16
    
    ds_fact = [(2,2),(2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3), (3,3)]]*num_levels
    ksu = [[(3,3), (3,3)]]*(num_levels - 1)

    model = MtlsdModel(
      in_channels=3,
      num_fmaps=num_fmaps,
      fmap_inc_factor=2,
      downsample_factors=ds_fact,
      kernel_size_down=ksd,
      kernel_size_up=ksu,
      constant_upsample=True
    )

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())
   
    phase_array_specs = gp.ArraySpec(
        voxel_size=voxel_size,
        dtype=np.uint16,
        interpolatable=True
        )
    mask_array_specs = gp.ArraySpec(
        voxel_size=voxel_size,
        dtype=np.uint64,
        interpolatable=False
        )

    sources = tuple(gp.ZarrSource(
            f'/nrs/funke/data/darts/synthetic_data/dataset1/{i}.zarr',  
            {
                phase: 'phase',
                mask: 'mask'
            },  
            {
                phase: phase_array_specs,
                mask: mask_array_specs
            }
            ) + gp.Normalize(phase) + gp.Pad(size=(60,60,60), key=mask) + gp.RandomLocation()
            for i in range(1, num_samples+1)
    )
        

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(
       mirror_only=None,
       transpose_only=None,
       mirror_probs=(0,0.5,0.5),
       transpose_probs=[0,0,0]
    )

    pipeline += NoiseAugment(
        phase,
        mode="gaussian",
    )

    pipeline += gp.IntensityAugment(
        array=phase,
        scale_min=0.5,
        scale_max=1.5,
        shift_min=-0.5,
        shift_max=0.5
    )
    
    pipeline += gp.AddAffinities(
        affinity_neighborhood=[
            [0, -1, 0],
            [0, 0, -1]],
        labels=mask,
        affinities=gt_affs,
        dtype=np.float32,
        affinities_mask=affs_weights
        )
    pipeline += AddLocalShapeDescriptor(
        mask,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=(0, 20, 20),
    )


    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=10)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': phase
        },
        outputs={
            0: pred_lsds,
            1: pred_affs
        },
        loss_inputs={
            'lsds_prediction': pred_lsds,
            'lsds_target': gt_lsds,
            'lsds_weights': lsds_weights,
            'affs_prediction': pred_affs,
            'affs_target': gt_affs,
            'affs_weights': affs_weights
        },
        log_dir = './tensorboard_summaries/run3',
        log_every = 1
        )
    
    dataset_names = {
    phase: 'phase',
    mask: 'mask',
    pred_lsds: 'pred_lsds',
    gt_lsds: 'gt_lsds',
    lsds_weights: 'lsds_weights',
    pred_affs: 'pred_affs',
    gt_affs: 'gt_affs',
    affs_weights: 'affs_weights'
    }

    output_dir = '/nrs/funke/data/darts/synthetic_data/snapshots_folder/run3'
    output_filename = 'Snapshot_{iteration}.zarr'
    every = 1000
    pipeline += gp.Snapshot(
        dataset_names,
        output_dir=output_dir,
        output_filename=output_filename,
        every=every,
        additional_request=None,
        compression_type=None,
        dataset_dtypes=None,
        store_value_range=False
    )

    with gp.build(pipeline):
        progress = tqdm(range(iterations))
        for i in progress:
            print(f'Training iteration {i}')
            pipeline.request_batch(request)
            progress.set_description(f'Training iteration {i}') 


if __name__ == '__main__':
    train(100000,64)
