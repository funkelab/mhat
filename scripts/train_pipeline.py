import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Train
from lsd.train.gp import AddLocalShapeDescriptor
from tqdm import tqdm

voxel_size = gp.Coordinate((1, 1, 1))
input_shape = gp.Coordinate((3, 400, 36))
output_shape = gp.Coordinate((3, 400, 36))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

class MtlsdModel(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        constant_upsample):
  
        super().__init__()

        # create unet
        self.unet = UNet(
          in_channels=in_channels,
          num_fmaps=num_fmaps,
          fmap_inc_factor=fmap_inc_factor,
          downsample_factors=downsample_factors,
          kernel_size_down=kernel_size_down,
          kernel_size_up=kernel_size_up,
          constant_upsample=constant_upsample,
          padding='same')

        # create lsd and affs heads
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        # pass raw through unet
        z = self.unet(input)

        # pass output through heads
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds, affs

# combine the lsds and affs losses

class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        lsds_prediction,
        lsds_target,
        lsds_weights,
        affs_prediction,
        affs_target,
        affs_weights,
    ):

        # calc each loss and combine
        loss1 = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        loss2 = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss1 + loss2
    
def train(
    iterations,
    batch_size
   ):
    
    phase = gp.ArrayKey('PHASE')
    mask = gp.ArrayKey('MASK')
    # gt_lsds = gp.ArrayKey('GT_LSDS')
    # lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    # pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    
    request = gp.BatchRequest()

    request.add(phase, input_size)
    request.add(mask, output_size)
    # request.add(gt_lsds, output_size)
    # request.add(lsds_weights, output_size)
    # request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    num_samples = 200
    num_fmaps = 16
    
    ds_fact = [(1,2,2),(1,2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3,3), (3,3,3)]]*num_levels
    ksu = [[(3,3,3), (3,3,3)]]*(num_levels - 1)

    model = MtlsdModel(
      in_channels=1,
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
        voxel_size=input_shape,
        dtype=np.uint16,
        interpolatable=True
        )
    mask_array_specs = gp.ArraySpec(
        voxel_size=input_shape,
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

    #pipeline += gp.Normalize(phase)    

    pipeline += gp.SimpleAugment(
       mirror_only=None,
       transpose_only=None,
       mirror_probs=(0,0.5,0.5),
       transpose_probs=[0,0,0]
    )

    pipeline += gp.IntensityAugment(
        array=phase,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1
    )

    # pipeline += AddLocalShapeDescriptor(
    #     mask,
    #     gt_lsds,
    #     lsds_mask=lsds_weights,
    #     sigma=20,
    # )

    pipeline += gp.AddAffinities(
    affinity_neighborhood=[
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]],
    labels=mask,
    affinities=gt_affs,
    dtype=np.float32,
    affinities_mask=affs_weights
    )

    pipeline += gp.Stack(batch_size)

   # pipeline += gp.PreCache(num_workers=10)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': phase
        },
        outputs={
            # 0: pred_lsds,
            1: pred_affs
        },
        loss_inputs={
            # 'lsds_prediction': pred_lsds,
            # 'lsds_target': gt_lsds,
            # 'lsds_weights': lsds_weights,
            'affs_prediction': pred_affs,
            'affs_target': gt_affs,
            'affs_weights': affs_weights
        })

    with gp.build(pipeline):
        progress = tqdm(range(iterations))
        for i in progress:
            print(f'Training iteration {i}')
            batch = pipeline.request_batch(request)
            zarr_group = zarr.create_group(f'/nrs/funke/data/darts/synthetic_data/debug/{i}.zarr', "w")
            zarr_group['phase'] = batch[phase]
            zarr_group['mask'] = batch[mask]

            progress.set_description(f'Training iteration {i}') 


if __name__ == '__main__':
    train(5,2)