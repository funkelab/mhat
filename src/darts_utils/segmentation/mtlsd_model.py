import torch
from funlib.learn.torch.models import UNet, ConvPass
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
        self.lsd_head = ConvPass(num_fmaps, 6, [[1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, 2, [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        # pass raw through unet
        z = self.unet(input)

        # pass output through heads
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)
        # add dummy time dimension
        lsds = torch.unsqueeze(lsds, dim=2)
        affs = torch.unsqueeze(affs, dim=2)

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
        print(f"{lsds_prediction.shape=}")
        print(f"{lsds_target.shape=}")
        print(f"{affs_prediction.shape=}")
        print(f"{affs_target.shape=}")
        loss1 = self._calc_loss(lsds_prediction, lsds_target,  lsds_weights)
        loss2 = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss1 + loss2
    
