import gunpowder as gp
import zarr
import numpy as np
from darts_utils.segmentation import MtlsdModel
import waterz 

from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed


def predict(checkpoint, phase_data, phase_file):
    phase = gp.ArrayKey("PHASE")
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    zarrfile = zarr.open(phase_data + "/phase", "r")

    voxel_size = gp.Coordinate((1, 1, 1))
    size = gp.Coordinate((3, 400, 36))
    output_shape = gp.Coordinate((1, 400, 36))

    input_size = size*voxel_size
    output_size = output_shape*voxel_size

    scan_request = gp.BatchRequest()

    scan_request.add(phase, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    context = (input_size - output_size) / 2

    phase_array_specs = gp.ArraySpec(
        voxel_size=voxel_size,
        dtype=np.uint16,
        interpolatable=True
        )

    phase_source = gp.ZarrSource(
        phase_data,
        {phase: phase_file},
        {phase: phase_array_specs}
        )
    
    with gp.build(phase_source):
        total_input_roi = phase_source.spec[phase].roi
        #total_output_roi = phase_source.spec[phase].roi
        total_output_roi = phase_source.spec[phase].roi.grow(-context, -context)
        # total_input_roi = [1:2, 0:36, 0:400]
        # total_output_roi = [0:3, 0:36, 0:400]

    lsd_shape = total_output_roi.get_shape() / voxel_size
    aff_shape = total_output_roi.get_shape() / voxel_size
    print(total_output_roi)
    print(lsd_shape)
    print(aff_shape)

    # generating the zarr file for saving
    zarrfile = zarr.open(target_dir + "/" + output_file, 'a')

   #zarrfile.create_dataset('phase', shape= total_input_roi.get_shape() / voxel_size)
    zarrfile.create_dataset('pred_lsds', shape = (6, lsd_shape[0], lsd_shape[1], lsd_shape[2]))
    zarrfile.create_dataset('pred_affs', shape = (2, aff_shape[0], aff_shape[1], aff_shape[2]))

    in_channels = 3
    num_fmaps = 16
    fmap_inc_factor = 2
    downsample_factors = [(2, 2), (2, 2)]
    num_levels = len(downsample_factors) + 1
    kernel_size_down = [[(3, 3), (3, 3)]] * num_levels
    kernel_size_up = [[(3, 3), (3, 3)]] * (num_levels - 1)
    constant_upsample = True

    model = MtlsdModel(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=constant_upsample
    )

    model.eval()

    pipeline = phase_source

    pipeline += gp.Normalize(phase)

    #pipeline += gp.Unsqueeze([phase])

    pipeline += gp.Stack(1)

    pipeline += gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={
            'input': phase
        },
        outputs={
            0: pred_lsds,
            1: pred_affs})

    #pipeline += gp.Squeeze([phase])

    pipeline += gp.Squeeze([phase, pred_lsds, pred_affs])

    dataset_names = {
        pred_lsds: 'pred_lsds',
        pred_affs: 'pred_affs',
    }

    pipeline += gp.ZarrWrite(
        dataset_names = dataset_names,
        output_dir = target_dir,
        output_filename = output_file
    )

    pipeline += gp.Scan(scan_request)

    predict_request = gp.BatchRequest()

    predict_request[phase] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    # print(
    #     f"\tphase: {batch[phase].data}, \tPred LSDS: {batch[pred_lsds].data}, \tPred Affs: {batch[pred_affs].data}")
    
    # return batch[phase].data, batch[pred_lsds].data, batch[pred_affs].data


def watershed_from_boundary_distance(
        
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret

def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        id_offset=0,
        min_seed_distance=3):

    mean_affs = 0.5*(affs[0] + affs[1])
  
    boundary_mask = mean_affs>0.5*max_affinity_value

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
   
    for time in range(0, affs.shape[1]):
        boundary_distances = distance_transform_edt(boundary_mask[time])

        frags, id_offset = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask[time],
            id_offset=id_offset,
            min_seed_distance=min_seed_distance)
        fragments[time] = frags
        
    return fragments

def get_segmentation(zarr_path, threshold):
    zarr_root = zarr.open(zarr_path, 'a')
    affinities = zarr_root["pred_affs"][:]

    fragments = watershed_from_affinities(affinities)
    zarr_root['fragments'] = fragments
    thresholds = [threshold]

    segmentation = np.zeros(fragments.shape, dtype=np.uint64)
    for time in range(affinities.shape[1]):
        data = np.expand_dims(fragments[time], axis=0)
        generator = waterz.agglomerate(
            affs=affinities.astype(np.float32),
            fragments=data,
            thresholds=thresholds,
        )

        seg = next(generator)
        segmentation[time] = seg
    
    zarr_root['pred_mask'] = segmentation
    zarr_root['pred_mask'].attrs['resolution'] = (1, 1, 1)


if __name__ == "__main__":

    data_zarr = "/nrs/funke/data/darts/synthetic_data/test1/18.zarr"
    phase_file = 'phase'
    checkpoint = "/groups/funke/home/sistaa/code/SyMBac/model_checkpoint_100000"

    data_root = zarr.open(data_zarr)
    if "pred_affs" not in data_root:
        predict(checkpoint, data_zarr, phase_file)

    threshold = 0.9

    # ws_affs = np.stack([
    #     pred_affs[0],
    #     pred_affs[1]]
    # )

    get_segmentation(data_zarr, threshold)

    #zarr_file['segmentation'].attrs['offset'] = offset