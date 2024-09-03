from pprint import pprint

import nd2
import zarr

infile = "/Volumes/funke/data/darts/240411_phase_fluor/110424_LIB518_40xPh_epi.nd2"
outfile = "/Volumes/funke/data/darts/240411_phase_fluor/240411_phase_fluor.zarr"
outgroup = "raw"

with nd2.ND2File(infile) as myfile:
    metadata = myfile.metadata
    attributes = myfile.attributes
    ome_metadata = myfile.ome_metadata()
    num_fovs = myfile.shape[1]
    # pprint(myfile.attributes)
    pprint(myfile.shape)
    pprint(myfile.ndim)
    pprint(myfile.dtype)
    pprint(myfile.sizes)
    pprint(myfile.voxel_size())

    data = myfile.to_dask().rechunk(chunks=(1, 1, 1, 2960, 5056))
    print(data.chunksize)

    channels = next(iter(ome_metadata.images)).pixels.channels
    channels_dict = {"channels": [channel.model_dump_json() for channel in channels]}
    root = zarr.open(outfile)
    root.attrs.put(channels_dict)

    for fov in range(num_fovs):
        fov_data = data[:, fov]
        pprint(fov_data.shape)
        fov_group = f"{fov}/{outgroup}"
        print(fov_group)
        fov_data.to_zarr(url=outfile, component=fov_group, dimension_separator="/")
        break
