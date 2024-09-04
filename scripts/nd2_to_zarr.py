from pprint import pprint
from dataclasses import asdict
import ome_types

import nd2
import zarr

infile = "/nrs/funke/data/darts/240411_phase_fluor/110424_LIB518_40xPh_epi.nd2"
outfile = "/nrs/funke/data/darts/240411_phase_fluor/240411_phase_fluor.zarr"
outgroup = "raw"

with nd2.ND2File(infile) as myfile:
    # save all metadata in the zarr attrs
    root = zarr.open(outfile)
    root.attrs.update({"metadata": asdict(myfile.metadata)})
    root.attrs.update({"attributes": myfile.attributes._asdict()})
    root.attrs.update({"ome_metadata": myfile.ome_metadata().model_dump_json()})
    # include special channels attribute because we know we need it
    channels = next(iter(myfile.ome_metadata().images)).pixels.channels
    channels_dict = {"channels": [channel.model_dump_json() for channel in channels]}
    root.attrs.update(channels_dict)

    data = myfile.to_dask().rechunk(chunks=(1, 1, 1, 740, 1264))
    print(data.chunksize)

    num_fovs = myfile.shape[1] 
    for fov in range(num_fovs):
        fov_data = data[:, fov]
        fov_group = f"{fov}/{outgroup}"
        print(fov_group)
        pprint(fov_data.shape)
        fov_data.to_zarr(url=outfile, component=fov_group, dimension_separator="/")
