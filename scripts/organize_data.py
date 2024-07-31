from pathlib import Path
import shutil
import os


for file_num in range(1,101):
    old_zarr_directory = Path(f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}.zarr')
    old_csv_file = Path(f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}.csv')
    new_directory = Path(f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}')

    old_zarr_name = f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}/{file_num}.zarr'
    new_zarr_name = f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}/data.zarr'
    old_csv_name = f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}/{file_num}.csv'
    new_csv_name = f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}/gt_tracks.csv'

    os.mkdir(new_directory)

    shutil.move(old_zarr_directory, new_directory)
    shutil.move(old_csv_file, new_directory)

    os.rename(old_zarr_name, new_zarr_name)
    os.rename(old_csv_name, new_csv_name)

    print(f"New directory created with Zarr directory, existing CSV, and new CSV at: {new_directory}")
