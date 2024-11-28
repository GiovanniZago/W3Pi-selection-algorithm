import h5py
import numpy as np
from tqdm import tqdm
import csv

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_FixedPoint.hdf5", "r") as f_hdf:
    with open(DATA_PATH + "PuppiSignal_224_128_up.csv", "w") as f_csv:
        csv_writer = csv.writer(f_csv)

        header = "CMD,D,D,D,D,D,D,D,D,TLAST,TKEEP"
        csv_writer.writerow(header)

        for grp_name, grp in tqdm(f_hdf.items()):
            rows_per_var = 28
            cols_per_var = 8
            new_shape = (rows_per_var, cols_per_var)

            pts = np.array(grp["pt"])
            pts.resize(new_shape, refcheck=False)
            etas = np.array(grp["eta"])
            etas.resize(new_shape, refcheck=False)
            phis = np.array(grp["phi"])
            phis.resize(new_shape, refcheck=False)
            pdg_ids = np.array(grp["pdg_id"])
            pdg_ids.resize(new_shape, refcheck=False)


            data = np.hstack((pts, etas, phis, pdg_ids))
            
            for row in data:
                csv_writer.writerow(["DATA"] + list(row) + ["0", "-1"])

