import numpy as np
import h5py
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_cppreco.hdf5", "w") as f_out:
    for ev_idx, ev_raw_data in tqdm(enumerate(pd.read_csv(DATA_PATH + "l1Nano_WTo3Pion_PU200_cppreco.csv", chunksize=4))):
        aiereco_triplet_idxs = ev_raw_data[" D"].to_numpy()[:3]
        aiereco_w_mass = ev_raw_data[" D"].to_numpy()[3]
        aiereco_time = ev_raw_data[" TIME_NS"].to_numpy()[3]

        grp = f_out.create_group(f"{ev_idx}")
        grp.create_dataset("cppreco_triplet_idxs", data=aiereco_triplet_idxs, dtype=np.int16)
        grp.create_dataset("cppreco_w_mass", data=aiereco_w_mass, dtype=np.float32)
        grp.create_dataset("cppreco_time", data=aiereco_time, dtype=np.float64)