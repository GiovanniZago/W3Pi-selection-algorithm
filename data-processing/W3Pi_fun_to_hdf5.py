import numpy as np
import h5py
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_funreco.hdf5", "w") as f_out:
    for ev_idx, ev_raw_data in tqdm(enumerate(pd.read_csv(DATA_PATH + "l1Nano_WTo3Pion_PU200_funreco.csv", chunksize=4))):
        funreco_triplet_idxs = ev_raw_data[" D"].to_numpy()[:3]
        funreco_w_mass = ev_raw_data[" D"].to_numpy()[3]

        grp = f_out.create_group(f"{ev_idx}")
        grp.create_dataset("funreco_triplet_idxs", data=funreco_triplet_idxs, dtype=np.int16)
        grp.create_dataset("funreco_w_mass", data=funreco_w_mass, dtype=np.float32)