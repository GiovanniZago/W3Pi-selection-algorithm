import numpy as np
import h5py
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

counter = 0

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_reco_v2.hdf5", "r") as f_reco:
        for (grp_name_gm, grp_gm), (grp_name_reco, grp_reco) in tqdm(zip(f_gm.items(), f_reco.items())):
            reco_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_reco["reco_triplet_idxs"]))
            if reco_right:
                counter += 1

            # not_reco = np.allclose(grp_reco["reco_triplet_idxs"], 0)

            # if not not_reco:
            #     counter += 1

        print(counter)