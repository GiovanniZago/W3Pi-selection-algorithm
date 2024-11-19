import numpy as np
import h5py
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

count_aiereco_gm = 0
count_reco_gm = 0

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_aiereco.hdf5", "r") as f_aiereco:
        with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_reco.hdf5", "r") as f_reco:
            for (grp_name_aiereco, grp_aiereco) in tqdm(f_aiereco.items()):
            
                if grp_name_aiereco in f_gm.keys():
                    if grp_name_aiereco not in f_reco.keys():
                        raise ValueError("The event should be there")
                    
                    grp_gm = f_gm[grp_name_aiereco]
                    aiereco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_aiereco["aiereco_triplet_idxs"]))

                    grp_reco = f_reco[grp_name_aiereco]
                    reco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_reco["reco_triplet_idxs"]))
    
                    if aiereco_gm_right:
                        count_aiereco_gm += 1

                    if reco_gm_right:
                        count_reco_gm += 1

print(count_aiereco_gm)
print(count_reco_gm)