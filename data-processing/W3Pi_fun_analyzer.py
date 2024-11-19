import numpy as np
import h5py
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

evts_funreco_gm = []
evts_reco_gm = []

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_funreco.hdf5", "r") as f_funreco:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_reco.hdf5", "r") as f_reco:
        with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
            for (grp_name_funreco, grp_funreco) in tqdm(f_funreco.items()):

                if grp_name_funreco not in f_gm.keys():
                    raise ValueError("The event should be inside f_gm")
                
                if grp_name_funreco not in f_reco.keys():
                    raise ValueError("The event should be inside f_reco")
                
                grp_gm = f_gm[grp_name_funreco]

                if (grp_gm.attrs["is_acc"] != 1) or (grp_gm.attrs["is_gm"] != 1):
                    continue

                funreco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_funreco["funreco_triplet_idxs"]))

                grp_reco = f_reco[grp_name_funreco]
                reco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_reco["reco_triplet_idxs"]))
    
                if funreco_gm_right:
                    evts_funreco_gm.append(int(grp_name_funreco))

                if reco_gm_right:
                    evts_reco_gm.append(int(grp_name_funreco))

evts_funreco_gm.sort()
evts_reco_gm.sort()

for ev_idx in evts_reco_gm:
    if ev_idx not in evts_funreco_gm:
        print(ev_idx)