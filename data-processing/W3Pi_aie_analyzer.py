import numpy as np
import h5py
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

evt_gm = []
evts_reco = []
evts_funreco = []
evts_aiereco = []

evt_list = np.arange(2000)

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_reco.hdf5", "r") as f_reco:
        with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_funreco.hdf5", "r") as f_funreco:
            with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_aiereco.hdf5", "r") as f_aiereco:
                for (grp_name_gm, grp_gm) in tqdm(f_gm.items()):
                    if int(grp_name_gm) not in evt_list:
                        continue

                    if grp_gm.attrs["is_acc"] != 1 or grp_gm.attrs["is_gm"] != 1:
                        continue

                    evt_gm.append(int(grp_name_gm))
                
                    if grp_name_gm in f_reco.keys():
                        grp_reco = f_reco[grp_name_gm]
                        reco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_reco["reco_triplet_idxs"]))

                        if reco_gm_right:
                            evts_reco.append(int(grp_name_gm))

                    if grp_name_gm in f_funreco.keys():
                        grp_funreco = f_funreco[grp_name_gm]
                        funreco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_funreco["funreco_triplet_idxs"]))

                        if funreco_gm_right:
                            evts_funreco.append(int(grp_name_gm))

                    if grp_name_gm in f_aiereco.keys():
                        grp_aiecreco = f_aiereco[grp_name_gm]
                        aiereco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_aiecreco["aiereco_triplet_idxs"]))

                        if aiereco_gm_right:
                            evts_aiereco.append(int(grp_name_gm))



print(len(evt_gm))
print(len(evts_reco))
print(len(evts_funreco))
print(len(evts_aiereco))