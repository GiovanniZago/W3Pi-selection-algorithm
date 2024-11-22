import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep

hep.style.use("CMS")

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

cor_funreco_gm = []
par_funreco_gm = []
not_funreco_gm = []
wrong_funreco_gm = []

cor_aiereco_gm = []
par_aiereco_gm = []
not_aiereco_gm = []
wrong_aiereco_gm = []

evt_list = np.arange(2000)

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_funreco.hdf5", "r") as f_funreco:
        with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_aiereco.hdf5", "r") as f_aiereco:
            for (grp_name_gm, grp_gm) in tqdm(f_gm.items()):
                if int(grp_name_gm) not in evt_list:
                    continue

                if (grp_gm.attrs["is_acc"] != 1) or (grp_gm.attrs["is_gm"] != 1):
                    continue
                
                
                if grp_name_gm not in f_funreco.keys():
                    raise ValueError("The event should be inside f_funreco")

                grp_funreco = f_funreco[grp_name_gm]
                funreco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_funreco["funreco_triplet_idxs"]))
                funreco_gm_check = [idx in grp_gm.attrs["gm_triplet_idxs"][...] for idx in grp_funreco["funreco_triplet_idxs"][...]]

                if funreco_gm_right:
                    cor_funreco_gm.append(int(grp_name_gm))

                elif np.any(funreco_gm_check):
                    if np.allclose(grp_funreco["funreco_triplet_idxs"], [0, 0, 0]):
                        not_funreco_gm.append(int(grp_name_gm))
                
                    else:
                        par_funreco_gm.append(int(grp_name_gm))


                elif np.allclose(grp_funreco["funreco_triplet_idxs"], [0, 0, 0]):
                    not_funreco_gm.append(int(grp_name_gm))

                else:
                    wrong_funreco_gm.append(int(grp_name_gm))
                
                if grp_name_gm not in f_aiereco.keys():
                    raise ValueError("The event should be inside f_aiereco")
                
                grp_aiereco = f_aiereco[grp_name_gm]

                aiereco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_aiereco["aiereco_triplet_idxs"]))
                aiereco_gm_check = [idx in grp_gm.attrs["gm_triplet_idxs"][...] for idx in grp_aiereco["aiereco_triplet_idxs"][...]]

                if aiereco_gm_right:
                    cor_aiereco_gm.append(int(grp_name_gm))

                elif np.any(aiereco_gm_check):
                    if np.allclose(grp_aiereco["aiereco_triplet_idxs"], [0, 0, 0]):
                        not_aiereco_gm.append(int(grp_name_gm))

                    else:
                        par_aiereco_gm.append(int(grp_name_gm))

                elif np.allclose(grp_aiereco["aiereco_triplet_idxs"], [0, 0, 0]):
                    not_aiereco_gm.append(int(grp_name_gm))

                else:
                    wrong_aiereco_gm.append(int(grp_name_gm))


cor_funreco_gm.sort()
par_funreco_gm.sort()
wrong_funreco_gm.sort()
not_funreco_gm.sort()

cor_aiereco_gm.sort()
par_aiereco_gm.sort()
wrong_aiereco_gm.sort()
not_aiereco_gm.sort()

n_cor_funreco = len(cor_funreco_gm)
n_par_funreco = len(par_funreco_gm)
n_wrong_funreco = len(wrong_funreco_gm)
n_not_funreco = len(not_funreco_gm)

n_cor_aiereco = len(cor_aiereco_gm)
n_par_aiereco = len(par_aiereco_gm)
n_wrong_aiereco = len(wrong_aiereco_gm)
n_not_aiereco = len(not_aiereco_gm)

categories = ["correct", "partial", "wrong", "not_reco"]
values = {
    'x86 simulation': (n_cor_funreco, n_par_funreco, n_wrong_funreco, n_not_funreco),
    'AIE simulation': (n_cor_aiereco, n_par_aiereco, n_wrong_aiereco, n_not_aiereco),
}

x = np.arange(len(categories))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Triplet reco category')
ax.set_ylabel('Event counts')
ax.set_xticks(x + width / 2, categories)
ax.set_ylim(0, 300)
ax.legend(loc='upper left', ncols=2)

plt.show()