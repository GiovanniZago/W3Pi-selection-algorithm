import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep

hep.style.use("CMS")

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

total_funreco = []
cor_funreco_gm = []
par_funreco_gm = []
not_funreco_gm = []
wrong_funreco_gm = []

cor_funreco_mass = []
cor_gm_mass = []
par_funreco_mass = []
par_gm_mass = []

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_funreco.hdf5", "r") as f_funreco:
        for (grp_name_gm, grp_gm) in tqdm(f_gm.items()):
            if grp_name_gm not in f_funreco.keys():
                raise ValueError("The event should be inside f_funreco")
            
            grp_funreco = f_funreco[grp_name_gm]
            
            if grp_funreco["funreco_w_mass"][...].item() > 0:
                total_funreco.append(int(grp_name_gm))


            if (grp_gm.attrs["is_acc"] != 1) or (grp_gm.attrs["is_gm"] != 1):
                continue

            funreco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_funreco["funreco_triplet_idxs"]))
            funreco_gm_check = [idx in grp_gm.attrs["gm_triplet_idxs"][...] for idx in grp_funreco["funreco_triplet_idxs"][...]]

            if funreco_gm_right:
                cor_funreco_gm.append(int(grp_name_gm))
                cor_funreco_mass.append(grp_funreco["funreco_w_mass"][...].item())
                cor_gm_mass.append(grp_gm["gen_w_mass"][...].item())

            elif np.any(funreco_gm_check):
                if np.allclose(grp_funreco["funreco_triplet_idxs"], [0, 0, 0]):
                    not_funreco_gm.append(int(grp_name_gm))
                
                else:
                    par_funreco_gm.append(int(grp_name_gm))
                    par_funreco_mass.append(grp_funreco["funreco_w_mass"][...].item())
                    par_gm_mass.append(grp_gm["gen_w_mass"][...].item())

            elif np.allclose(grp_funreco["funreco_triplet_idxs"], [0, 0, 0]):
                not_funreco_gm.append(int(grp_name_gm))

            else:
                wrong_funreco_gm.append(int(grp_name_gm))

cor_funreco_gm.sort()
par_funreco_gm.sort()
wrong_funreco_gm.sort()
not_funreco_gm.sort()

n_cor = len(cor_funreco_gm)
n_par = len(par_funreco_gm)
n_wrong = len(wrong_funreco_gm)
n_not = len(not_funreco_gm)

print(len(total_funreco))
print(len(cor_funreco_mass))
print(len(cor_gm_mass))
print(len(par_funreco_mass))
print(len(par_gm_mass))

plt.figure()
barplot = plt.bar(["correct", "partial", "wrong", "not_reco"], [n_cor, n_par, n_wrong, n_not])
plt.bar_label(barplot, labels=[n_cor, n_par, n_wrong, n_not])
plt.xlabel("Triplet reco category")
plt.ylabel("Event counts")
plt.ylim(0, 6000)
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(cor_funreco_mass, bins=20, label="x86 simulation", alpha=0.7)
plt.hist(cor_gm_mass, bins=25, label="dataset", alpha=0.5)
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(par_funreco_mass, bins=20, label="x86 simulation", alpha=0.7)
plt.hist(par_gm_mass, bins=25, label="dataset", alpha=0.5)
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()