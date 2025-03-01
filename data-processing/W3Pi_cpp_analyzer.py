import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep

hep.style.use("CMS")

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

total_aiereco = []
cor_cppreco_gm = []
par_cppreco_gm = []
not_cppreco_gm = []
wrong_cppreco_gm = []

cor_cppreco_mass = []
cor_gm_mass = []
par_cppreco_mass = []
par_gm_mass = []

evt_list = np.arange(2000)

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_cppreco.hdf5", "r") as f_cppreco:
        for (grp_name_gm, grp_gm) in tqdm(f_gm.items()):
            if int(grp_name_gm) not in evt_list:
                continue

            if grp_name_gm not in f_cppreco.keys():
                raise ValueError("The event should be inside f_aiereco")

            grp_cppreco = f_cppreco[grp_name_gm]
            
            if grp_cppreco["cppreco_w_mass"][...].item() > 0:
                total_aiereco.append(int(grp_name_gm))

            if (grp_gm.attrs["is_acc"] != 1) or (grp_gm.attrs["is_gm"] != 1):
                continue


            cppreco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_cppreco["cppreco_triplet_idxs"]))
            cppreco_gm_check = [idx in grp_gm.attrs["gm_triplet_idxs"][...] for idx in grp_cppreco["cppreco_triplet_idxs"][...]]

            if cppreco_gm_right:
                cor_cppreco_gm.append(int(grp_name_gm))
                cor_cppreco_mass.append(grp_cppreco["cppreco_w_mass"][...].item())
                cor_gm_mass.append(grp_gm["gen_w_mass"][...].item())

            elif np.any(cppreco_gm_check):
                if np.allclose(grp_cppreco["cppreco_triplet_idxs"], [0, 0, 0]):
                    not_cppreco_gm.append(int(grp_name_gm))
                
                else:
                    par_cppreco_gm.append(int(grp_name_gm))
                    par_cppreco_mass.append(grp_cppreco["cppreco_w_mass"][...].item())
                    par_gm_mass.append(grp_gm["gen_w_mass"][...].item())

            elif np.allclose(grp_cppreco["cppreco_triplet_idxs"], [0, 0, 0]):
                not_cppreco_gm.append(int(grp_name_gm))

            else:
                wrong_cppreco_gm.append(int(grp_name_gm))

cor_cppreco_gm.sort()
par_cppreco_gm.sort()
wrong_cppreco_gm.sort()
not_cppreco_gm.sort()

n_cor = len(cor_cppreco_gm)
n_par = len(par_cppreco_gm)
n_wrong = len(wrong_cppreco_gm)
n_not = len(not_cppreco_gm)

print(len(total_aiereco))
print(len(cor_cppreco_mass))
print(len(cor_gm_mass))
print(len(par_cppreco_mass))
print(len(par_gm_mass))

plt.figure()
barplot = plt.bar(["correct", "partial", "wrong", "not_reco"], [n_cor, n_par, n_wrong, n_not])
plt.bar_label(barplot, labels=[n_cor, n_par, n_wrong, n_not])
plt.xlabel("Triplet reco category")
plt.ylabel("Event counts")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(cor_cppreco_mass, bins=20, label="C++ simulation", alpha=0.7)
plt.hist(cor_gm_mass, bins=25, label="dataset", alpha=0.5)
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(par_cppreco_mass, bins=20, label="C++ simulation", alpha=0.7)
plt.hist(par_gm_mass, bins=25, label="dataset", alpha=0.5)
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()