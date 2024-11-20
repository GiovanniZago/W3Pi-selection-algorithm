import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep

hep.style.use("CMS")

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

total_aiereco = []
cor_aiereco_gm = []
par_aiereco_gm = []
not_aiereco_gm = []
wrong_aiereco_gm = []

cor_aiereco_mass = []
cor_gm_mass = []
par_aiereco_mass = []
par_gm_mass = []

evt_list = np.arange(2000)

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "r") as f_gm:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_aiereco.hdf5", "r") as f_aiereco:
        for (grp_name_gm, grp_gm) in tqdm(f_gm.items()):
            if int(grp_name_gm) not in evt_list:
                continue

            if grp_name_gm not in f_aiereco.keys():
                raise ValueError("The event should be inside f_aiereco")

            grp_aiereco = f_aiereco[grp_name_gm]
            
            if grp_aiereco["aiereco_w_mass"][...].item() > 0:
                total_aiereco.append(int(grp_name_gm))

            if (grp_gm.attrs["is_acc"] != 1) or (grp_gm.attrs["is_gm"] != 1):
                continue


            aiereco_gm_right = np.allclose(np.sort(grp_gm.attrs["gm_triplet_idxs"]), np.sort(grp_aiereco["aiereco_triplet_idxs"]))
            aiereco_gm_check = [idx in grp_gm.attrs["gm_triplet_idxs"][...] for idx in grp_aiereco["aiereco_triplet_idxs"][...]]

            if aiereco_gm_right:
                cor_aiereco_gm.append(int(grp_name_gm))
                cor_aiereco_mass.append(grp_aiereco["aiereco_w_mass"][...].item())
                cor_gm_mass.append(grp_gm["gen_w_mass"][...].item())

            elif np.any(aiereco_gm_check):
                if np.allclose(grp_aiereco["aiereco_triplet_idxs"], [0, 0, 0]):
                    not_aiereco_gm.append(int(grp_name_gm))
                
                else:
                    par_aiereco_gm.append(int(grp_name_gm))
                    par_aiereco_mass.append(grp_aiereco["aiereco_w_mass"][...].item())
                    par_gm_mass.append(grp_gm["gen_w_mass"][...].item())

            elif np.allclose(grp_aiereco["aiereco_triplet_idxs"], [0, 0, 0]):
                not_aiereco_gm.append(int(grp_name_gm))

            else:
                wrong_aiereco_gm.append(int(grp_name_gm))

cor_aiereco_gm.sort()
par_aiereco_gm.sort()
wrong_aiereco_gm.sort()
not_aiereco_gm.sort()

n_cor = len(cor_aiereco_gm)
n_par = len(par_aiereco_gm)
n_wrong = len(wrong_aiereco_gm)
n_not = len(not_aiereco_gm)

print(len(total_aiereco))
print(len(cor_aiereco_mass))
print(len(cor_gm_mass))
print(len(par_aiereco_mass))
print(len(par_gm_mass))

plt.figure()
barplot = plt.bar(["correct", "partial", "wrong", "not_reco"], [n_cor, n_par, n_wrong, n_not])
plt.bar_label(barplot, labels=[n_cor, n_par, n_wrong, n_not])
plt.xlabel("Triplet reco category")
plt.ylabel("Event counts")
plt.ylim(0, 300)
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(cor_aiereco_mass, bins=20, label="AIE simulation", alpha=0.7)
plt.hist(cor_gm_mass, bins=25, label="dataset", alpha=0.5)
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(par_aiereco_mass, bins=20, label="AIE simulation", alpha=0.7)
plt.hist(par_gm_mass, bins=25, label="dataset", alpha=0.5)
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()