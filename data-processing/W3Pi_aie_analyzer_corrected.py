import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep

hep.style.use("CMS")

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"
MASS_P = 0.13957039

total_aiereco = []
cor_aiereco_gm = []
par_aiereco_gm = []
not_aiereco_gm = []
wrong_aiereco_gm = []

cor_aiereco_mass = []
cor_gm_mass = []
cor_reco_mass = []
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

                pt = grp_gm["pt"][...]
                eta = grp_gm["eta"][...]
                phi = grp_gm["phi"][...]

                pt0 = pt[grp_aiereco["aiereco_triplet_idxs"][0]]
                pt1 = pt[grp_aiereco["aiereco_triplet_idxs"][1]]
                pt2 = pt[grp_aiereco["aiereco_triplet_idxs"][2]]

                eta0 = eta[grp_aiereco["aiereco_triplet_idxs"][0]]
                eta1 = eta[grp_aiereco["aiereco_triplet_idxs"][1]]
                eta2 = eta[grp_aiereco["aiereco_triplet_idxs"][2]]

                phi0 = phi[grp_aiereco["aiereco_triplet_idxs"][0]]
                phi1 = phi[grp_aiereco["aiereco_triplet_idxs"][1]]
                phi2 = phi[grp_aiereco["aiereco_triplet_idxs"][2]]

                px0 = pt0 * np.cos(phi0)
                py0 = pt0 * np.sin(phi0)
                pz0 = pt0 * np.sinh(eta0)
                e0 = np.sqrt(px0**2 + py0**2 + pz0**2 + MASS_P**2)

                px1 = pt1 * np.cos(phi1)
                py1 = pt1 * np.sin(phi1)
                pz1 = pt1 * np.sinh(eta1)
                e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + MASS_P**2)

                px2 = pt2 * np.cos(phi2)
                py2 = pt2 * np.sin(phi2)   
                pz2 = pt2 * np.sinh(eta2)
                e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + MASS_P**2)

                invariant_mass = np.sqrt((e0 + e1 + e2)**2 - (px0 + px1 + px2)**2 - (py0 + py1 + py2)**2 - (pz0 + pz1 + pz2)**2)
                cor_reco_mass.append(invariant_mass)

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
plt.hist(cor_reco_mass, bins=20, label="L1 Scouting, GEN-matched", alpha=0.5)
hep.cms.label(label="Phase-2 Simulation Preliminary", data=True);
plt.xlabel("Triplet invariant mass (GeV)")
plt.ylabel("Counts")
plt.legend(fontsize="16")
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