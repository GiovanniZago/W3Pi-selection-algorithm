import numpy as np
import h5py
import pandas as pd
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(legacy="1.25")

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.width = 1000

MIN_PT         = 28 # 9
MED_PT         = 48 # 15
HIG_PT         = 60 # 20
MIN_DR2_ANGSEP = 0.5 * 0.5
MIN_MASS       = 60 # 60
MIN_DR2        = 0.01 * 0.01
PI             = 720
MAX_MASS       = 100 # 100
MAX_DR2        = 0.25 * 0.25
MAX_ISO        = 0.5 # USUALLY 0.5
PT_CONV        = 0.25
F_CONV         = np.pi / PI
F_CONV2        = (np.pi / PI ) ** 2

P_BUNCHES = 13
V_SIZE = 8

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

VERBOSE = False
DEBUG_MASKS = False
START = False

def ang_diff(x, y):
    if (np.abs(x) > PI or np.abs(y) > PI):
        raise ValueError("The two inputs must be inside the interval [-PI, PI]")

    diff = x - y 
    if diff > PI:
        diff += (-2) * PI
    elif diff < -PI:
        diff += 2 * PI

    return diff 

def main():
    if not START:
        exit

    file = "PuppiSignal_104.hdf5"
    # file = "Puppi_104.hdf5"

    ev_idx_list = []
    part_idxs_list = []
    reco_mass_list = []

    with h5py.File(DATA_PATH + file, "r") as f:
        keys = [int(k) for k in f.keys()]
        keys.sort()
        keys_subset = [keys[1]]

        for k in tqdm(keys_subset):
            if VERBOSE or DEBUG_MASKS:
                print(f"EVENT #{k}")

            idx_dataset = str(k)
            event = f[idx_dataset][()]

            c = {"pdg_id": 0, "phi": 1, "eta": 2, "pt": 3}
            ev_size = f.attrs["ev_size"]

            pdg_ids      = event[:,c["pdg_id"]].astype("int32")
            phis         = event[:,c["phi"]].astype("int32")
            etas         = event[:,c["eta"]].astype("int32")
            pts          = event[:,c["pt"]].astype("int32")
            zeros_vector = np.zeros(ev_size, dtype="int32")

            # PRELIMINARY DATA FILTERING
            # find masks for minpt cut and pdg_id selection
            pdg_id_mask1 = np.abs(pdg_ids) == 211
            pdg_id_mask2 = np.abs(pdg_ids) == 11
            pdg_id_mask = pdg_id_mask1 | pdg_id_mask2

            # CALCULATE ISOLATION
            iso_mask = np.zeros(ev_size, dtype="int16") 

            for ii in range(ev_size):
                eta_cur = etas[ii]
                phi_cur = phis[ii]
                pt_cur = pts[ii]
                pt_sum = 0

                d_eta = eta_cur - etas

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_cur, phis[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 *= F_CONV2

                pts_mask = (dr2 >= MIN_DR2) & (dr2 <= MAX_DR2)
                pts_to_sum = np.where(pts_mask, pts, zeros_vector)
                pt_sum = np.sum(pts_to_sum)

                if pt_sum <= (MAX_ISO * pt_cur):
                    iso_mask[ii] = 1


            # combine masks and filter data based on **individual** characteristic
            filter_mask = pdg_id_mask & iso_mask

            pdg_ids = np.where(filter_mask, pdg_ids, zeros_vector)
            etas = np.where(filter_mask, etas, zeros_vector)
            phis = np.where(filter_mask, phis, zeros_vector)
            pts = np.where(filter_mask, pts, zeros_vector)

            # DIVIDE DATA INTO PT GROUPS
            # find masks for med pt and high pt
            min_pt_mask = pts >= MIN_PT
            med_pt_mask = pts >= MED_PT
            hig_pt_mask = pts >= HIG_PT

            # fix masks so each particle belongs only to one group
            min_pt_mask = np.logical_xor(min_pt_mask, med_pt_mask)
            med_pt_mask = np.logical_xor(med_pt_mask, hig_pt_mask)

            min_pt_mask = min_pt_mask.astype("int16")
            med_pt_mask = med_pt_mask.astype("int16")
            hig_pt_mask = hig_pt_mask.astype("int16")

            etas_min_pt = np.where(min_pt_mask, etas, zeros_vector).astype("int32")
            phis_min_pt = np.where(min_pt_mask, phis, zeros_vector).astype("int32")

            etas_med_pt = np.where(med_pt_mask, etas, zeros_vector).astype("int32")
            phis_med_pt = np.where(med_pt_mask, phis, zeros_vector).astype("int32")

            etas_hig_pt = np.where(hig_pt_mask, etas, zeros_vector).astype("int32")
            phis_hig_pt = np.where(hig_pt_mask, phis, zeros_vector).astype("int32")

            if DEBUG_MASKS:
                print("DATA AND MASKS")
                df_angsep = pd.DataFrame(data=np.vstack((pdg_ids, phis, etas, pts, pdg_id_mask, iso_mask, filter_mask, min_pt_mask, med_pt_mask, hig_pt_mask)).T, 
                                         columns=["DATA:PDG_ID", "DATA:PHIS", "DATA:ETAS", "DATA:PTS", "PDG_ID_MASK", "ISO_MASK", "PDG_ID & ISO MASK", "MIN_PT_MASK", "MED_PT_MASK", "HIGH_PT_MASK"], 
                                         dtype="int32")
                print(df_angsep)
                print("\n\n")

            # CALCULATE ANGULAR SEPARATION
            # variables to keep track of what's happening
            best_triplet_score = 0
            triplet_idxs = []
            reco_mass = 0

            n_hig_pt = np.sum(hig_pt_mask)

            for ii in range(n_hig_pt):
                # take the index of the next entry inside hig_pt_mask
                hig_target_idx0 = np.nonzero(hig_pt_mask)[0][ii].astype("int16")
                eta_hig_pt_target0 = etas_hig_pt[hig_target_idx0]
                phi_hig_pt_target0 = phis_hig_pt[hig_target_idx0]

                # create a new hig pt mask that excludes the selected hig pt particle
                hig_pt_mask_cur0 = np.copy(hig_pt_mask)
                hig_pt_mask_cur0[hig_target_idx0] = 0

                # calculate ang sep between the current hig pt particle and
                # all the other hig pt particles
                d_eta = eta_hig_pt_target0 - etas_hig_pt

                d_phi = np.zeros(ev_size)
                for yy in range(ev_size):
                    d_phi[yy] = ang_diff(phi_hig_pt_target0, phis_hig_pt[yy])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                angsep0 = np.where(hig_pt_mask_cur0, is_ge_mindr2, zeros_vector)
                n_angsep0 = np.sum(angsep0)

                if n_angsep0 < 2:
                    continue

                for jj in range(n_angsep0):
                    # take the index of the next entry inside hig_pt_mask
                    hig_target_idx1 = np.nonzero(angsep0)[0][jj].astype("int16")
                    eta_hig_pt_target1 = etas_hig_pt[hig_target_idx1]
                    phi_hig_pt_target1 = phis_hig_pt[hig_target_idx1]
                    # create a new hig pt mask that excludes the selected hig pt par
                    hig_pt_mask_cur1 = np.copy(angsep0)
                    hig_pt_mask_cur1[hig_target_idx1] = 0

                    # calculate ang sep between the current hig pt particle and
                    # all the other hig pt particles
                    d_eta = eta_hig_pt_target1 - etas_hig_pt
                    d_phi = np.zeros(ev_size)
                    for yy in range(ev_size):
                        d_phi[yy] = ang_diff(phi_hig_pt_target1, phis_hig_pt[yy])

                    dr2 = d_eta ** 2 + d_phi ** 2
                    dr2 = dr2 * F_CONV2
                    is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                    angsep1 = np.where(hig_pt_mask_cur1, is_ge_mindr2, zeros_vector)
                    n_angsep1 = np.sum(angsep1)

                    if n_angsep1 < 1:
                        continue

                    for kk in range(n_angsep1):
                        # take the index of the next entry inside hig_pt_mask
                        hig_target_idx2 = np.nonzero(angsep1)[0][kk].astype("int16")
                        eta_hig_pt_target2 = etas_hig_pt[hig_target_idx2]
                        phi_hig_pt_target2 = phis_hig_pt[hig_target_idx2]

                        d_eta = eta_hig_pt_target2 - eta_hig_pt_target0 # now it's a scalar
                        d_phi = ang_diff(phi_hig_pt_target2, phi_hig_pt_target0) # now it's a scalar

                        dr2 = d_eta ** 2 + d_phi ** 2
                        dr2 = dr2 * F_CONV2
                        is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                        if not is_ge_mindr2:
                            continue

                        charges = [np.sign(pdg_ids[p_idx]) if np.abs(pdg_ids[p_idx]) == 211 else -np.sign(pdg_ids[p_idx]) for p_idx in [hig_target_idx0, hig_target_idx1, hig_target_idx2]]
                        abs_charge = np.abs(np.sum(charges))

                        if abs_charge == 1:
                            mass1 = 0.13957039 if (np.abs(charges[0]) > 0) else 0.1349768
                            px1 = pts[hig_target_idx0] * PT_CONV * np.cos(phis[hig_target_idx0] * F_CONV)
                            py1 = pts[hig_target_idx0] * PT_CONV * np.sin(phis[hig_target_idx0] * F_CONV)
                            pz1 = pts[hig_target_idx0] * PT_CONV * np.sinh(etas[hig_target_idx0] * F_CONV)
                            e1 = np.sqrt(px1 ** 2 + py1 ** 2 + pz1 ** 2 + mass1 ** 2)

                            mass2 = 0.13957039 if (np.abs(charges[1]) > 0) else 0.1349768
                            px2 = pts[hig_target_idx1] * PT_CONV * np.cos(phis[hig_target_idx1] * F_CONV)
                            py2 = pts[hig_target_idx1] * PT_CONV * np.sin(phis[hig_target_idx1] * F_CONV)
                            pz2 = pts[hig_target_idx1] * PT_CONV * np.sinh(etas[hig_target_idx1] * F_CONV)
                            e2 = np.sqrt(px2 ** 2 + py2 ** 2 + pz2 ** 2 + mass2 ** 2)

                            mass3 = 0.13957039 if (np.abs(charges[2]) > 0) else 0.1349768
                            px3 = pts[hig_target_idx2] * PT_CONV * np.cos(phis[hig_target_idx2] * F_CONV)
                            py3 = pts[hig_target_idx2] * PT_CONV * np.sin(phis[hig_target_idx2] * F_CONV)
                            pz3 = pts[hig_target_idx2] * PT_CONV * np.sinh(etas[hig_target_idx2] * F_CONV)
                            e3 = np.sqrt(px3 ** 2 + py3 ** 2 + pz3 ** 2 + mass3 ** 2)

                            px_tot = px1 + px2 + px3
                            py_tot = py1 + py2 + py3
                            pz_tot = pz1 + pz2 + pz3
                            e_tot = e1 + e2 + e3

                            e_tot2 = e_tot ** 2
                            p_tot2 = px_tot ** 2 + py_tot ** 2 + pz_tot ** 2

                            invariant_mass = np.sqrt(e_tot2 - p_tot2)

                            if (invariant_mass >= MIN_MASS) and (invariant_mass <= MAX_MASS):
                                triplet_score = pts[hig_target_idx0] + pts[hig_target_idx1] + pts[hig_target_idx2]

                                if triplet_score > best_triplet_score:
                                    best_triplet_score = triplet_score
                                    triplet_idxs = [hig_target_idx0, hig_target_idx1, hig_target_idx2]
                                    reco_mass = invariant_mass

            if (len(triplet_idxs) > 0):
                ev_idx_list.append(k)
                part_idxs_list.append(triplet_idxs)
                reco_mass_list.append(reco_mass)    




if __name__ == "__main__":
    main()