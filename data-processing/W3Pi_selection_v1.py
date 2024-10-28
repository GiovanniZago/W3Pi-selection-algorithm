import numpy as np
import h5py
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

# warnings.simplefilter("error")

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.width = 1000

MIN_PT         = 7 # 9
MED_PT         = 12 # 15
HIG_PT         = 15 # 20
MIN_DR2_ANGSEP = 0.5 * 0.5
MIN_MASS       = 60 # 60
MIN_DR2        = 0.01 * 0.01
PI             = 720
MAX_MASS       = 100 # 100
MAX_DR2        = 0.25 * 0.25
MAX_ISO        = 0.5 # 2.0 or 0.4
PT_CONV        = 0.25
F_CONV         = np.pi / PI
F_CONV2        = (np.pi / PI ) ** 2

P_BUNCHES = 13
V_SIZE = 8

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

VERBOSE = True
DEBUG_MASKS = False

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
    file = "PuppiSignal_104.hdf5"
    # file = "Puppi_104.hdf5"

    with h5py.File(DATA_PATH + file, "r") as f:
        keys = [int(k) for k in f.keys()]
        keys.sort()
        keys_subset = keys[0:20_000]

        invariant_masses = []
        n_filtered_triplets = []

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
                eta_med_pt_target = etas[ii]
                phi_med_pt_target = phis[ii]
                pt_cur = pts[ii]
                pt_sum = 0

                d_eta = eta_med_pt_target - etas
            
                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_med_pt_target, phis[jj])

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
            # variables to place indexes
            hig_target_idx = None
            med_target_idx = None
            min_target_idx = None

            # define the triplet and related relativistic variables
            triplets = []
            charge = None
            invariant_mass = None
            n_filtered = 0

            # check all the possible combinations between high pt and med pt
            n_hig_pt = np.sum(hig_pt_mask)

            # each entry contains a list where there may be the index of the current high pt particle
            # in correspondence of the position marked by the index of the medpt particle
            # the same for the other two arrays later on the code
            angsep_idx_med_hig_array = np.zeros((n_hig_pt, ev_size), dtype="int16")

            for ii in range(n_hig_pt):
                # take the index of the next entry inside hig_pt_mask
                hig_target_idx = np.nonzero(hig_pt_mask)[0][ii].astype("int16")
                eta_hig_pt_target = etas_hig_pt[hig_target_idx]
                phi_hig_pt_target = phis_hig_pt[hig_target_idx]

                d_eta = eta_hig_pt_target - etas_med_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for yy in range(ev_size):
                    d_phi[yy] = ang_diff(phi_hig_pt_target, phis_med_pt[yy])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                # update vector with indexes
                angsep_idx_med_hig_array[ii] = np.where(is_ge_mindr2, 1, angsep_idx_med_hig_array[ii])

                # the following is needed to verify that te angular separation happens between a 
                # particle that has passed the med pt cut and another one
                angsep_idx_med_hig_array[ii] = np.where(med_pt_mask, angsep_idx_med_hig_array[ii], zeros_vector)

                # find how many ang sep med pt particles there are for the given high pt particle
                n_medpt_per_higpt = len(np.nonzero(angsep_idx_med_hig_array[ii])[0])
                angsep_idx_min_med_array = np.zeros((n_medpt_per_higpt, ev_size), dtype="int16")

                for jj in range(n_medpt_per_higpt):
                    med_target_idx = np.nonzero(angsep_idx_med_hig_array[ii])[0][jj].astype("int16")
                    eta_med_pt_target = etas_med_pt[med_target_idx]
                    phi_med_pt_target = phis_med_pt[med_target_idx]

                    d_eta = eta_med_pt_target - etas_min_pt # scalar - vector

                    d_phi = np.zeros(ev_size)
                    for yy in range(ev_size):
                        d_phi[yy] = ang_diff(phi_med_pt_target, phis_min_pt[yy])

                    dr2 = d_eta ** 2 + d_phi ** 2
                    dr2 = dr2 * F_CONV2
                    is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                    # select the indexes of min_pt partices that are ang sep from the high pt particle
                    # identified by idx_target1
                    angsep_idx_min_med_array[jj] = np.where(is_ge_mindr2, 1, angsep_idx_min_med_array[jj])

                    # make sure that the min pt particles corresponging to the min pt indexes have passed
                    # the min pt cut
                    angsep_idx_min_med_array[jj] = np.where(min_pt_mask, angsep_idx_min_med_array[jj], zeros_vector)

                    # find how many ang sep min pt particles there are for the given med pt particle
                    n_minpt_per_medpt = len(np.nonzero(angsep_idx_min_med_array[jj])[0])
                    angsep_idx_min_hig_array = np.zeros((n_minpt_per_medpt, ev_size), dtype="int16")

                    for kk in range(n_minpt_per_medpt):
                        min_target_idx = np.nonzero(angsep_idx_min_med_array[jj])[0][kk].astype("int16")
                        eta_min_pt_target = etas_min_pt[min_target_idx]
                        phi_min_pt_target = phis_min_pt[min_target_idx]

                        d_eta = eta_min_pt_target - eta_hig_pt_target # scalar - scalar
                        d_phi = ang_diff(phi_min_pt_target, phi_hig_pt_target)

                        dr2 = d_eta ** 2 + d_phi ** 2
                        dr2 = dr2 * F_CONV2
                        is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP # now is_ge_mindr2 is a scalar!!

                        if is_ge_mindr2:
                            charge = np.sign(pdg_ids[min_target_idx]) + np.sign(pdg_ids[med_target_idx]) + np.sign(pdg_ids[hig_target_idx])
                            abs_charge = np.abs(charge)

                            if abs_charge == 1:
                                n_filtered += 1

                                mass1 = 0.13957039 if (np.abs(pdg_ids[min_target_idx]) > 0) else 0.1349768
                                px1 = pts[min_target_idx] * PT_CONV * np.cos(phis[min_target_idx] * F_CONV)
                                py1 = pts[min_target_idx] * PT_CONV * np.sin(phis[min_target_idx] * F_CONV)
                                pz1 = pts[min_target_idx] * PT_CONV * np.sinh(etas[min_target_idx] * F_CONV)
                                e1 = np.sqrt((pts[min_target_idx] * PT_CONV * np.cosh(etas[min_target_idx] * F_CONV)) ** 2 + mass1 ** 2)

                                mass2 = 0.13957039 if (np.abs(pdg_ids[med_target_idx]) > 0) else 0.1349768
                                px2 = pts[med_target_idx] * PT_CONV * np.cos(phis[med_target_idx] * F_CONV)
                                py2 = pts[med_target_idx] * PT_CONV * np.sin(phis[med_target_idx] * F_CONV)
                                pz2 = pts[med_target_idx] * PT_CONV * np.sinh(etas[med_target_idx] * F_CONV)
                                e2 = np.sqrt((pts[med_target_idx] * PT_CONV * np.cosh(etas[med_target_idx] * F_CONV)) ** 2 + mass2 ** 2)

                                mass3 = 0.13957039 if (np.abs(pdg_ids[hig_target_idx]) > 0) else 0.1349768
                                px3 = pts[hig_target_idx] * PT_CONV * np.cos(phis[hig_target_idx] * F_CONV)
                                py3 = pts[hig_target_idx] * PT_CONV * np.sin(phis[hig_target_idx] * F_CONV)
                                pz3 = pts[hig_target_idx] * PT_CONV * np.sinh(etas[hig_target_idx] * F_CONV)
                                e3 = np.sqrt((pts[hig_target_idx] * PT_CONV * np.cosh(etas[hig_target_idx] * F_CONV)) ** 2 + mass3 ** 2)

                                px_tot = px1 + px2 + px3
                                py_tot = py1 + py2 + py3
                                pz_tot = pz1 + pz2 + pz3
                                e_tot = e1 + e2 + e3

                                e_tot2 = e_tot ** 2
                                p_tot2 = px_tot ** 2 + py_tot ** 2 + pz_tot ** 2

                                invariant_mass = np.sqrt(e_tot2 - p_tot2)
                                invariant_masses.append(invariant_mass)

                                # triplets.append(([min_target_idx, med_target_idx, hig_target_idx], invariant_mass))
                
            n_filtered_triplets.append(n_filtered
                                       )
            if VERBOSE:
                print(f"# filtered triplets = {n_filtered}")
                print(f"# min_pt = {np.sum(min_pt_mask)}")
                print(f"# med_pt = {np.sum(med_pt_mask)}")
                print(f"# hig_pt = {np.sum(hig_pt_mask)}")
                print("\n")
        
        plt.hist(invariant_masses, bins=20)
        plt.show()


                


                


            

if __name__ == "__main__":
    main()