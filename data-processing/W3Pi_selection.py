import numpy as np
import h5py
import pandas as pd

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.width = 1000

MIN_PT         = 7 # 9
MED_PT         = 12 # 15
HIG_PT         = 15 # 20
MIN_DR2_ANGSEP = 0.5 * 0.5
MIN_MASS       = 40 # 60
MIN_DR2        = 0.01 * 0.01
PI             = 720
MAX_MASS       = 150 # 100
MAX_DR2        = 0.25 * 0.25
MAX_ISO        = 0.5 # 2.0 or 0.4
F_CONV2        = (np.pi / PI ) ** 2

P_BUNCHES = 13
V_SIZE = 8

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

DEBUG = True

def ang_diff(x, y):
    if (np.abs(x) > PI or np.abs(y) > PI):
        raise ValueError("The two inputs must be inside the interval [-PI, PI]")

    diff = x - y 
    if diff > PI:
        diff += (-2) * PI
    elif diff < -PI:
        diff += 2 * PI

    return diff 

def isolation(p_index: int, n_particles: int, etas: np.ndarray, phis: np.ndarray, pts: np.ndarray):
    is_isolated = 0
    pt_sum = 0

    eta_cur = etas[p_index]
    phi_cur = phis[p_index]
    pt_cur = pts[p_index]

    for ii in range(n_particles):
        if (p_index == ii):
            continue
        
        d_eta = eta_cur - etas[ii]
        d_phi = ang_diff(phi_cur, phis[ii])
        d_r2 = d_eta * d_eta + d_phi * d_phi
        d_r2 *= F_CONV2 # conversion to floating point

        if (d_r2 >= MIN_DR2) and (d_r2 <= MAX_DR2):
            pt_sum = pt_sum + pts[ii]

    if pt_sum <= (MAX_ISO * pt_cur):
        is_isolated = 1

    return is_isolated

def main():
    file = "Puppi_fix104mod1.hdf5"

    with h5py.File(DATA_PATH + file, "r") as f:
        keys = [int(k) for k in f.keys()]
        keys.sort()
        keys_subset = [keys[1563]]

        for k in keys_subset:
            if DEBUG:
                print(f"EVENT #{k}")
                print("\n")

            idx_dataset = str(k)
            event = f[idx_dataset][()]

            c = {"pdg_id": 0, "phi": 1, "eta": 2, "pt": 3}
            ev_size = f.attrs["ev_size"]

            pdg_ids = event[:,c["pdg_id"]]
            phis = event[:,c["phi"]]
            etas = event[:,c["eta"]]
            pts = event[:,c["pt"]]
            zeros_vector = np.zeros(ev_size, dtype="int32")

            # PRELIMINARY DATA FILTERING
            # find masks for minpt cut and pdg_id selection
            pdg_id_mask1 = np.abs(pdg_ids) == 211
            pdg_id_mask2 = np.abs(pdg_ids) == 11
            pdg_id_mask = pdg_id_mask1 | pdg_id_mask2

            # CALCULATE ISOLATION
            iso_mask = np.zeros(ev_size, dtype="int") 

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

            etas_min_pt = np.where(min_pt_mask, etas, zeros_vector)
            phis_min_pt = np.where(min_pt_mask, phis, zeros_vector)

            etas_med_pt = np.where(med_pt_mask, etas, zeros_vector)
            phis_med_pt = np.where(med_pt_mask, phis, zeros_vector)

            etas_hig_pt = np.where(hig_pt_mask, etas, zeros_vector)
            phis_hig_pt = np.where(hig_pt_mask, phis, zeros_vector)

            if DEBUG:
                print("DATA AND MASKS")
                df_angsep = pd.DataFrame(data=np.vstack((pdg_ids, phis, etas, pts, pdg_id_mask, iso_mask, filter_mask, min_pt_mask, med_pt_mask, hig_pt_mask)).T, 
                                         columns=["DATA:PDG_ID", "DATA:PHIS", "DATA:ETAS", "DATA:PTS", "PDG_ID_MASK", "ISO_MASK", "PDG_ID & ISO MASK", "MIN_PT_MASK", "MED_PT_MASK", "HIGH_PT_MASK"], 
                                         dtype="int32")
                print(df_angsep)
                print("\n\n")

            # CALCULATE ANGULAR SEPARATION
            index_vector = np.arange(ev_size)
            angsep_idx_med_min = np.zeros(ev_size, dtype="int32")
            angsep_idx_hig_min = np.zeros(ev_size, dtype="int32")

            for ii in range(ev_size):
                eta_cur = etas_min_pt[ii]
                phi_cur = phis_min_pt[ii]
                idx_cur = index_vector[ii]

                # the following is needed to verify that te angular separation happens between a 
                # particle that has passed the min_pt cut and another one
                idx_to_append = 0 if min_pt_mask[idx_cur] == 0 else idx_cur+1

                # (1) angular separation between med and min
                d_eta = eta_cur - etas_med_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_cur, phis_med_pt[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                # update vector with indexes
                angsep_idx_med_min = np.where(is_ge_mindr2, idx_to_append, angsep_idx_med_min)

                # (2) angular separation between high and min
                d_eta = eta_cur - etas_hig_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_cur, phis_hig_pt[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP
                
                # update vector with indexes
                angsep_idx_hig_min = np.where(is_ge_mindr2, idx_to_append, angsep_idx_hig_min)
            

            # the following is needed to verify that te angular separation happens between a 
            # particle that has passed the med_pt cut and another one
            angsep_idx_med_min = np.where(med_pt_mask, angsep_idx_med_min, zeros_vector)

            # the following is needed to verify that te angular separation happens between a 
            # particle that has passed the high_pt cut and another one
            angsep_idx_hig_min = np.where(hig_pt_mask, angsep_idx_hig_min, zeros_vector)

            # (3) angular separation between high and med
            angsep_idx_hig_med = np.zeros(ev_size, dtype="int32")

            for ii in range(ev_size):
                eta_cur = etas_med_pt[ii]
                phi_cur = phis_med_pt[ii]
                idx_cur = index_vector[ii]

                # angular separation between med and min
                d_eta = eta_cur - etas_hig_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_cur, phis_hig_pt[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                # the following is needed to verify that te angular separation happens between a 
                # particle that has passed the med_pt cut and another one
                idx_to_append = 0 if med_pt_mask[idx_cur] == 0 else idx_cur+1

                angsep_idx_hig_med = np.where(is_ge_mindr2, idx_to_append, angsep_idx_hig_med)
            
            # the following is needed to verify that te angular separation happens between a 
            # particle that has passed the high_pt cut and another one
            angsep_idx_hig_med = np.where(hig_pt_mask, angsep_idx_hig_med, zeros_vector)

            if DEBUG:
                print("ANGULAR SEPARATION:")
                df_angsep = pd.DataFrame(data=np.vstack((angsep_idx_med_min, angsep_idx_hig_min, angsep_idx_hig_med)).T, 
                                         columns=["MED/MIN", "HIGH/MIN", "HIGH/MED"], 
                                         dtype="int32")
                print(df_angsep)
                print("\n\n")

            # test1 = np.any(angsep_idx_med_min != 0)
            # test2 = np.any(angsep_idx_hig_min != 0)
            # test3 = np.any(angsep_idx_hig_med != 0)

            # if (test1 and test2 and test3):
            #     print(f"Event #{k} tests successful")

            triplet = []
            for ii in range(ev_size):
                idx_cur = index_vector[ii]

                is_idx_eq1 = angsep_idx_hig_min == (idx_cur+1)
                idx_target1 = np.nonzero(is_idx_eq1)[0]

                if len(idx_target1) == 0:
                    continue
                
                idx_target1 = idx_target1[0]
                idx_target2 = angsep_idx_hig_med[idx_target1]
                idx_target2 -= 1

                if angsep_idx_med_min[idx_target2] == idx_cur:
                    triplet = np.array([idx_cur, idx_target2, idx_target1])
                    break

            if len(triplet) > 0:
                print(f"Triplet at Event #{k}:   {triplet}")
            

if __name__ == "__main__":
    main()