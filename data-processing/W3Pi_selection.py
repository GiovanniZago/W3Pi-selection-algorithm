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
PT_CONV        = 0.25
F_CONV         = np.pi / PI
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
        keys_subset = [keys[1498]]

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
            iso_mask = np.zeros(ev_size, dtype="int32") 

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
            # variables to place indexes
            index_vector = np.arange(ev_size, dtype="int32")
            angsep_idx_hig_med = np.zeros(ev_size, dtype="int32")
            angsep_idx_hig_target_min = np.zeros(ev_size, dtype="int32")
            angsep_idx_med_target_min = np.zeros(ev_size, dtype="int32")
            hig_target_idx = None
            med_target_idx = None
            min_target_idx = None

            # define the triplet and related relativistic variables
            triplet = [-1, -1, -1]
            charge = None
            invariant_mass = None

            # (1) angular separation between high and med
            for ii in range(ev_size):
                eta_cur = etas_med_pt[ii]
                phi_cur = phis_med_pt[ii]
                idx_cur = index_vector[ii]

                # the following is needed to verify that te angular separation happens between a 
                # particle that has passed the med pt cut and another one
                idx_to_append = 0 if med_pt_mask[idx_cur] == 0 else idx_cur+1

                d_eta = eta_cur - etas_hig_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_cur, phis_hig_pt[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                # update vector with indexes
                angsep_idx_hig_med = np.where(is_ge_mindr2, idx_to_append, angsep_idx_hig_med)

            # the following is needed to verify that te angular separation happens between a 
            # particle that has passed the high pt cut and another one
            angsep_idx_hig_med = np.where(hig_pt_mask, angsep_idx_hig_med, zeros_vector)

            # find the high pt target index
            foo = np.nonzero(angsep_idx_hig_med)[0]
        
            if (len(foo) > 0):
                # (2) angular separation between high pt target and min
                hig_target_idx = foo[0].astype("int32")
                eta_hig_pt_target = etas_hig_pt[hig_target_idx] * np.ones(ev_size, dtype="int32")
                phi_hig_pt_target = phis_hig_pt[hig_target_idx] * np.ones(ev_size, dtype="int32")

                d_eta = eta_hig_pt_target - etas_min_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_hig_pt_target[jj], phis_min_pt[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                # select the indexes of min_pt partices that are ang sep from the high pt particle
                # identified by idx_target1
                angsep_idx_hig_target_min = np.where(is_ge_mindr2, index_vector+1, angsep_idx_hig_target_min)
                
                # make sure that the min pt particles corresponging to the min pt indexes have passed
                # the min pt cut
                angsep_idx_hig_target_min = np.where(min_pt_mask, angsep_idx_hig_target_min, zeros_vector)

                # (3) angular separation between med pt target and min 
                med_target_idx = angsep_idx_hig_med[hig_target_idx] - 1 # minus 1 because we appended idx_cur+1
                
                eta_med_pt_target = etas_med_pt[med_target_idx] * np.ones(ev_size)
                phi_med_pt_target = phis_med_pt[med_target_idx] * np.ones(ev_size)

                d_eta = eta_med_pt_target - etas_min_pt # scalar - vector

                d_phi = np.zeros(ev_size)
                for jj in range(ev_size):
                    d_phi[jj] = ang_diff(phi_med_pt_target[jj], phis_min_pt[jj])

                dr2 = d_eta ** 2 + d_phi ** 2
                dr2 = dr2 * F_CONV2
                is_ge_mindr2 = dr2 >= MIN_DR2_ANGSEP

                # select the indexes of min_pt partices that are ang sep from the high pt particle
                # identified by idx_target1
                angsep_idx_med_target_min = np.where(is_ge_mindr2, index_vector+1, angsep_idx_med_target_min)
                
                # make sure that the min pt particles corresponging to the min pt indexes have passed
                # the min pt cut
                angsep_idx_med_target_min = np.where(min_pt_mask, angsep_idx_med_target_min, zeros_vector)

                # find a common element between the two arrays
                # the mask on > 0 elements is necessary to avoid the zero placeholders 
                # in case of no angular separation
                goo = np.intersect1d(angsep_idx_hig_target_min[angsep_idx_hig_target_min > 0], 
                                     angsep_idx_med_target_min[angsep_idx_med_target_min > 0])

                if len(goo) > 0:
                    min_target_idx = goo[0]
                    min_target_idx -= 1 # because we have inserted index_vector + 1 both in angsep_idx_hig_target_min and angsep_idx_med_target_min
                    triplet = [min_target_idx, med_target_idx, hig_target_idx]
                    charge = np.sign(pdg_ids[min_target_idx]) + np.sign(pdg_ids[med_target_idx]) + np.sign(pdg_ids[hig_target_idx])

                    if charge == 1:
                        mass1 = 0.13957039 if (np.abs(pdg_ids[min_target_idx]) > 0) else 0.1349768
                        px1 = pts[min_target_idx] * PT_CONV * np.cos(phis[min_target_idx] * F_CONV)
                        py1 = pts[min_target_idx] * PT_CONV * np.sin(phis[min_target_idx] * F_CONV)
                        pz1 = pts[min_target_idx] * PT_CONV * np.sinh(etas[min_target_idx] * F_CONV)
                        e1 = np.sqrt((pts[min_target_idx] * PT_CONV * np.cosh(etas[min_target_idx] * F_CONV)) ** 2 + mass1 ** 2)

                        mass2 = 0.13957039 if (np.abs(pdg_ids[min_target_idx]) > 0) else 0.1349768
                        px2 = pts[med_target_idx] * PT_CONV * np.cos(phis[med_target_idx] * F_CONV)
                        py2 = pts[med_target_idx] * PT_CONV * np.sin(phis[med_target_idx] * F_CONV)
                        pz2 = pts[med_target_idx] * PT_CONV * np.sinh(etas[med_target_idx] * F_CONV)
                        e2 = np.sqrt((pts[med_target_idx] * PT_CONV * np.cosh(etas[med_target_idx] * F_CONV)) ** 2 + mass2 ** 2)

                        mass3 = 0.13957039 if (np.abs(pdg_ids[min_target_idx]) > 0) else 0.1349768
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

                        if e_tot2 > p_tot2:
                            invariant_mass = np.sqrt(e_tot2 - p_tot2)

            if DEBUG:
                print("ANGULAR SEPARATION:")
                df_angsep = pd.DataFrame(data=np.vstack((angsep_idx_hig_med, angsep_idx_hig_target_min, angsep_idx_med_target_min)).T, 
                                         columns=["HIG/MED", f"HIG_TARGET({hig_target_idx})/MIN", f"MED_TARGET({med_target_idx})/MIN"], 
                                         dtype="int32")
                print(df_angsep)
                print("\n\n")

            print(f"Triplet at Event #{k}:   {triplet}      Tot charge = {charge}")

            if invariant_mass:
                print(f"Invariant mass = {invariant_mass} GeV")


            

if __name__ == "__main__":
    main()