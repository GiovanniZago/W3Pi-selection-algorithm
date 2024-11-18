import numpy as np
import h5py
import pandas as pd
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

N_HIG          = 16
MIN_PT         = 28 # 7
MED_PT         = 48 # 15
HIG_PT         = 60 # 20
MIN_DR2_ANGSEP = 0.5 * 0.5
MIN_MASS       = 60 # 60
MIN_DR2        = 0.01 * 0.01
PI             = 720
MAX_MASS       = 100 # 100
MAX_DR2        = 0.25 * 0.25
MAX_ISO        = 0.5 
PT_CONV        = 0.25
F_CONV         = np.pi / PI
F_CONV2        = (np.pi / PI ) ** 2
MASS_P         = 0.13957039

with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.125X_v1.hdf5", "r") as f_in:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.125X_v1_reco.hdf5", "w") as f_out:
        ev_idxs = [int(key) for key in f_in.keys()]
        ev_idxs.sort()


        for ev_idx in tqdm(ev_idxs):
            grp_out = f_out.create_group(str(ev_idx))

            grp = f_in[str(ev_idx)]
            pts = grp["pt"][...]
            etas = grp["eta"][...]
            phis = grp["phi"][...]
            pdg_ids = grp["pdg_id"][...]
            
            min_pt_count = 0
            med_pt_count = 0
            hig_pt_count = 0
            is_filter = np.zeros(N_HIG, dtype=np.int16)
            is_filter_idx = 0

            for i, (pt, pdg_id) in enumerate(zip(pts, pdg_ids)):
                is_min_pt = pt >= MIN_PT * PT_CONV
                is_pdg_id = (np.abs(pdg_id) == 211) or (np.abs(pdg_id) == 11)
                
                if (is_filter_idx < N_HIG) and is_min_pt and is_pdg_id:
                    is_filter[is_filter_idx] = i + 1
                    is_filter_idx += 1
                    min_pt_count += 1
                    
                    if pt >= MED_PT * PT_CONV: med_pt_count += 1
                    if pt >= HIG_PT * PT_CONV: hig_pt_count += 1


            is_filter_mask = is_filter > 0
            skip_event = (min_pt_count < 3) or (med_pt_count < 2) or (hig_pt_count < 1)

            is_iso_filter = np.zeros_like(is_filter, dtype=np.int16)
            pts_iso_filter = np.zeros_like(is_filter, dtype=np.float32)
            etas_iso_filter = np.zeros_like(is_filter, dtype=np.float32)
            phis_iso_filter = np.zeros_like(is_filter, dtype=np.float32)
            pdg_ids_iso_filter = np.zeros_like(is_filter, dtype=np.int16)

            for i, idx in enumerate(is_filter):
                if skip_event or (not idx):
                    continue
                
                d_eta = etas[idx - 1] - etas
                d_phi = phis[idx - 1] - phis
                d_phi = np.where(d_phi < -np.pi, d_phi + 2 * np.pi, d_phi)
                d_phi = np.where(d_phi > np.pi, d_phi - 2 * np.pi, d_phi)
                dr2 = d_eta * d_eta + d_phi * d_phi

                dr2_mask = (dr2 >= (MIN_DR2 * F_CONV2)) & (dr2 <= (MAX_DR2 * F_CONV2))
                pt_sum = np.sum(pts[dr2_mask])

                if pt_sum <= (pts[idx - 1] * MAX_ISO):
                    is_iso_filter[i] = idx
                    pts_iso_filter[i] = pts[idx - 1]
                    etas_iso_filter[i] = etas[idx - 1]
                    phis_iso_filter[i] = phis[idx - 1]
                    pdg_ids_iso_filter[i] = pdg_ids[idx - 1]

            is_iso_filter_mask = is_iso_filter > 0
            skip_event = (np.sum(is_iso_filter_mask) < 3)

            triplet_idxs = np.zeros(3, dtype=np.int16)
            invariant_mass = 0
            w_mass = 0
            triplet_score = 0
            best_triplet_score = 0

            for i0, idx0 in enumerate(is_iso_filter):
                if skip_event or (not idx0):
                    continue

                pt_hig_pt_target0 = pts_iso_filter[i0]
                eta_hig_pt_target0 = etas_iso_filter[i0]
                phi_hig_pt_target0 = phis_iso_filter[i0]

                for i1, idx1 in enumerate(is_iso_filter):
                    if (i0 == i1) or (not idx1):
                        continue

                    d_eta = eta_hig_pt_target0 - etas_iso_filter[i1]
                    d_phi = phi_hig_pt_target0 - phis_iso_filter[i1]
                    d_phi = d_phi if (d_phi <= np.pi) else d_phi - 2 * np.pi
                    d_phi = d_phi if (d_phi >= -np.pi) else d_phi + 2 * np.pi
                    dr2 = d_eta * d_eta + d_phi * d_phi

                    if (dr2 < (MIN_DR2_ANGSEP * F_CONV2)):
                        continue

                    pt_hig_pt_target1 = pts_iso_filter[i1]
                    eta_hig_pt_target1 = etas_iso_filter[i1]
                    phi_hig_pt_target1 = phis_iso_filter[i1]

                    for i2, idx2 in enumerate(is_iso_filter):
                        if (i2 == i0) or (i2 == i1) or (not idx2):
                            continue

                        d_eta = eta_hig_pt_target1 - etas_iso_filter[i2]
                        d_phi = phi_hig_pt_target1 - phis_iso_filter[i2]
                        d_phi = d_phi if (d_phi <= np.pi) else d_phi - 2 * np.pi
                        d_phi = d_phi if (d_phi >= -np.pi) else d_phi + 2 * np.pi
                        dr2 = d_eta * d_eta + d_phi * d_phi

                        if (dr2 < (MIN_DR2_ANGSEP * F_CONV2)):
                            continue

                        pt_hig_pt_target2 = pts_iso_filter[i2]
                        eta_hig_pt_target2 = etas_iso_filter[i2]
                        phi_hig_pt_target2 = phis_iso_filter[i2]

                        charge0 = np.sign(pdg_ids_iso_filter[i0]) if np.abs(pdg_ids_iso_filter[i0]) == 211 else -np.sign(pdg_ids_iso_filter[i0])
                        charge1 = np.sign(pdg_ids_iso_filter[i1]) if np.abs(pdg_ids_iso_filter[i1]) == 211 else -np.sign(pdg_ids_iso_filter[i1])
                        charge2 = np.sign(pdg_ids_iso_filter[i2]) if np.abs(pdg_ids_iso_filter[i2]) == 211 else -np.sign(pdg_ids_iso_filter[i2])
                        charge_tot = charge0 + charge1 + charge2

                        if np.abs(charge_tot) != 1:
                            continue

                        px0 = pt_hig_pt_target0 * np.cos(phi_hig_pt_target0)
                        py0 = pt_hig_pt_target0 * np.sin(phi_hig_pt_target0)
                        pz0 = pt_hig_pt_target0 * np.sinh(eta_hig_pt_target0)
                        e0 = np.sqrt(px0 * px0 + py0 * py0 + pz0 * pz0 + MASS_P * MASS_P)

                        px1 = pt_hig_pt_target1 * np.cos(phi_hig_pt_target1)
                        py1 = pt_hig_pt_target1 * np.sin(phi_hig_pt_target1)
                        pz1 = pt_hig_pt_target1 * np.sinh(eta_hig_pt_target1)
                        e1 = np.sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + MASS_P * MASS_P)

                        px2 = pt_hig_pt_target2 * np.cos(phi_hig_pt_target2)
                        py2 = pt_hig_pt_target2 * np.sin(phi_hig_pt_target2)
                        pz2 = pt_hig_pt_target2 * np.sinh(eta_hig_pt_target2)
                        e2 = np.sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + MASS_P * MASS_P)

                        px_tot = px0 + px1 + px2
                        py_tot = py0 + py1 + py2
                        pz_tot = pz0 + pz1 + pz2
                        e_tot = e0 + e1 + e2

                        invariant_mass = np.sqrt(e_tot * e_tot - px_tot * px_tot - py_tot * py_tot - pz_tot * pz_tot)

                        if (invariant_mass < MIN_MASS) or (invariant_mass > MAX_MASS):
                            invariant_mass = 0
                            continue
                        
                        triplet_score = pt_hig_pt_target0 + pt_hig_pt_target1 + pt_hig_pt_target2

                        if triplet_score > best_triplet_score:
                            best_triplet_score = triplet_score
                            triplet_idxs[0] = idx0 - 1
                            triplet_idxs[1] = idx1 - 1
                            triplet_idxs[2] = idx2 - 1
                            w_mass = invariant_mass

            grp_out.create_dataset("reco_triplet_idxs", data=triplet_idxs, dtype=np.int16)
            grp_out.create_dataset("reco_w_mass", data=w_mass, dtype=np.float32)







