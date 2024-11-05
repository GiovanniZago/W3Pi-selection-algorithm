DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

MIN_PT  = 28 # 9
MED_PT  = 48 # 15
HIG_PT  = 60 # 20
PT_CONV = 0.25

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.width = 1000

def get_pt_group(pt):
    if pt >= HIG_PT * PT_CONV:
        return 2
    
    elif pt >= MED_PT * PT_CONV:
        return 1
    
    elif pt >= MIN_PT * PT_CONV:
        return 0
    
    else:
        return -1

with uproot.open(DATA_PATH + "l1Nano_WTo3Pion_PU200.root") as f_in:
    with uproot.recreate(DATA_PATH + "l1Nano_WTo3Pion_genmatched_PU200.root") as f_out:
        tree = f_in.get("Events")

        # FOR EACH EVENT extract the columns of the ttree and append them 
        # to an ak array. For example, the pdg_id array will be like
        # [pdg_id_event0_array, pdg_id_event1_array, ...]
        # So basically each key in branches is an array of arrays
        branches = tree.arrays()
        
        # metadata variables
        n_events = 50_000 # look at the keys of the correspondent hdf5 file
        n_gen_acceptance = 0
        n_gen_match = 0
        
        # genmatched_tree variables
        ev_idx_list = []
        n_puppi_list = []
        part_idxs_list = []
        pt_groups_list = []
        gen_mass_list = []

        for ev_idx in tqdm(range(n_events)):
            genpi_etas = branches["GenPi_eta"][ev_idx]
            genpi_pts = branches["GenPi_pt"][ev_idx]

            # check eta and pt acceptance condition
            if (not np.all(np.abs(genpi_etas) < 2.4)) or (not np.all(genpi_pts > 2)):
                continue
            
            n_gen_acceptance += 1

            # check Genmatch condition
            gen_idx = branches["Puppi_GenPiIdx"][ev_idx]
            idx0 = ak.where(gen_idx == 0)
            idx1 = ak.where(gen_idx == 1)
            idx2 = ak.where(gen_idx == 2)

            if (ak.count(idx0) > 0) and (ak.count(idx1) > 0)  and (ak.count(idx2) > 0):
                n_gen_match += 1

                n_puppi = branches["nPuppi"][ev_idx]
                pts = branches["Puppi_pt"][ev_idx]
                gen_mass = ak.to_numpy(branches["GenW_mass"][ev_idx]).item()
                part_idxs = [ak.to_numpy(idx0).item(), ak.to_numpy(idx1).item(), ak.to_numpy(idx2).item()]
                part_idxs.sort()

                pt_group0 = get_pt_group(pts[idx0])
                pt_group1 = get_pt_group(pts[idx1])
                pt_group2 = get_pt_group(pts[idx2])

                ev_idx_list.append(ev_idx)
                n_puppi_list.append(n_puppi)
                part_idxs_list.append(part_idxs)
                pt_groups_list.append([pt_group0, pt_group1, pt_group2])
                gen_mass_list.append(gen_mass)
            
        f_out["genmatched_tree"] = {"ev_idx": ak.Array(ev_idx_list), 
                                    "n_puppi": ak.Array(n_puppi_list), 
                                    "part_idxs": ak.Array(part_idxs_list), 
                                    "pt_groups": ak.Array(pt_groups_list), 
                                    "gen_mass": ak.Array(gen_mass_list)}
        
        f_out["metadata"] = {"n_events": ak.Array([n_events]), 
                             "n_gen_acceptance": ak.Array([n_gen_acceptance]), 
                             "n_gen_match": ak.Array([n_gen_match])}

                


