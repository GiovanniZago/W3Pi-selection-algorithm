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
    tree = f_in.get("Events")

    # FOR EACH EVENT extract the columns of the ttree and append them 
    # to an ak array. For example, the pdg_id array will be like
    # [pdg_id_event0_array, pdg_id_event1_array, ...]
    # So basically each key in branches is an array of arrays
    branches = tree.arrays()
    
    # metadata variables
    n_events = 50_000 # look at the keys of the correspondent hdf5 file
    n_gen_acceptance = 0
    
    for ev_idx in tqdm(range(n_events)):
        GenPi_etas = branches["GenPi_eta"][ev_idx]
        GenPi_pts = branches["GenPi_pt"][ev_idx]

        if np.all(np.abs(GenPi_etas) < 2.4) and np.all(GenPi_pts > 2):
            n_gen_acceptance += 1

    print(n_gen_acceptance)

                


