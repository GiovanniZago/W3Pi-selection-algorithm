DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

MIN_PT  = 7 # 9
MED_PT  = 12 # 15
HIG_PT  = 15 # 20
PT_CONV = 0.25

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.width = 1000

def get_pt_group(pt):
    if pt >= HIG_PT * PT_CONV:
        return "HIG"
    
    elif pt >= MED_PT * PT_CONV:
        return "MED"
    
    elif pt >= MIN_PT * PT_CONV:
        return "MIN"
    
    else:
        return "NULL"

with uproot.open(DATA_PATH + "l1Nano_WTo3Pion_PU200.root") as f:
    tree = f.get("Events")

    # FOR EACH EVENT extract the columns of the ttree and append them 
    # to an ak array. For example, the pdg_id array will be like
    # [pdg_id_event0_array, pdg_id_event1_array, ...]
    # So basically each key in branches is an array of arrays
    branches = tree.arrays()
    n_events = 50_000 # look at the keys of the correspondent hdf5 file

    print(branches["GenW_mass"][6])
    print(branches["Puppi_GenPiIdx"][6].tolist())

    # df = pd.DataFrame(columns=["ev_idx", "puppi_idx", "pt_groups"])

    # for ev_idx in tqdm(range(n_events)):
    #     nPuppi = branches["nPuppi"][ev_idx]
    #     gen_idx = branches["Puppi_GenPiIdx"][ev_idx]
    #     pts = branches["Puppi_pt"][ev_idx]

    #     zeros_vector = ak.Array([0] * nPuppi)
    #     idx0 = ak.where(gen_idx == 0)
    #     idx1 = ak.where(gen_idx == 1)
    #     idx2 = ak.where(gen_idx == 2)

    #     if (ak.count(idx0) > 0) and (ak.count(idx1) > 0)  and (ak.count(idx2) > 0) :
    #         pt_group0 = get_pt_group(pts[idx0])
    #         pt_group1 = get_pt_group(pts[idx1])
    #         pt_group2 = get_pt_group(pts[idx2])

    #         df.loc[len(df)] = [ev_idx, [idx0, idx1, idx2], [pt_group0, pt_group1, pt_group2]]
    
    # group_counts = df["pt_groups"].value_counts()
    # ax = group_counts.plot(kind="bar")

    # for idx, count in enumerate(group_counts):
    #     ax.text(idx, count + 0.1, str(count), ha="center", va="bottom")
    
    # plt.subplots_adjust(bottom=0.2)

    # plt.show()

