DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

import uproot
import awkward as ak
import numpy as np
import pandas as pd

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.width = 1000

with uproot.open(DATA_PATH + "l1Nano_WTo3Pion_PU200.root") as f:
    tree = f.get("Events")

    # FOR EACH EVENT extract the columns of the ttree and append them 
    # to an ak array. For example, the pdg_id array will be like
    # [pdg_id_event0_array, pdg_id_event1_array, ...]
    # So basically each key in branches is an array of arrays
    branches = tree.arrays()
    
    # select specific event
    n_events = 50_000
    # event_idx = 6
    # event = branches[event_idx]
    
    # GenW_mass = event["GenW_mass"]
    # nPuppi = event["nPuppi"]

    # columns = ["Puppi_pdgId", "Puppi_phi", "Puppi_eta", "Puppi_pt", "Puppi_GenPiIdx"]
    # df_data = {col: event[col] for col in columns}
    # df = pd.DataFrame(df_data)
    # df["Puppi_phi_FP"] = df["Puppi_phi"].apply(lambda x: int(x * (720 / np.pi)))
    # df["Puppi_eta_FP"] = df["Puppi_eta"].apply(lambda x: int(x * (720 / np.pi)))
    # df["Puppi_pt_FP"] = df["Puppi_pt"].apply(lambda x: int(x / 0.25))
    # print(df)

    for idx in range(n_events):
        nPuppi = branches["nPuppi"][idx]
        gen_idx = branches["Puppi_GenPiIdx"][idx]
        pts = branches["Puppi_pt"][idx]

        zeros_vector = ak.Array([0] * nPuppi)


