DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

import h5py
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm

MIN_PT  = 28 # 9
MED_PT  = 48 # 15
HIG_PT  = 60 # 20
PT_CONV = 0.25

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
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200.hdf5", "w") as f_out:
        tree = f_in.get("Events")

        # FOR EACH EVENT extract the columns of the ttree and append them 
        # to an ak array. For example, the pdg_id array will be like
        # [pdg_id_event0_array, pdg_id_event1_array, ...]
        # So basically each key in branches is an array of arrays
        branches = tree.arrays()
        
        # metadata variables
        n_events = tree.num_entries # look at the keys of the correspondent hdf5 file
        n_gen_acceptance = 0
        n_gen_match = 0
        
        for ev_idx in tqdm(range(n_events)):
            grp = f_out.create_group(f"{ev_idx}")
            grp.create_dataset("pt", data=branches["Puppi_pt"][ev_idx].to_numpy(), dtype=np.float32)
            grp.create_dataset("eta", data=branches["Puppi_eta"][ev_idx].to_numpy(), dtype=np.float32)
            grp.create_dataset("phi", data=branches["Puppi_phi"][ev_idx].to_numpy(), dtype=np.float32)
            grp.create_dataset("pdg_id", data=branches["Puppi_pdgId"][ev_idx].to_numpy(), dtype=np.int16)
            grp.create_dataset("gen_pi_idx", data=branches["Puppi_GenPiIdx"][ev_idx].to_numpy(), dtype=np.int16)
            grp.create_dataset("gen_pi_pt", data=branches["GenPi_pt"][ev_idx].to_numpy(), dtype=np.float32)
            grp.create_dataset("gen_pi_eta", data=branches["GenPi_eta"][ev_idx].to_numpy(), dtype=np.float32)
            grp.create_dataset("gen_pi_phi", data=branches["GenPi_phi"][ev_idx].to_numpy(), dtype=np.float32)
            grp.create_dataset("gen_pi_mass", data=branches["GenPi_mass"][ev_idx].to_numpy(), dtype=np.float32)

            gen_pi_pt = grp["gen_pi_pt"][...]
            gen_pi_eta = grp["gen_pi_eta"][...]
            gen_pi_idx = grp["gen_pi_idx"][...]

            grp.attrs["n_puppi"] = branches["nPuppi"][ev_idx]

            if (np.all(gen_pi_pt) > 2) and (np.all(gen_pi_eta) < 2.4):
                grp.attrs["is_acceptance"] = True
            
            else:
                grp.attrs["is_acceptance"] = False

            idx0 = np.where(gen_pi_idx == 0)[0]
            idx1 = np.where(gen_pi_idx == 1)[0]
            idx2 = np.where(gen_pi_idx == 2)[0]

            pt = grp["pt"][...]

            if idx0.size or idx1.size or idx2.size:
                if idx0.size and idx1.size and idx2.size:
                    grp.attrs["is_genmatch"] = 1
                    grp.attrs["genmatch_pi_triplet"] = np.array([idx0, idx1, idx2]).flatten()
                    grp.attrs["genmatch_pi_group"] = np.array([get_pt_group(pt[idx0]), 
                                                               get_pt_group(pt[idx1]), 
                                                               get_pt_group(pt[idx2])])

                
                else:
                    idx0 = idx0 if idx0.size else np.array([-1])
                    idx1 = idx1 if idx1.size else np.array([-1])
                    idx2 = idx2 if idx2.size else np.array([-1])
                    grp.attrs["is_genmatch"] = 0
                    grp.attrs["genmatch_pi_triplet"] = np.array([idx0, idx1, idx2]).flatten()
                    grp.attrs["genmatch_pi_group"] = np.array([get_pt_group(pt[idx0]), 
                                                               get_pt_group(pt[idx1]), 
                                                               get_pt_group(pt[idx2])])
            
            else:
                grp.attrs["is_genmatch"] = -1
                grp.attrs["genmatch_pi_triplet"] = np.array([-1, -1, -1])
                grp.attrs["genmatch_pi_group"] = np.array([-1, -1, -1])

            


       

                


