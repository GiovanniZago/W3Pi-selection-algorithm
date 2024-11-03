import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

with uproot.open(DATA_PATH + "l1Nano_WTo3Pion_genmatched_PU200.root") as f_genmatch:
    with uproot.open(DATA_PATH + "l1Nano_WTo3Pion_reco_PU200_v1.root") as f_reco:
        # gen matched events
        tree_genmatch = f_genmatch.get("genmatched_tree")
        branches_genmatch = tree_genmatch.arrays()
        evts_genmatch = branches_genmatch["ev_idx"]
        triplets_genmatch = branches_genmatch["part_idxs"]
        pt_groups_genmatch = branches_genmatch["pt_groups"]
        mass_genmatch = branches_genmatch["gen_mass"]

        # reco events
        tree_reco = f_reco.get("reco_tree")
        branches_reco = tree_reco.arrays()
        evts_reco = branches_reco["ev_idx"]
        triplets_reco = branches_reco["part_idxs"]
        mass_reco = branches_reco["reco_mass"]

        # stats
        n_evts_genmatch = len(evts_genmatch)
        n_evts_reco = len(evts_reco)
        reco_score = (n_evts_reco / n_evts_genmatch) * 100
        print(f"No. gentmatched events = {n_evts_genmatch}")
        print(f"No. reconstructed events = {n_evts_reco} ({reco_score:.2f} %)")
        print("\n\n")

        if n_evts_reco > n_evts_genmatch:
            raise ValueError(f"Reconstructed events ({n_evts_reco}) are more than Genmatched events ({n_evts_genmatch})")

        for i, evt_reco in enumerate(evts_reco):
            is_genmatched = np.isin(evt_reco, evts_genmatch)

            if not is_genmatched:
                triplet_reco = triplets_reco[i].to_numpy()
                
                print(f"+++++++++++++++++++++++++++++++++++++++ Event #{evt_reco} has a reconstructed triplet that is not genmatched")   
                print(f"Reconstructed triplet: {triplet_reco}")
                print(f"Reconstructed W mass: {mass_reco[i]}")
                print("\n")
                continue
                        
            triplet_reco = triplets_reco[i].to_numpy()
            i_gm = np.where(evt_reco == evts_genmatch)[0]

            if len(i_gm) > 1:
                raise ValueError(f"Event #{evt_reco} has multiple instances in evts_genmatch: {i_gm}")
            
            i_gm = i_gm[0]
            triplet_genmatch = triplets_genmatch[i_gm].to_numpy()
            are_equal = np.allclose(np.sort(triplet_genmatch), np.sort(triplet_reco))

            if are_equal:
                print(f"Event #{evt_reco} genmatched triplet is equal to reconstructed triplet")
                print(f"Genmatched triplet: {triplet_genmatch}")
                print(f"Generated W mass: {mass_genmatch[i_gm]}")
                print(f"Reconstructed triplet: {triplet_reco}")
                print(f"Reconstructed W mass: {mass_reco[i]}")
                print("\n")

            else:
                print(f"*************************************** Event #{evt_reco} genmatched triplet differs reconstructed triplet")
                print(f"Genmatched triplet: {triplet_genmatch}")
                print(f"Generated W mass: {mass_genmatch[i_gm]}")
                print(f"Genmatched triplet pt groups: {pt_groups_genmatch[i_gm].to_numpy()}")
                print(f"Reconstructed triplet: {triplet_reco}")
                print(f"Reconstructed W mass: {mass_reco[i]}")
                print("\n")
                







        