import numpy as np
import puppi
import pandas as pd

pd.options.display.max_rows = 200

MIN_PT1     = 7 # 9
MIN_PT2     = 12 # 15
MIN_PT3     = 15 # 20
MIN_DELTAR2 = 0.5 * 0.5
MIN_MASS    = 40 # 60
MIN_DR2     = 0.01 * 0.01
PI          = 720
MAX_MASS    = 150 # 100
MAX_DR2     = 0.25 * 0.25
MAX_ISO     = 2.0 # 0.4

def ang_diff(x, y):
    if (np.abs(x) > PI):
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
        d_r2 *= (np.pi / PI) ** 2 # conversion to floating point

        if (d_r2 >= MIN_DR2) and (d_r2 <= MAX_DR2):
            pt_sum = pt_sum + pts[ii]
    
    pt_sum *= 0.25
    pt_cur *= 0.25

    if pt_sum <= (MAX_ISO * pt_cur):
        is_isolated = 1

    return is_isolated

def ang_sep(eta1, eta2, phi1, phi2):
    is_ang_sep = True

    d_eta = eta1 - eta2
    d_phi = ang_diff(phi1, phi2)
    d_r2 = d_eta * d_eta + d_phi * d_phi

    if d_r2 < MIN_DELTAR2:
        is_ang_sep = False
    
    return is_ang_sep


# DATAFRAME FUNCTIONS
def filter_pt_pdgid(row: pd.Series):
    if ((abs(row["pdg_id"]) != 211) and (abs(row["pdg_id"]) != 11)) or (row["pt"] < MIN_PT1):
        row["pdg_id"] = 0

    return row

def is_iso(row, df):
    p_index = row.name
    n_particles = len(df)
    etas = df["eta"].to_numpy()
    phi = df["phi"].to_numpy()
    pts = df["pt"].to_numpy()

    is_iso = isolation(p_index, n_particles, etas, phi, pts)
    return is_iso

def W3PiSelection(cands: list):
    pass

def main():
    file = "Puppi_fix104mod.dump"

    with puppi.PuppiData(file) as myPuppi:
        headers_data, parts_data = myPuppi.get_lines_data(104, 207) # select one event

        # set to zero the headers by keeping only the index and four zeros
        headers_data = [(header[0], 0, 0, 0, 0) for header in headers_data]

        # merge headers and particles and sort by index
        data = headers_data + parts_data
        data.sort(key=lambda x: x[0])

        # use a pandas dataframe
        columns = ["idx", "pdg_id", "phi", "eta", "pt"]
        df = pd.DataFrame(data, columns=columns)

        # filter particles with correct pdg_id and min pt
        df = df.apply(lambda x: filter_pt_pdgid(x), axis=1)

        # verify isolation of each particle
        df["is_iso"] = df.apply(lambda x: is_iso(x, df), axis=1)

        # keep only the rows that satisfy minpt, pdg_id and isolation
        df = df[(df["is_iso"] == 1) & (df["pdg_id"] != 0)]
        # df = df[(df["pdg_id"] != 0)]
        print(df.head(100).to_string(index=False))








if __name__ == "__main__":
    main()