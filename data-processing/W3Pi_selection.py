import numpy as np

MIN_PT1     = 7 # 9
MIN_PT2     = 12 # 15
MIN_PT3     = 15 # 20
MIN_DELTAR2 = 0.5 * 0.5
MIN_MASS    = 40 # 60
MIN_DR2     = 0.01 * 0.01

MAX_MASS    = 150 # 100
MAX_DR2     = 0.25 * 0.25
MAX_ISO     = 2.0 # 0.4

def zeroTwoPiAngle(x, y):
    if (x > 2 * np.pi) or (x < 0) or (y > 2 * np.pi) or (y < 0):
        raise ValueError("The two inputs must be inside the interval [0, 2pi]")

    diff = x - y + np.pi
    if diff > 3 * np.pi:
        diff = diff - np.pi

    return diff 

def isolation(p_index: int, n_particles: int, etas: np.ndarray, phis: np.ndarray, pts: np.ndarray):
    is_isolated = False
    pt_sum = 0

    for ii in range(n_particles):
        if (p_index == ii):
            continue
        
        d_eta = etas[p_index] - etas[ii]
        d_phi = zeroTwoPiAngle(phis[p_index], phis[ii])
        d_r2 = d_eta * d_eta + d_phi * d_phi

        if (d_r2 >= MIN_DR2) and (d_r2 <= MAX_DR2):
            pt_sum = pt_sum + pts[ii]
    
    if pt_sum <= MAX_ISO * pts[p_index]:
        is_isolated = True

    return is_isolated

def ang_sep(eta1, eta2, phi1, phi2):
    is_ang_sep = True

    d_eta = eta1 - eta2
    d_phi = zeroTwoPiAngle(phi1, phi2)
    d_r2 = d_eta * d_eta + d_phi * d_phi

    if d_r2 < MIN_DELTAR2:
        is_ang_sep = False
    
    return is_ang_sep

def W3PiSelection(cands: list):
    pass

def main():
    pass

if __name__ == "__main__":
    main()