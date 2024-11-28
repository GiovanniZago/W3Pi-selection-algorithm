import h5py
import numpy as np
import sys

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"
LINE_SIZE = 8

def unpack_header(line_str):
    data = int.from_bytes(line_str, sys.byteorder)

    vld_header = (data >> 62) & 0b11                    # Bits 63-62 (2 bits)
    err_bit    = (data >> 61) & 0b1                     # Bit 61 (1 bit)
    lr_num     = (data >> 56) & 0b11111                 # Bits 60-56 (5 bits)
    orbit_cnt  = (data >> 24) & 0xFFFFFFFF              # Bits 55-24 (32 bits)
    bx_cnt     = (data >> 12) & 0xFFF                   # Bits 23-12 (12 bits)
    n_cand     = data & 0xFF                            # Bits 11-00 (12 bits) BUT ONLY 8 EFFECTIVELY USED (look at the binary mask indeed)

    return {
        "vld_header": vld_header,
        "err_bit": err_bit,
        "lr_num": lr_num,
        "orbit_cnt": orbit_cnt,
        "bx_cnt": bx_cnt,
        "n_cand": n_cand
    }

def unpack_particle(line_str):
    data = int.from_bytes(line_str, sys.byteorder)

    pdg_id      = (data >> 37) & 0b111          # Bits 39-37 (3 bits)
    phi_sign    = (data >> 36) & 0b1            # Bit 36 is the sign bit for phi
    phi_payload = (data >> 26) & 0x3FF          # Bits 35-26 (10 bits) are the non-sign bits for phi
    eta_sign    = (data >> 25) & 0b1            # Bit 25 is the sign bit for eta
    eta_payload = (data >> 14) & 0x7FF          # Bits 24-14 (11 bits) are the non-sign bits for eta
    pt          = data & 0x3FFF                 # Bits 13-00 (14 bits) 

    phi = 0
    for ii in range(phi_payload.bit_length()):
        cur_bit = (phi_payload >> ii) & 0b1

        if cur_bit:
            phi += 2 ** ii

    phi += (-1) * phi_sign * 2 ** (11 - 1)

    eta = 0
    for ii in range(eta_payload.bit_length()):
        cur_bit = (eta_payload >> ii) & 0b1

        if cur_bit:
            eta += 2 ** ii

    eta += (-1) * eta_sign * 2 ** (12 - 1)

    return {
        "pdg_id": pdg_id, 
        "phi": phi, 
        "eta": eta, 
        "pt": pt
    }

with open(DATA_PATH + "puppi_WTo3Pion_PU200.dump", "rb") as f_bin:
    with h5py.File(DATA_PATH + "l1Nano_WTo3Pion_PU200_FixedPoint.hdf5", "w") as f_hdf:
        is_header = True
        ev_idx = 0

        while True:
            header_bytes = f_bin.read(LINE_SIZE)

            if not header_bytes:
                break

            grp = f_hdf.create_group(f"{ev_idx}")

            header_data = unpack_header(header_bytes)
            n_puppi = header_data["n_cand"]
            grp.attrs["n_puppi"] = n_puppi

            pts = np.zeros(n_puppi, dtype=np.int16)
            etas = np.zeros(n_puppi, dtype=np.int16)
            phis = np.zeros(n_puppi, dtype=np.int16)
            pdg_ids = np.zeros(n_puppi, dtype=np.int16)

            for i in range(n_puppi):
                part_bytes = f_bin.read(LINE_SIZE)

                if not part_bytes:
                    raise ValueError("Expected particle not found")
                
                part_data = unpack_particle(part_bytes)
                pts[i] = part_data["pt"]
                etas[i] = part_data["eta"]
                phis[i] = part_data["phi"]
                pdg_ids[i] = part_data["pdg_id"]

            grp.create_dataset("pt", data=pts, dtype=np.int16)
            grp.create_dataset("eta", data=etas, dtype=np.int16)
            grp.create_dataset("phi", data=phis, dtype=np.int16)
            grp.create_dataset("pdg_id", data=pdg_ids, dtype=np.int16)

            ev_idx += 1


                

            

