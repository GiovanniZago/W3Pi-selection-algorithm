import sys

def unpack_header(row_bytes):
    data = int.from_bytes(row_bytes, sys.byteorder)

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

def unpack_particle(row_bytes):
    data = int.from_bytes(row_bytes, sys.byteorder)

    pid         = (data >> 37) & 0b111          # Bits 39-37 (3 bits)
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

    match pid:
        case 0: # 000
            pdg_id = 130

        case 1: # 001
            pdg_id = 22

        case 2: # 010
            pdg_id = -211

        case 3: # 011
            pdg_id = 211

        case 4: # 100
            pdg_id = 11

        case 5: # 101
            pdg_id = -11

        case 6: # 110
            pdg_id = 13

        case 7: # 111
            pdg_id = -13

        case _:
            pdg_id = "ERR"

    return {
        "pdg_id": pdg_id, 
        "phi": phi, 
        "eta": eta, 
        "pt": pt
    }

def puppi_unpacker_native64(file_name):
    ii        = 0
    start_idx = 0
    is_header = True
    p_count   = 0
    p_counter = 0
    
    header_data = []
    part_data   = []

    with open(file_name, "rb") as puppi_dump:
        while True:
            row_bytes = puppi_dump.read(8) # read 8 bytes = 64 bits
            ii += 1

            if not row_bytes: 
                break

            if is_header:
                row_data = unpack_header(row_bytes)
                start_idx = ii
                header_data.append((start_idx, row_data["vld_header"], row_data["err_bit"], row_data["lr_num"], row_data["orbit_cnt"], row_data["bx_cnt"], row_data["n_cand"]))
                ev_size = (row_data["vld_header"] << 6) |  (row_data["err_bit"] << 5) | row_data["lr_num"]

                if row_data["n_cand"] > 0: 
                    is_header = False

                    if ev_size:
                        p_count = ev_size
                    else:
                        p_count = row_data["n_cand"]
                    

            else:
                row_data = unpack_particle(row_bytes)
                part_data.append((start_idx, row_data["pdg_id"], row_data["phi"], row_data["eta"], row_data["pt"]))
                p_counter += 1
                if p_counter == p_count:
                    is_header = True
                    p_counter = 0

    return (header_data, part_data)

def main():
    pass

if __name__ == "__main__":
    main()