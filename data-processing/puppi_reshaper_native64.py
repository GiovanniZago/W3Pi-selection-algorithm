from puppi_unpacker_native64 import unpack_header 
import struct

ADJ_TO = 4
DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

def puppi_reshaper_native64(puppi_old, puppi_new, adj_to=ADJ_TO):
    ii          = 0
    is_header   = True
    p_count     = 0
    p_counter   = 0
    p_to_append = 0

    with open(puppi_old, "rb") as puppi_dump_old:
        with open(puppi_new, "wb") as puppi_dump_new:
            while True:
                row_bytes = puppi_dump_old.read(8)
                ii += 1

                if not row_bytes: 
                    break

                if is_header:
                    row_data = unpack_header(row_bytes)
                    p_count = row_data["n_cand"]

                    rem = p_count % adj_to
                    if rem:
                        ev_size = (int(p_count / adj_to) + 1) * adj_to
                    else:
                        ev_size = p_count

                    word = struct.pack("Q", (ev_size << 56) | (row_data["bx_cnt"] << 12) | p_count)
                    puppi_dump_new.write(word)
                    p_to_append = ev_size - p_count

                    is_header = False

                else:
                    puppi_dump_new.write(row_bytes)
                    p_counter += 1

                    if p_counter == p_count:
                        for _ in range(p_to_append):
                            puppi_dump_new.write(b"\x00" * 8)
                            ii += 1
                        is_header = True
                        p_counter = 0

def main():
    puppi_reshaper_native64(DATA_PATH + "Puppi.dump", DATA_PATH + f"Puppi_adj{ADJ_TO}.dump")

if __name__ == "__main__":
    main()