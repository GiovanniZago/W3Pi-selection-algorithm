import struct

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

def puppi_to_aiecsv(puppi_file, csv_file_H, csv_file_L):
    with open(puppi_file, "rb") as puppi_dump:
        with open(csv_file_H, "w") as puppi_csv_H:
            with open(csv_file_L, "w") as puppi_csv_L:
                puppi_csv_H.write("CMD,D,TLAST,TKEEP\n")
                puppi_csv_L.write("CMD,D,TLAST,TKEEP\n")

                while True:
                    row_bytes = puppi_dump.read(8)

                    if not row_bytes:
                        break

                    row_data = struct.unpack("ii", row_bytes)
                    puppi_csv_H.write("DATA," + f"{str(row_data[1])}," + "0," + "-1\n")
                    puppi_csv_L.write("DATA," + f"{str(row_data[0])}," + "0," + "-1\n")


            
def main():
    puppi_to_aiecsv(DATA_PATH + "Puppi_adj4.dump", DATA_PATH + "aie_data/Puppi_adj4_H.csv", DATA_PATH + "aie_data/Pupppi_adj4_L.csv")

if __name__ == "__main__":
    main()