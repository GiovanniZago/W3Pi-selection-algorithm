from puppi_unpacker_native64 import puppi_unpacker_native64 as unpacker
from puppi_unpacker_native64 import puppi_unpacker_mod_native64 as unpacker_mod

DATA_PATH = "/home/giovanni/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/"

def print_table(data, column_names, start=0, end=10):
    columns = len(column_names)
    col_widths = [max(len(str(row[i])) for row in data) for i in range(columns)]
    col_widths = [max(len(column_names[i]), col_widths[i]) for i in range(columns)]
    format_str = ' | '.join([f'{{:<{width}}}' for width in col_widths])

    print(format_str.format(*column_names))
    print('-' * (sum(col_widths) + 3 * (columns - 1)))
    for row in data[start:end+1]:
        print(format_str.format(*row))

def main():
    header_data, part_data = unpacker(DATA_PATH + "Puppi.dump")
    columns_head = ["start_idx", "vld_header", "err_bit", "lr_number", "orbit_cnt", "bx_cnt", "n_cand"]
    columns_part = ["start_idx", "pdg_id", "phi", "eta", "pt"]

    print_table(header_data, columns_head, start=3500, end=3600)
    print("\n\n\n")
    print_table(part_data, columns_part, start=0, end=10)

if __name__ == "__main__":
    main()