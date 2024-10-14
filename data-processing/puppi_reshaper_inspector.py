from puppi_unpacker_native64 import puppi_unpacker_native64 as unpacker

def print_table(data, headers, start=0, end=10):
    columns = len(headers)
    col_widths = [max(len(str(row[i])) for row in data) for i in range(columns)]
    col_widths = [max(len(headers[i]), col_widths[i]) for i in range(columns)]
    format_str = ' | '.join([f'{{:<{width}}}' for width in col_widths])

    print(format_str.format(*headers))
    print('-' * (sum(col_widths) + 3 * (columns - 1)))
    for row in data[start:end]:
        print(format_str.format(*row))

def main():
    header_data, part_data = unpacker("./data/Puppi.dump")
    columns_head = ["start_idx", "vld_header", "err_bit", "lr_number", "orbit_cnt", "bx_cnt", "n_cand"]
    columns_part = ["start_idx", "pdg_id", "phi", "eta", "pt"]

    print_table(header_data, columns_head)
    print("\n\n\n")
    print_table(part_data, columns_part, start=44, end=50)

if __name__ == "__main__":
    main()