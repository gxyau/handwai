#!/usr/bin/env python

in_f  = input("Please insert TSV file: ").strip()
out_f = input("Please insert output CoNLL file: ").strip()

def tsv_to_conll(input_file, output_file):
    with open(in_f, 'r') as f:
        lines = f.readlines()
    
    with open(out_f, 'w') as f:
        for line in lines:
            try:
                items = line.strip().split()
                f.write(f"{items[0]}\t{items[1]}\t{items[2]}\n")
            except IndexError:
                f.write("\n")

if __name__ == "__main__":
    tsv_to_conll(in_f, out_f)

