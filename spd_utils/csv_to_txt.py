# Converts coordinates.csv to a format where:
# - The first line contains n (the number of vectors) and d (the dimension of each vector),
#   separated by a space
# - Each subsequent line contains the d coordinates of a vector, separated by spaces
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser(description='Convert csv to txt')
parser.add_argument('input_file', type=str, help='Input csv file')
parser.add_argument('output_file', type=str, help='Output txt file')
args = parser.parse_args()

def csv_to_txt(csv_path, txt_path):
    with open(csv_path, 'r', newline='') as f_in, open(txt_path, 'w') as f_out:
        reader = csv.reader(f_in)
        vectors = []
        for row in reader:
            if not row:
                continue
            vector = np.asarray(row, dtype=np.float32)
            vectors.append(vector)
        n = len(vectors)
        d = len(vectors[0]) if n > 0 else 0
        f_out.write(f"{n} {d}\n")
        for vector in vectors:
            f_out.write(' '.join(map(str, vector)) + '\n')

csv_to_txt(args.input_file, args.output_file)
