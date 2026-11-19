import numpy as np
import struct
import argparse
import csv

def csv_to_fvecs(csv_path, fvecs_path):
    with open(csv_path, 'r', newline='') as f_in, open(fvecs_path, 'wb') as f_out:
        reader = csv.reader(f_in)
        dim = None
        for row in reader:
            if not row:
                continue
            vector = np.asarray(row, dtype=np.float32)
            if dim is None:
                dim = len(vector)
            elif len(vector) != dim:
                raise ValueError(f"Inconsistent vector dimension: expected {dim}, got {len(vector)}")
            # Write vector dimension (int32)
            f_out.write(struct.pack('<i', dim))
            # Write vector values (float32)
            f_out.write(vector.astype('<f4').tobytes())

parser = argparse.ArgumentParser(description='Convert fvecs to csv')
parser.add_argument('input_file', type=str, help='Input fvecs file')
parser.add_argument('output_file', type=str, help='Output csv file')
args = parser.parse_args()

csv_to_fvecs(args.input_file, args.output_file)
