# Adapted from: https://gist.github.com/danoneata/49a807f47656fedbb389
# Usage: python fvecs_to_csv.py [input.fvecs] [output.csv]

import argparse
import numpy as np

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


parser = argparse.ArgumentParser(description='Convert fvecs to csv')
parser.add_argument('input_file', type=str, help='Input fvecs file')
parser.add_argument('output_file', type=str, help='Output csv file')
args = parser.parse_args()

fvecs = fvecs_read(args.input_file)
np.savetxt(args.output_file, fvecs, delimiter=',', fmt='%.2f')
