import networkx as nx
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser(description='Visualize a graph from an adjacency list')
parser.add_argument('input_file', help='Path to the adjacency list file')
parser.add_argument('output_file', help='Path to save the output image')
parser.add_argument('--coords_file', help='Path of 2D coordinates csv')
args = parser.parse_args()

G = nx.Graph()

with open(args.input_file, 'r') as f:
    line = f.readline().strip()
    while line:
        parts = [part for part in re.split(r":|,| ", line) if part]
        v = int(parts[0])
        neighbors = list(map(int, parts[1:]))
        for neighbor in neighbors:
            G.add_edge(v, neighbor)
        line = f.readline().strip()

plt.figure(figsize=(10, 10))
if args.coords_file: # Draw the graph using the provided 2D coordinates
    coords_df = pd.read_csv(args.coords_file, header=None)
    if coords_df.shape[1] != 2:
        raise ValueError("Coordinate csv file must have exactly 2 columns for 2D plotting")
    pos = {index: (row[0], row[1]) for index, row in coords_df.iterrows()}
    nx.draw(G, pos=pos, with_labels=True)
else:
    nx.draw(G, with_labels=True)
plt.savefig(args.output_file)
