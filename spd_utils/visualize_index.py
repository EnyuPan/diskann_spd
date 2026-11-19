import networkx as nx
import matplotlib.pyplot as plt
import argparse
import re

parser = argparse.ArgumentParser(description='Visualize a graph from an adjacency list')
parser.add_argument('input_file', help='Path to the adjacency list file')
parser.add_argument('output_file', help='Path to save the output image')
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
nx.draw(G, with_labels=True)
plt.savefig(args.output_file)
