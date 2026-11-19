# Example graph fixtures

Each subdirectory contains a `graph.txt` adjacency list and a
`coordinates.csv` file with one point per node. Node IDs are zero-based and
match coordinate row numbers; dimensions vary by fixture. Adjacency is written
as `node_id: neighbor_ids`; these examples use symmetric edges.

| Fixture | Structure and use |
| --- | --- |
| `path` | Six-node chain; tests long, sequential traversal. |
| `complete` | Five-node complete graph; tests highly connected neighborhoods. |
| `balanced_tree` | Five-level, 31-node binary tree with 5D coordinates; tests branching and hierarchical routing. |
| `two_branch_tree` | Two extended branches from a common root; tests queries on either side of a fork. |
| `two_clusters_bridge` | Two compact groups joined by one edge; tests crossing a narrow connection. |
| `ring_with_shortcuts` | Cycle with long-range links; tests local loops and shortcut routing. |

These are small topology/visualization fixtures, not prebuilt DiskANN binary
indices. The coordinate files contain no header, like `sample_graph.csv`.
