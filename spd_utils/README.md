## Export DiskANN graph as adjacency list

### Example of exporting a graph index for the Sift-small dataset as an adjacency list

Build the project (note the `DPORTABLE` flag is necessary on the infernal cluster):
```
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON .. && make -j 
```
Download the 10,000-vector Sift-small dataset:
```
mkdir build/data && cd build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xf siftsmall.tar.gz
cd ..
./apps/utils/fvecs_to_bin float data/siftsmall/siftsmall_learn.fvecs data/siftsmall/siftsmall_learn.fbin
./apps/utils/fvecs_to_bin float data/siftsmall/siftsmall_query.fvecs data/siftsmall/siftsmall_query.fbin
```

Compute ground truth (should be optional):
(from the `build` directory)
```
./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/siftsmall/siftsmall_learn.fbin --query_file  data/siftsmall/siftsmall_query.fbin --gt_file data/siftsmall/siftsmall_query_learn_gt100 --K 100
```

Build the graph index and export the adjacency list to `graph.txt`:
```
./apps/build_memory_index  --data_type float --dist_fn l2 --data_path data/siftsmall/siftsmall_learn.fbin --index_path_prefix data/siftsmall/index_siftsmall_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2 --graph_output data/siftsmall/graph.txt
```
The output file `graph.txt` lists the neighbors of each node after it. For example, the first two rows for the SIFT-small index are:
```
0: 11970 16061 23575 13345 16410 2735 15916 5203 1541 22837 17048 15762 16346 231 13362 23257 5128 7034 20282 15706 19398 23900 240 14239 6802 17477 2780 7208 23699 5587 8317 8019
1: 9765 15271 2 10725 17904 19636 15797 13287 1747 7954 465 15640 2009 2060 1136 3435 10982 18082 8543 359 13737 2118 15187 23068 20264 2239 4943 225 15301 4282 12430 20617
```
The node ids match the order in which they are read from the dataset.

Search the index using the query set (optional):
```
 ./apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/siftsmall/index_siftsmall_learn_R32_L50_A1.2 --query_file data/siftsmall/siftsmall_query.fbin  --gt_file data/siftsmall/siftsmall_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path data/siftsmall/res
```
## Work with example graphs
Each directory under `example_graphs` contains:
- A `graph.txt` adjacency list
- A `coordinates.csv` file containing vector coordinates


Visualize the adjacency structure of a graph (note this does not reflect the coordinates of the points):
```
python3 visualize_index.py [adjacency list file] [path for output png]
```

Convert a fvecs dataset to csv: (SIFT-small example)
```
python fvecs_to_csv.py [fvecs file] [path for output csv file]
```
The resulting csv contains the coordinates of the data vectors. For example, the first two rows of the SIFT-small dataset:
```
97.00,18.00,9.00,9.00,0.00,0.00,2.00,36.00,21.00,16.00,59.00,106.00,64.00,0.00,5.00,27.00,5.00,38.00,126.00,68.00,19.00,0.00,0.00,2.00,6.00,38.00,26.00,15.00,4.00,0.00,0.00,2.00,61.00,16.00,8.00,16.00,28.00,19.00,15.00,50.00,126.00,95.00,44.00,6.00,9.00,12.00,7.00,59.00,5.00,33.00,88.00,60.00,101.00,95.00,5.00,1.00,69.00,49.00,11.00,20.00,61.00,25.00,0.00,2.00,46.00,18.00,0.00,0.00,24.00,38.00,15.00,68.00,116.00,18.00,0.00,0.00,8.00,63.00,57.00,126.00,7.00,1.00,0.00,0.00,34.00,126.00,90.00,26.00,109.00,8.00,0.00,3.00,15.00,39.00,20.00,72.00,4.00,38.00,12.00,3.00,7.00,9.00,1.00,14.00,2.00,4.00,4.00,11.00,14.00,26.00,57.00,50.00,11.00,8.00,9.00,3.00,1.00,60.00,56.00,6.00,111.00,4.00,1.00,1.00,0.00,2.00,4.00,26.00
67.00,22.00,31.00,15.00,0.00,0.00,0.00,8.00,49.00,40.00,59.00,46.00,25.00,2.00,2.00,13.00,20.00,72.00,84.00,58.00,15.00,0.00,1.00,3.00,19.00,47.00,18.00,34.00,36.00,16.00,13.00,2.00,62.00,99.00,93.00,20.00,4.00,8.00,2.00,21.00,65.00,118.00,118.00,46.00,4.00,11.00,5.00,11.00,17.00,15.00,118.00,118.00,118.00,37.00,6.00,23.00,17.00,10.00,9.00,55.00,57.00,28.00,10.00,41.00,118.00,10.00,0.00,3.00,10.00,28.00,8.00,93.00,94.00,13.00,1.00,0.00,7.00,74.00,90.00,87.00,0.00,0.00,1.00,26.00,118.00,118.00,28.00,2.00,2.00,4.00,7.00,40.00,39.00,25.00,9.00,6.00,58.00,2.00,2.00,13.00,10.00,2.00,1.00,51.00,44.00,1.00,0.00,0.00,4.00,17.00,58.00,65.00,6.00,1.00,1.00,8.00,17.00,48.00,23.00,9.00,8.00,4.00,3.00,28.00,27.00,7.00,3.00,5.00
```
Convert csv to fvecs:
```
python csv_to_fvecs.py [csv file] [path for output fvecs file]
```
