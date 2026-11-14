# Fast precomputed DBSCAN

## sklearn 1.8.0 semantics audit

The implementation was checked against the installed scikit-learn 1.8.0
sources:

- `sklearn/cluster/_dbscan.py`: `DBSCAN.fit` builds all radius neighborhoods,
  counts each neighborhood, marks samples with at least `min_samples` members
  as core, initializes every label to `-1`, and calls `dbscan_inner`.
- `sklearn/neighbors/_base.py`: the generic radius reducer uses
  `np.where(d <= radius)`, so the radius comparison is inclusive.
- `sklearn/cluster/_dbscan_inner.pyx`: the outer loop visits sample indices in
  increasing order. Every still-unlabeled core starts the next cluster. Its
  depth-first search labels all reached samples but expands neighborhoods only
  for core samples.

Therefore, on the symmetric precomputed matrices used by this project:

1. Core clusters are connected components of the core-only radius graph.
2. Cluster numbers follow each component's smallest core-sample index.
3. A non-core border sample reachable from several components is retained by
   the first expanded component, which is the smallest cluster number.

One dtype detail is also reproduced. In sklearn 1.8.0, float32 precomputed
matrices use the optimized native-dtype radius reducer. Float16 matrices use
the generic pairwise path and are promoted to float64 before comparison. This
matters for a stored `float16(0.6)`, whose exact value is slightly above 0.6.

## Implementation and memory bound

`fast_dbscan_precomputed` performs three passes without changing or converting
the complete caller-owned matrix:

1. Count inclusive radius neighbors in row blocks and find core samples.
2. Build an upper-triangular core graph one row block at a time. Each temporary
   int32/bool CSR is reduced with `scipy.sparse.csgraph.connected_components`,
   then merged into a global union-find. Total graph edges are never resident
   simultaneously.
3. Assign every non-core sample the minimum component number among its core
   neighbors; samples without a core neighbor remain `-1`.

The default block size is 1,024 rows. It can be lowered with
`PCLHD_FAST_DBSCAN_BLOCK_ROWS` if a single unusually dense block approaches
the 40 GiB RSS guard. The implementation raises `MemoryError` with an explicit
external-memory/GPU connected-components recommendation instead of switching
backends silently.

Each pass prints elapsed time, upper core-edge count where applicable, and the
largest sampled Linux `/proc/self/status` `VmRSS`.
