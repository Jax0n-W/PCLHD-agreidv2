"""Memory-bounded DBSCAN for dense, symmetric precomputed distances.

The implementation reproduces scikit-learn's unweighted DBSCAN labeling
order while avoiding its object array containing every sample's neighbors.
It scans the caller-owned distance array in row blocks and never casts or
copies the complete matrix.
"""
from __future__ import absolute_import, print_function

import os
import threading
import time
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


DEFAULT_BLOCK_ROWS = 1024
MAX_RSS_BYTES = 40 * 1024 ** 3


def read_vmrss_bytes(pid=None):
    """Read current resident memory from Linux procfs, if available."""
    target = "self" if pid is None else str(int(pid))
    path = "/proc/{}/status".format(target)
    try:
        with open(path, "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _format_bytes(value):
    if value is None:
        return "unavailable"
    return "{:.3f} GiB".format(float(value) / (1024 ** 3))


class _MemoryTracker(object):
    def __init__(self):
        self.peak = read_vmrss_bytes()

    def sample(self):
        current = read_vmrss_bytes()
        if current is not None:
            if self.peak is None or current > self.peak:
                self.peak = current
        return current

    @contextmanager
    def monitor(self, interval=0.02):
        stop = threading.Event()

        def poll():
            while not stop.wait(interval):
                self.sample()

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self.sample()
            stop.set()
            thread.join()


def _block_rows():
    raw_value = os.environ.get(
        "PCLHD_FAST_DBSCAN_BLOCK_ROWS", str(DEFAULT_BLOCK_ROWS))
    try:
        value = int(raw_value)
    except ValueError:
        raise ValueError(
            "PCLHD_FAST_DBSCAN_BLOCK_ROWS must be an integer, got {!r}"
            .format(raw_value))
    if value < 1:
        raise ValueError("PCLHD_FAST_DBSCAN_BLOCK_ROWS must be positive")
    return value


def _epsilon_neighbors(values, eps):
    # sklearn 1.8's optimized precomputed reducer handles float32 in its
    # native dtype, while float16 takes the generic pairwise path and is
    # promoted to float64 before the inclusive radius comparison.
    if values.dtype == np.float16:
        values = values.astype(np.float64)
    return np.less_equal(values, eps)


def _check_input(dist, eps, min_samples):
    if not isinstance(dist, np.ndarray):
        raise TypeError("dist must be a dense numpy array or numpy memmap")
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square precomputed distance matrix")
    if dist.shape[0] == 0:
        raise ValueError("dist must contain at least one sample")
    if not np.issubdtype(dist.dtype, np.floating):
        raise TypeError("dist must have a floating dtype, got {}".format(
            dist.dtype))
    if not np.isfinite(eps) or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive number")
    if isinstance(min_samples, (bool, np.bool_)):
        raise TypeError("min_samples must be a positive integer")
    try:
        min_samples_value = int(min_samples)
    except (TypeError, ValueError, OverflowError):
        raise TypeError("min_samples must be a positive integer")
    if min_samples_value != min_samples or min_samples_value < 1:
        raise ValueError("min_samples must be a positive integer")
    return min_samples_value


def _raise_if_chunk_exceeds_limit(tracker, edge_count, node_count):
    current = tracker.sample()
    if current is None:
        return
    # scipy may materialize a transpose plus work arrays for weak components.
    # The multiplier deliberately overestimates a bool/int32 CSR edge.
    estimated_extra = int(edge_count) * 16 + int(node_count) * 24
    if current + estimated_extra <= MAX_RSS_BYTES:
        return
    raise MemoryError(
        "fast DBSCAN refused a core-edge chunk with {:,} edges: current "
        "VmRSS {} plus estimated scipy workspace {} exceeds the 40 GiB "
        "limit. Retry with a smaller PCLHD_FAST_DBSCAN_BLOCK_ROWS value. "
        "For graphs that remain too dense, use an external-memory or GPU "
        "connected-components implementation; no backend fallback was made."
        .format(edge_count, _format_bytes(current),
                _format_bytes(estimated_extra)))


def _find(parent, node):
    node = int(node)
    root = node
    while int(parent[root]) != root:
        root = int(parent[root])
    while node != root:
        next_node = int(parent[node])
        parent[node] = root
        node = next_node
    return root


def _union(parent, rank, left, right):
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if rank[left_root] < rank[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    if rank[left_root] == rank[right_root]:
        rank[left_root] += 1


def _merge_chunk_components(parent, rank, component_labels,
                            row_edge_counts, column_indices, row_offset):
    active_rows = np.flatnonzero(row_edge_counts).astype(
        np.int32, copy=False)
    active_rows += int(row_offset)
    active_columns = np.unique(column_indices)
    active = np.union1d(active_rows, active_columns)
    if active.size < 2:
        return

    local_labels = component_labels[active]
    order = np.argsort(local_labels, kind="stable")
    active = active[order]
    local_labels = local_labels[order]
    boundaries = np.flatnonzero(local_labels[1:] != local_labels[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.intp), boundaries))
    stops = np.concatenate((boundaries, np.array(
        [active.size], dtype=np.intp)))
    for start, stop in zip(starts, stops):
        if stop - start < 2:
            continue
        leader = int(active[start])
        for position in range(int(start) + 1, int(stop)):
            _union(parent, rank, leader, int(active[position]))


def _chunk_core_graph(neighbor_mask, first_row, core_count, tracker):
    row_counts = np.count_nonzero(neighbor_mask, axis=1).astype(
        np.int64, copy=False)
    edge_count = int(row_counts.sum(dtype=np.int64))
    if edge_count == 0:
        return row_counts, None, None, 0

    _raise_if_chunk_exceeds_limit(tracker, edge_count, core_count)
    column_indices = np.empty(edge_count, dtype=np.int32)
    cursor = 0
    for local_row in range(neighbor_mask.shape[0]):
        columns = np.flatnonzero(neighbor_mask[local_row])
        next_cursor = cursor + columns.size
        column_indices[cursor:next_cursor] = columns
        cursor = next_cursor

    indptr = np.empty(core_count + 1, dtype=np.int64)
    indptr[:first_row + 1] = 0
    np.cumsum(row_counts, dtype=np.int64,
              out=indptr[first_row + 1:
                         first_row + neighbor_mask.shape[0] + 1])
    last_row = first_row + neighbor_mask.shape[0]
    if last_row < core_count:
        indptr[last_row + 1:] = edge_count
    data = np.ones(edge_count, dtype=np.bool_)
    graph = csr_matrix(
        (data, column_indices, indptr),
        shape=(core_count, core_count), copy=False)
    with tracker.monitor():
        _, component_labels = connected_components(
            graph, directed=False, return_labels=True)
    tracker.sample()
    return row_counts, column_indices, component_labels, edge_count


def fast_dbscan_precomputed(dist, eps, min_samples):
    """Cluster a dense symmetric precomputed distance matrix.

    The returned ``np.intp`` labels match sklearn DBSCAN sample-for-sample,
    including cluster numbers, for symmetric precomputed distances. The input
    array is neither mutated nor converted to another dtype.
    """
    min_samples = _check_input(dist, eps, min_samples)
    sample_count = int(dist.shape[0])
    block_rows = _block_rows()
    tracker = _MemoryTracker()
    print(
        "[fast-dbscan] samples={:,}, dtype={}, eps={:g}, min_samples={}, "
        "block_rows={}".format(
            sample_count, dist.dtype, float(eps), min_samples, block_rows))

    # Pass 1: count eps-neighbors, including the diagonal when it is <= eps.
    pass_start = time.monotonic()
    neighbor_counts = np.empty(sample_count, dtype=np.int64)
    for start in range(0, sample_count, block_rows):
        stop = min(start + block_rows, sample_count)
        neighbor_mask = _epsilon_neighbors(dist[start:stop], eps)
        neighbor_counts[start:stop] = np.count_nonzero(
            neighbor_mask, axis=1)
        tracker.sample()
        del neighbor_mask
    core_mask = neighbor_counts >= min_samples
    core_indices = np.flatnonzero(core_mask).astype(np.intp, copy=False)
    core_count = int(core_indices.size)
    print(
        "[fast-dbscan] pass 1/3 neighbor counts: {:.3f}s, cores={:,}, "
        "peak VmRSS={}".format(
            time.monotonic() - pass_start, core_count,
            _format_bytes(tracker.peak)))

    # Pass 2: build only one block of the upper core graph at a time. Local
    # scipy components are merged into a tiny global union-find structure.
    pass_start = time.monotonic()
    parent = np.arange(core_count, dtype=np.int32)
    rank = np.zeros(core_count, dtype=np.uint8)
    total_edges = 0
    for first_core in range(0, core_count, block_rows):
        last_core = min(first_core + block_rows, core_count)
        source_indices = core_indices[first_core:last_core]
        distance_block = dist[np.ix_(source_indices, core_indices)]
        neighbor_mask = _epsilon_neighbors(distance_block, eps)
        for local_row, core_position in enumerate(
                range(first_core, last_core)):
            neighbor_mask[local_row, :core_position + 1] = False
        tracker.sample()
        del distance_block

        row_counts, columns, local_components, edge_count = \
            _chunk_core_graph(
                neighbor_mask, first_core, core_count, tracker)
        total_edges += edge_count
        if edge_count:
            _merge_chunk_components(
                parent, rank, local_components, row_counts, columns,
                first_core)
        del neighbor_mask, row_counts, columns, local_components

    core_cluster_labels = np.empty(core_count, dtype=np.intp)
    root_to_label = {}
    for core_position in range(core_count):
        root = _find(parent, core_position)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        core_cluster_labels[core_position] = root_to_label[root]
    component_count = len(root_to_label)
    print(
        "[fast-dbscan] pass 2/3 core components: {:.3f}s, upper edges={:,}, "
        "components={:,}, peak VmRSS={}".format(
            time.monotonic() - pass_start, total_edges, component_count,
            _format_bytes(tracker.peak)))

    # Pass 3: a border sample belongs to the earliest sklearn component that
    # reaches it, which is the minimum component number among core neighbors.
    pass_start = time.monotonic()
    labels = np.full(sample_count, -1, dtype=np.intp)
    labels[core_indices] = core_cluster_labels
    non_core_indices = np.flatnonzero(~core_mask).astype(np.intp, copy=False)
    if core_count:
        for start in range(0, non_core_indices.size, block_rows):
            rows = non_core_indices[start:start + block_rows]
            distance_block = dist[np.ix_(rows, core_indices)]
            neighbor_mask = _epsilon_neighbors(distance_block, eps)
            tracker.sample()
            del distance_block
            for local_row, sample_index in enumerate(rows):
                core_neighbors = np.flatnonzero(neighbor_mask[local_row])
                if core_neighbors.size:
                    labels[int(sample_index)] = np.min(
                        core_cluster_labels[core_neighbors])
            del neighbor_mask
    tracker.sample()
    print(
        "[fast-dbscan] pass 3/3 border assignment: {:.3f}s, "
        "non-core={:,}, peak VmRSS={}".format(
            time.monotonic() - pass_start, non_core_indices.size,
            _format_bytes(tracker.peak)))
    return labels


def dump_cluster_distance_if_requested(
        dist, epoch, requested_epoch, output_dir):
    """Persist one ALL distance matrix for offline backend verification."""
    if requested_epoch is None or int(requested_epoch) < 0:
        return None
    if int(epoch) != int(requested_epoch):
        return None
    path = os.path.join(
        output_dir, "cluster_dist_all_epoch_{}.npy".format(int(epoch)))
    np.save(path, dist, allow_pickle=False)
    print("Saved ALL clustering distance matrix: {} (shape={}, dtype={})"
          .format(path, dist.shape, dist.dtype))
    return path
