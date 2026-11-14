#!/usr/bin/env python3
"""Compare sklearn and fast DBSCAN on a saved real distance matrix."""
from __future__ import print_function

import argparse
import json
import multiprocessing
import os
import os.path as osp
import sys
import tempfile
import time

import numpy as np
from sklearn.cluster import DBSCAN


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustercontrast.utils.fast_dbscan import (  # noqa: E402
    fast_dbscan_precomputed, read_vmrss_bytes)


def _format_bytes(value):
    if value is None:
        return "unavailable"
    return "{:.3f} GiB".format(float(value) / (1024 ** 3))


def _worker(matrix_path, backend, eps, min_samples, labels_path,
            metrics_path):
    dist = np.load(matrix_path, allow_pickle=False)
    start = time.monotonic()
    if backend == "sklearn":
        labels = DBSCAN(
            eps=eps, min_samples=min_samples,
            metric="precomputed", n_jobs=-1).fit_predict(dist)
    else:
        labels = fast_dbscan_precomputed(dist, eps, min_samples)
    elapsed = time.monotonic() - start
    np.save(labels_path, labels, allow_pickle=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({
            "backend": backend,
            "seconds": elapsed,
            "shape": list(dist.shape),
            "dtype": str(dist.dtype),
        }, handle, indent=2, sort_keys=True)


def _run_backend(matrix_path, backend, eps, min_samples, directory):
    labels_path = osp.join(directory, backend + "_labels.npy")
    metrics_path = osp.join(directory, backend + "_metrics.json")
    process = multiprocessing.Process(
        target=_worker,
        args=(matrix_path, backend, eps, min_samples,
              labels_path, metrics_path))
    process.start()
    peak_rss = None
    while process.is_alive():
        current = read_vmrss_bytes(process.pid)
        if current is not None and (peak_rss is None or current > peak_rss):
            peak_rss = current
        process.join(timeout=0.05)
    current = read_vmrss_bytes(process.pid)
    if current is not None and (peak_rss is None or current > peak_rss):
        peak_rss = current
    process.join()
    if process.exitcode:
        raise RuntimeError(
            "{} worker exited with code {}".format(
                backend, process.exitcode))
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["peak_rss"] = peak_rss
    labels = np.load(labels_path, allow_pickle=False)
    print("{}: {:.3f}s, peak VmRSS={}, labels={:,}".format(
        backend, metrics["seconds"], _format_bytes(peak_rss), labels.size))
    return labels, metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify fast DBSCAN against sklearn on a .npy matrix")
    parser.add_argument("matrix", help="saved square Jaccard distance .npy")
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--min-samples", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    matrix_path = osp.abspath(osp.expanduser(args.matrix))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square, got {}".format(matrix.shape))
    print("Matrix: {} | shape={} | dtype={} | size={}".format(
        matrix_path, matrix.shape, matrix.dtype,
        _format_bytes(matrix.nbytes)), flush=True)
    del matrix

    with tempfile.TemporaryDirectory(prefix="verify_fast_dbscan_") as temp_dir:
        sklearn_labels, sklearn_metrics = _run_backend(
            matrix_path, "sklearn", args.eps, args.min_samples, temp_dir)
        fast_labels, fast_metrics = _run_backend(
            matrix_path, "fast", args.eps, args.min_samples, temp_dir)
        equal = np.array_equal(sklearn_labels, fast_labels)
        mismatch_count = int(np.count_nonzero(
            sklearn_labels != fast_labels))
        print("Labels exactly equal: {} (mismatches={:,})".format(
            equal, mismatch_count))
        if not equal:
            first = int(np.flatnonzero(
                sklearn_labels != fast_labels)[0])
            print("First mismatch at {}: sklearn={}, fast={}".format(
                first, sklearn_labels[first], fast_labels[first]))
            return 1
        speedup = (sklearn_metrics["seconds"] / fast_metrics["seconds"]
                   if fast_metrics["seconds"] else float("inf"))
        print("Speedup (sklearn/fast): {:.3f}x".format(speedup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
