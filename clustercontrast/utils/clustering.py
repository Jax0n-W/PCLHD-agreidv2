"""Shared DBSCAN construction for AG-ReID training engines."""
from __future__ import absolute_import

from sklearn.cluster import DBSCAN


DEFAULT_EPS = 0.3


def eps_log_suffix(eps):
    value = float(eps)
    if abs(value - DEFAULT_EPS) < 1e-12:
        return ""
    return "_eps{:g}".format(value)


def append_eps_log_suffix(name, eps):
    suffix = eps_log_suffix(eps)
    if not suffix or str(name).endswith(suffix):
        return name
    return "{}{}".format(name, suffix)


def _build_clusterers(names, eps, n_jobs):
    return {
        name: DBSCAN(
            eps=float(eps), min_samples=4,
            metric="precomputed", n_jobs=n_jobs)
        for name in names
    }


def build_agreid_stage1_clusterers(args):
    """Build aerial and ground DBSCAN instances from ``args.eps``."""
    return _build_clusterers(("ir", "rgb"), args.eps, -1)


def build_agreid_stage2_clusterers(args):
    """Build aerial, ground, and joint DBSCAN instances from ``args.eps``."""
    return _build_clusterers(("ir", "rgb", "all"), args.eps, -1)


def build_agreid_three_domain_clusterers(args):
    """Build far, near, ground, and joint DBSCAN instances."""
    return _build_clusterers(
        ("far", "near", "ground", "all"), args.eps,
        args.cluster_n_jobs)
