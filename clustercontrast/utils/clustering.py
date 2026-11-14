"""Shared DBSCAN construction for AG-ReID training engines."""
from __future__ import absolute_import

from sklearn.cluster import DBSCAN

from .fast_dbscan import fast_dbscan_precomputed


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


def resolve_eps_all(args):
    """Return the joint-clustering eps, inheriting ``args.eps`` by default."""
    value = getattr(args, "eps_all", None)
    return float(args.eps if value is None else value)


def eps_all_log_suffix(eps, eps_all):
    """Name only runs whose Stage-2 ALL eps differs from the domain eps."""
    value = float(eps if eps_all is None else eps_all)
    if abs(value - float(eps)) < 1e-12:
        return ""
    return "_epsall{:g}".format(value)


def append_eps_all_log_suffix(name, eps, eps_all):
    suffix = eps_all_log_suffix(eps, eps_all)
    if not suffix or str(name).endswith(suffix):
        return name
    return "{}{}".format(name, suffix)


def append_stage2_eps_log_suffix(name, eps, eps_all):
    """Append the complete Stage-2 eps suffix exactly once."""
    suffix = eps_log_suffix(eps) + eps_all_log_suffix(eps, eps_all)
    if not suffix or str(name).endswith(suffix):
        return name
    result = append_eps_log_suffix(name, eps)
    return append_eps_all_log_suffix(result, eps, eps_all)


class PrecomputedDBSCAN(object):
    """Small backend-switching facade used by every AG-ReID clusterer."""

    def __init__(self, eps, min_samples=4, n_jobs=-1, backend="sklearn"):
        if backend not in ("sklearn", "fast"):
            raise ValueError("unknown DBSCAN backend: {}".format(backend))
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.n_jobs = n_jobs
        self.backend = backend
        self._sklearn = None
        if backend == "sklearn":
            self._sklearn = DBSCAN(
                eps=self.eps, min_samples=self.min_samples,
                metric="precomputed", n_jobs=self.n_jobs)

    def fit_predict(self, dist):
        if self.backend == "fast":
            return fast_dbscan_precomputed(
                dist, eps=self.eps, min_samples=self.min_samples)
        self._sklearn.eps = self.eps
        return self._sklearn.fit_predict(dist)


def _build_clusterers(names, eps, n_jobs, backend):
    return {
        name: PrecomputedDBSCAN(
            eps=float(eps), min_samples=4, n_jobs=n_jobs,
            backend=backend)
        for name in names
    }


def build_agreid_stage1_clusterers(args):
    """Build aerial and ground DBSCAN instances from ``args.eps``."""
    return _build_clusterers(
        ("ir", "rgb"), args.eps, -1,
        getattr(args, "dbscan_backend", "sklearn"))


def build_agreid_stage2_clusterers(args):
    """Build domain DBSCANs from eps and the joint DBSCAN from eps_all."""
    backend = getattr(args, "dbscan_backend", "sklearn")
    clusterers = _build_clusterers(("ir", "rgb"), args.eps, -1, backend)
    clusterers.update(_build_clusterers(
        ("all",), resolve_eps_all(args), -1, backend))
    return clusterers


def build_agreid_three_domain_clusterers(args):
    """Build far, near, ground, and joint DBSCAN instances."""
    backend = getattr(args, "dbscan_backend", "sklearn")
    clusterers = _build_clusterers(
        ("far", "near", "ground"), args.eps, args.cluster_n_jobs,
        backend)
    clusterers.update(_build_clusterers(
        ("all",), resolve_eps_all(args), args.cluster_n_jobs, backend))
    return clusterers
