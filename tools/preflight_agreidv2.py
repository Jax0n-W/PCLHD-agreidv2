#!/usr/bin/env python3
"""CPU-only AG-ReID.v2 data and protocol preflight."""
from __future__ import absolute_import, print_function

import argparse
import glob
import os.path as osp
import sys
from collections import Counter


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agreid_identity import (  # noqa: E402
    DATASET_SPECS, parse_agreid_name, parse_agreidv2_manifest,
    resolve_agreidv2_paths, validate_agreidv2_protocol)


def inspect_train(data_dir):
    _, inner_root = resolve_agreidv2_paths(data_dir)
    spec = DATASET_SPECS["agreidv2"]
    train_root = osp.join(inner_root, spec["train_dir"])
    if not osp.isdir(train_root):
        raise FileNotFoundError(
            "AG-ReID.v2 train_all not found: {}".format(train_root))
    paths = sorted(glob.glob(
        osp.join(train_root, "**", "*.jpg"), recursive=True))
    parsed = [parse_agreid_name(osp.basename(path)) for path in paths]
    camera_counts = Counter(item.camid for item in parsed)
    identity_count = len({item.pid for item in parsed})
    aerial_count = camera_counts.get(0, 0)
    ground_count = camera_counts.get(2, 0) + camera_counts.get(3, 0)
    print("train_all images: {}".format(len(paths)))
    print("train_all camera counts: {}".format(
        dict(sorted(camera_counts.items()))))
    print("train_all identities (P/T/A): {}".format(identity_count))
    print("training domains: aerial(C0)={} ground(C2+C3)={}".format(
        aerial_count, ground_count))
    if (len(paths) != spec["train_images"]
            or dict(camera_counts) != spec["camera_images"]
            or identity_count != spec["train_identities"]
            or aerial_count != 21217
            or ground_count != 30313):
        raise RuntimeError(
            "AG-ReID.v2 train_all mismatch: expected images={}, cameras={}, "
            "identities=807, aerial=21217, ground=30313; actual images={}, "
            "cameras={}, identities={}, aerial={}, ground={}".format(
                spec["train_images"], spec["camera_images"], len(paths),
                dict(sorted(camera_counts.items())), identity_count,
                aerial_count, ground_count))


def inspect_protocols(data_dir):
    for protocol_key, spec in DATASET_SPECS["agreidv2"]["protocols"].items():
        query, gallery = parse_agreidv2_manifest(
            data_dir, protocol_key, validate=True)
        actual = validate_agreidv2_protocol(protocol_key, query, gallery)
        print(
            "{} {}: query={} gallery={} identities(q/g/union)={}/{}/{}"
            .format(
                protocol_key, spec["name"], actual["query_images"],
                actual["gallery_images"], actual["query_identities"],
                actual["gallery_identities"], actual["union_identities"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate AG-ReID.v2 without loading a model or GPU")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args(argv)
    outer_root, inner_root = resolve_agreidv2_paths(args.data_dir)
    print("AG-ReID.v2 outer root: {}".format(outer_root))
    print("AG-ReID.v2 inner root: {}".format(inner_root))
    inspect_train(outer_root)
    inspect_protocols(outer_root)
    print("AGREIDV2_PREFLIGHT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
