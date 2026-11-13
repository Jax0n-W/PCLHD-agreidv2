from __future__ import absolute_import, print_function

import glob
import os.path as osp
from collections import Counter

from agreid_identity import (
    DATASET_SPECS, parse_agreid_name, resolve_agreidv2_paths)


UNLABELED_PID = -1


def load_agreidv2_train_records(root):
    """Load and validate the complete v2 train split from image filenames."""
    outer_root, inner_root = resolve_agreidv2_paths(root)
    train_dir = osp.join(
        inner_root, DATASET_SPECS["agreidv2"]["train_dir"])
    if not osp.isdir(train_dir):
        raise RuntimeError(
            "AG-ReID.v2 training directory is not available: {}".format(
                train_dir))

    image_paths = sorted(glob.glob(
        osp.join(train_dir, "**", "*.jpg"), recursive=True))
    parsed_records = [
        (osp.abspath(path), parse_agreid_name(osp.basename(path)))
        for path in image_paths
    ]

    spec = DATASET_SPECS["agreidv2"]
    camera_counts = Counter(record.camid for _, record in parsed_records)
    identity_count = len({record.pid for _, record in parsed_records})
    expected_cameras = dict(spec["camera_images"])
    if (len(parsed_records) != spec["train_images"]
            or dict(camera_counts) != expected_cameras
            or identity_count != spec["train_identities"]):
        raise RuntimeError(
            "AG-ReID.v2 train_all count mismatch: expected images={}, "
            "camera_counts={}, identities={}; actual images={}, "
            "camera_counts={}, identities={}".format(
                spec["train_images"], expected_cameras,
                spec["train_identities"], len(parsed_records),
                dict(sorted(camera_counts.items())), identity_count))
    return outer_root, inner_root, train_dir, parsed_records


def build_unlabeled_domain(root, allowed_cameras, expected_images):
    """Return one camera-defined domain without exposing true IDs to training."""
    outer_root, inner_root, train_dir, records = \
        load_agreidv2_train_records(root)
    allowed_cameras = set(allowed_cameras)
    train = [
        (path, UNLABELED_PID, record.encoded_camid)
        for path, record in records if record.camid in allowed_cameras
    ]
    if len(train) != int(expected_images):
        raise RuntimeError(
            "AG-ReID.v2 domain count mismatch for cameras {}: expected {}, "
            "found {}".format(
                sorted(allowed_cameras), expected_images, len(train)))
    return outer_root, inner_root, train_dir, train
