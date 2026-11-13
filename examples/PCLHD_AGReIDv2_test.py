#!/usr/bin/env python3
"""Evaluate an AG-ReID.v2 checkpoint on all four official protocols."""
from __future__ import absolute_import, print_function

import argparse
import csv
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
from torch.backends import cudnn


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agreid_identity import (  # noqa: E402
    DATASET_SPECS, parse_agreidv2_manifest,
    validate_agreidv2_protocol)
from clustercontrast.utils.data import transforms as T  # noqa: E402
from clustercontrast.utils.serialization import load_checkpoint  # noqa: E402
from examples.PCLHD_AGReID_g2ag import (  # noqa: E402
    TestData, create_model, eval_agreid, extract_feat_modal)


PROTOCOL_CAMERAS = {
    "exp1": ({0}, {3}),
    "exp2": ({0}, {2}),
    "exp4": ({3}, {0}),
    "exp5": ({2}, {0}),
}


def modal_for_camid(camid):
    """Map AG-ReID.v2 camera C to the unchanged PCLHD model branch."""
    camid = int(camid)
    if camid == 0:
        return 2
    if camid in (2, 3):
        return 1
    raise ValueError("Unsupported AG-ReID.v2 camera C{}".format(camid))


def _assert_protocol_cameras(protocol_key, query, gallery):
    expected_query, expected_gallery = PROTOCOL_CAMERAS[protocol_key]
    actual_query = {entry.camid for entry in query}
    actual_gallery = {entry.camid for entry in gallery}
    if (actual_query != expected_query
            or actual_gallery != expected_gallery):
        raise RuntimeError(
            "AG-ReID.v2 {} camera mismatch: expected query/gallery {}/{}, "
            "actual {}/{}".format(
                protocol_key, sorted(expected_query),
                sorted(expected_gallery), sorted(actual_query),
                sorted(actual_gallery)))


def extract_entries_features(model, entries, transform, args):
    """Extract in manifest order while routing every image by its C field."""
    features = np.zeros((len(entries), 2048), dtype=np.float32)
    for modal in (1, 2):
        indexes = [
            index for index, entry in enumerate(entries)
            if modal_for_camid(entry.camid) == modal
        ]
        if not indexes:
            continue
        paths = [entries[index].path for index in indexes]
        labels = np.zeros(len(paths), dtype=np.int64)
        dataset = TestData(
            paths, labels, transform=transform,
            img_size=(args.width, args.height))
        loader = data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)
        modal_features = extract_feat_modal(
            model, loader, len(paths), modal=modal)
        for local_index, entry_index in enumerate(indexes):
            features[entry_index] = modal_features[local_index]
    return features


def evaluate_protocol(model, data_dir, protocol_key, transform, args):
    spec = DATASET_SPECS["agreidv2"]["protocols"][protocol_key]
    query, gallery = parse_agreidv2_manifest(
        data_dir, protocol_key, validate=True)
    counts = validate_agreidv2_protocol(protocol_key, query, gallery)
    _assert_protocol_cameras(protocol_key, query, gallery)
    print("\n{} {}: query={} gallery={} identities(q/g/union)={}/{}/{}"
          .format(
              protocol_key, spec["name"], len(query), len(gallery),
              counts["query_identities"], counts["gallery_identities"],
              counts["union_identities"]))

    query_features = extract_entries_features(model, query, transform, args)
    gallery_features = extract_entries_features(
        model, gallery, transform, args)
    similarities = np.matmul(query_features, gallery_features.T)
    query_pids = np.asarray([entry.pid for entry in query])
    gallery_pids = np.asarray([entry.pid for entry in gallery])
    query_camids = np.asarray([entry.camid for entry in query])
    gallery_camids = np.asarray([entry.camid for entry in gallery])
    cmc, mean_ap, minp = eval_agreid(
        -similarities, query_pids, gallery_pids,
        q_camids=query_camids, g_camids=gallery_camids)
    print(
        "{}: Rank-1={:.2%} Rank-5={:.2%} Rank-10={:.2%} "
        "mAP={:.2%} mINP={:.2%}".format(
            protocol_key, cmc[0], cmc[4], cmc[9], mean_ap, minp))
    return {
        "protocol": protocol_key,
        "direction": spec["name"],
        "manifest": spec["manifest"],
        "query_images": len(query),
        "gallery_images": len(gallery),
        "identities": counts["union_identities"],
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mean_ap),
        "mINP": float(minp),
        "checkpoint": osp.abspath(osp.expanduser(args.resume)),
        "seed": args.seed,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="PCLHD evaluation on four official AG-ReID.v2 protocols")
    parser.add_argument("--data-dir", required=True,
                        help="Outer directory containing manifests and AG-ReID.v2/")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arch", default="agw")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--pooling-type", default="gem")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.data_dir = osp.abspath(osp.expanduser(args.data_dir))
    args.resume = osp.abspath(osp.expanduser(args.resume))
    if not osp.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    print("==> Build model and load checkpoint: {}".format(args.resume))
    model = create_model(args)
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.ToPILImage(), T.Resize((args.height, args.width)),
        T.ToTensor(), normalize])

    rows = []
    for protocol_key in DATASET_SPECS["agreidv2"]["protocols"]:
        rows.append(evaluate_protocol(
            model, args.data_dir, protocol_key, transform, args))

    output_csv = args.output_csv or osp.join(
        osp.dirname(args.resume),
        "agreidv2_results_seed{}.csv".format(args.seed))
    output_csv = osp.abspath(osp.expanduser(output_csv))
    output_dir = osp.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "protocol", "direction", "manifest", "query_images",
        "gallery_images", "identities", "rank1", "rank5", "rank10",
        "mAP", "mINP", "checkpoint", "seed"]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nAG-ReID.v2 results CSV: {}".format(output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
