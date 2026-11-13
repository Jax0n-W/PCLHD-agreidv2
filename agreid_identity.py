"""Canonical AG-ReID identity and AG-ReID.v2 protocol parsing.

AG-ReID identity is the concatenation of the P, T and A fields. Some local
training copies rewrite ``A{a}C{c}`` as one ``C{a}{c}`` token; this module
handles both forms without collapsing identities that share only P.
"""
from __future__ import absolute_import

import os.path as osp
import re
from collections import namedtuple


AGREID_TRAIN_IDENTITIES = 199
AGREID_TEST_IDENTITIES = 189

DATASET_SPECS = {
    "agreid": {
        "name": "AG-ReID.v1",
        "train_identities": AGREID_TRAIN_IDENTITIES,
        "test_identities": AGREID_TEST_IDENTITIES,
    },
    "agreidv2": {
        "name": "AG-ReID.v2",
        "inner_dir": "AG-ReID.v2",
        "train_dir": "train_all",
        "train_identities": 807,
        "train_images": 51530,
        "camera_images": {0: 21217, 2: 21298, 3: 9015},
        "protocols": {
            "exp1": {
                "name": "A-to-C",
                "manifest": "exp1_aerial_to_cctv.txt",
                "query_images": 2356,
                "gallery_images": 6347,
                "identities": 534,
            },
            "exp2": {
                "name": "A-to-W",
                "manifest": "exp2_aerial_to_wearable.txt",
                "query_images": 2209,
                "gallery_images": 12912,
                "identities": 519,
            },
            "exp4": {
                "name": "C-to-A",
                "manifest": "exp4_cctv_to_aerial.txt",
                "query_images": 1811,
                "gallery_images": 14362,
                "identities": 534,
            },
            "exp5": {
                "name": "W-to-A",
                "manifest": "exp5_wearable_to_aerial.txt",
                "query_images": 2340,
                "gallery_images": 12568,
                "identities": 519,
            },
        },
    },
}

AGReIDName = namedtuple(
    "AGReIDName",
    ("pid", "camid", "encoded_camid", "person", "tracklet", "angle",
     "source"))
AGReIDProtocolEntry = namedtuple(
    "AGReIDProtocolEntry",
    ("path", "pid", "camid", "encoded_camid", "relative_path"))

_OFFICIAL_RE = re.compile(
    r"^P(?P<p>\d+)_?T(?P<t>\d+)_?A(?P<a>\d+)_?C(?P<c>\d+)",
    re.IGNORECASE)
_REWRITTEN_RE = re.compile(
    r"^P(?P<p>\d+)_T(?P<t>\d+)_C(?P<ac>\d+)_F\d+",
    re.IGNORECASE)


def compose_agreid_pid(person, tracklet, angle):
    """Return the official integer label formed by concatenating P/T/A."""
    fields = (str(person), str(tracklet), str(angle))
    if any(not field.isdigit() for field in fields):
        raise ValueError("AG-ReID P/T/A fields must contain digits: {}".format(
            fields))
    return int("".join(fields))


def parse_agreid_name(filename):
    """Parse an official or locally rewritten AG-ReID image filename.

    Official: ``P0001T04041A0C0F121.jpg``
    Rewritten: ``P0001_T04041_C00_F121.jpg`` where C00 stores A=0, C=0.

    For rewritten training files ``camid`` deliberately remains the complete
    encoded C token (for example C10 -> 10), preserving the historical camera
    metadata used by training samplers. Only the identity label is corrected.
    """
    name = osp.basename(str(filename))
    match = _OFFICIAL_RE.match(name)
    if match is not None:
        person = match.group("p")
        tracklet = match.group("t")
        angle = match.group("a")
        return AGReIDName(
            compose_agreid_pid(person, tracklet, angle),
            int(match.group("c")), int(angle + match.group("c")),
            person, tracklet, angle, "official")

    match = _REWRITTEN_RE.match(name)
    if match is not None:
        encoded_camera = match.group("ac")
        if len(encoded_camera) < 2:
            raise ValueError(
                "Rewritten AG-ReID C token must encode A and C: {}".format(
                    name))
        person = match.group("p")
        tracklet = match.group("t")
        angle = encoded_camera[0]
        return AGReIDName(
            compose_agreid_pid(person, tracklet, angle),
            int(encoded_camera), int(encoded_camera),
            person, tracklet, angle, "rewritten")

    raise ValueError("Cannot parse AG-ReID P/T/A identity from: {}".format(
        name))


def validate_identity_count(samples, expected, split_name):
    """Fail loudly when a complete official split has the wrong ID count."""
    identities = {int(sample[1]) for sample in samples}
    if identities and len(identities) != int(expected):
        raise RuntimeError(
            "AG-ReID {} identity count mismatch: expected {}, found {}. "
            "Check P/T/A parsing and the dataset split.".format(
                split_name, expected, len(identities)))
    return len(identities)


def get_dataset_spec(dataset_key):
    """Return a dataset specification without silently accepting typos."""
    try:
        return DATASET_SPECS[dataset_key]
    except KeyError:
        raise ValueError(
            "Unknown AG-ReID dataset key {!r}; expected one of {}".format(
                dataset_key, ", ".join(sorted(DATASET_SPECS))))


def resolve_agreidv2_paths(data_dir):
    """Return the documented outer root and inner ``AG-ReID.v2`` root."""
    outer_root = osp.abspath(osp.expanduser(str(data_dir)))
    inner_root = osp.join(
        outer_root, DATASET_SPECS["agreidv2"]["inner_dir"])
    return outer_root, inner_root


def get_agreidv2_protocol(protocol_key):
    """Return one official v2 protocol specification."""
    protocols = DATASET_SPECS["agreidv2"]["protocols"]
    try:
        return protocols[protocol_key]
    except KeyError:
        raise ValueError(
            "Unknown AG-ReID.v2 protocol {!r}; expected one of {}".format(
                protocol_key, ", ".join(sorted(protocols))))


def parse_agreidv2_manifest(data_dir, protocol_key, validate=True):
    """Parse one official v2 manifest into query and gallery entries.

    Manifest paths are relative to the inner ``AG-ReID.v2`` directory.
    A line beginning with ``query/`` belongs to the query split; every other
    non-empty line belongs to the gallery split.
    """
    outer_root, inner_root = resolve_agreidv2_paths(data_dir)
    protocol = get_agreidv2_protocol(protocol_key)
    manifest_path = osp.join(outer_root, protocol["manifest"])
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(
            "AG-ReID.v2 manifest not found: {}".format(manifest_path))
    if not osp.isdir(inner_root):
        raise FileNotFoundError(
            "AG-ReID.v2 inner directory not found: {}".format(inner_root))

    query = []
    gallery = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            relative_path = raw_line.strip()
            if not relative_path:
                continue
            normalized = relative_path.replace("/", osp.sep)
            full_path = osp.normpath(osp.join(inner_root, normalized))
            try:
                common = osp.commonpath((inner_root, full_path))
            except ValueError:
                common = ""
            if osp.normcase(common) != osp.normcase(inner_root):
                raise RuntimeError(
                    "AG-ReID.v2 manifest path escapes the dataset root at "
                    "{}:{}: {}".format(
                        manifest_path, line_number, relative_path))
            if validate and not osp.isfile(full_path):
                raise FileNotFoundError(
                    "AG-ReID.v2 manifest image not found at {}:{}: {}".format(
                        manifest_path, line_number, full_path))
            parsed = parse_agreid_name(osp.basename(relative_path))
            entry = AGReIDProtocolEntry(
                full_path, parsed.pid, parsed.camid,
                parsed.encoded_camid, relative_path)
            first_component = relative_path.replace("\\", "/").split("/", 1)[0]
            if first_component == "query":
                query.append(entry)
            else:
                gallery.append(entry)

    if validate:
        validate_agreidv2_protocol(protocol_key, query, gallery)
    return query, gallery


def validate_agreidv2_protocol(protocol_key, query, gallery):
    """Validate official v2 image and identity counts, reporting actuals."""
    protocol = get_agreidv2_protocol(protocol_key)
    query_ids = {int(entry.pid) for entry in query}
    gallery_ids = {int(entry.pid) for entry in gallery}
    union_ids = query_ids | gallery_ids
    actual = {
        "query_images": len(query),
        "gallery_images": len(gallery),
        "query_identities": len(query_ids),
        "gallery_identities": len(gallery_ids),
        "union_identities": len(union_ids),
    }
    expected_identities = int(protocol["identities"])
    valid = (
        actual["query_images"] == int(protocol["query_images"])
        and actual["gallery_images"] == int(protocol["gallery_images"])
        and actual["query_identities"] == expected_identities
        and actual["gallery_identities"] == expected_identities
        and actual["union_identities"] == expected_identities
    )
    if not valid:
        raise RuntimeError(
            "AG-ReID.v2 {} count mismatch: expected query/gallery images "
            "{}/{}, expected identities {}; actual query/gallery images "
            "{}/{}, actual query/gallery/union identities {}/{}/{}".format(
                protocol_key, protocol["query_images"],
                protocol["gallery_images"], expected_identities,
                actual["query_images"], actual["gallery_images"],
                actual["query_identities"], actual["gallery_identities"],
                actual["union_identities"]))
    return actual
