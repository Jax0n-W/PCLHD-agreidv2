from __future__ import absolute_import, print_function

import os.path as osp

from ..utils.data import BaseImageDataset
from ._agreidv2 import build_unlabeled_domain


class agreidv2_ir(BaseImageDataset):
    """AG-ReID.v2 aerial training domain (C0, model modal=2)."""

    def __init__(self, root, trial=1, verbose=True, **kwargs):
        super(agreidv2_ir, self).__init__()
        normalized_root = osp.abspath(osp.expanduser(root))
        if osp.basename(osp.normpath(normalized_root)).lower() in (
                "agreidv2_ir", "agreidv2_rgb"):
            normalized_root = osp.dirname(osp.normpath(normalized_root))

        self.root, self.dataset_dir, self.train_dir, self.train = \
            build_unlabeled_domain(normalized_root, (0,), 21217)
        self.query = []
        self.gallery = []
        self.num_train_pids = 0
        self.num_train_imgs = len(self.train)
        self.num_train_cams = len({sample[2] for sample in self.train})
        self.num_query_pids = self.num_query_imgs = self.num_query_cams = 0
        self.num_gallery_pids = self.num_gallery_imgs = \
            self.num_gallery_cams = 0

        if verbose:
            print("=> agreidv2_ir (aerial C0) loaded: {} unlabeled images"
                  .format(self.num_train_imgs))
