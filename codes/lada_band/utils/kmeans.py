# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm
import glob
import pickle


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            # print(x.dtype, self.C.dtype)
            x = x.to(self.C.dtype)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)

def kmeans_q(kmeans_model: ApplyKmeans, embedds: torch.Tensor):
    embedds = embedds.to(kmeans_model.C)
    feat_q = kmeans_model(embedds)
    return feat_q


def dump_label(feat_dir, km_path, lab_dir):
    apply_kmeans = ApplyKmeans(km_path)
    pkl_fp_list = glob.glob(os.path.join(feat_dir, '*.pkl'))
    for pkl_fp in tqdm.tqdm(pkl_fp_list):
        pkl_name = os.path.basename(pkl_fp)
        with open(pkl_fp, 'rb') as f:
            feat = pickle.load(f)
            f.close()
        feat_q = apply_kmeans(feat.cuda())
        with open(os.path.join(lab_dir, pkl_name), 'wb') as f:
            pickle.dump(feat_q, f)
            f.close()
        # break


    # generator, num = get_feat_iterator(feat_dir, split, nshard, rank)
    # iterator = generator()

    # lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
    # os.makedirs(lab_dir, exist_ok=True)
    # with open(lab_path, "w") as f:
    #     for feat in tqdm.tqdm(iterator, total=num):
    #         # feat = torch.from_numpy(feat).cuda()
    #         lab = apply_kmeans(feat).tolist()
    #         f.write(" ".join(map(str, lab)) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("km_path")
    parser.add_argument("lab_dir")
    args = parser.parse_args()
    os.makedirs(args.lab_dir, exist_ok=True)
    logging.info(str(args))

    dump_label(**vars(args))