#!/usr/bin/env python3
import struct
import numpy as np

BASE_DIR = "/nvmepool/smaghrebi"

def read_fvecs(filename):
    with open(filename, "rb") as f:
        content = f.read()
    dim = struct.unpack("i", content[:4])[0]
    record_size = 4 + 4 * dim
    num_vecs = len(content) // record_size
    data = np.zeros((num_vecs, dim), dtype=np.float32)
    for i in range(num_vecs):
        offset = i * record_size
        start = offset + 4
        data[i] = np.frombuffer(content[start : start + 4 * dim], dtype=np.float32)
    return data


def read_ground_truth(filename):
    with open(filename, "rb") as f:
        Q = struct.unpack("I", f.read(4))[0]
        K = struct.unpack("I", f.read(4))[0]
        idx = np.frombuffer(f.read(4 * Q * K), dtype=np.uint32).reshape(Q, K)
    return idx


def recall_at_k(gt, pred, k):
    hits = sum(len(set(gt[i, :k]) & set(pred[i, :k])) for i in range(gt.shape[0]))
    return hits / (gt.shape[0] * k)


