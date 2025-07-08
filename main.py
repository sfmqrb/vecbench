#!/usr/bin/env python3
import os
import argparse
import time

from helper import read_fvecs, read_ground_truth, recall_at_k, BASE_DIR
from hnsw_faiss import HNSW_faiss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FAISS HNSW with fvecs data, report recall and query time."
    )
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        help="Dataset size suffix (e.g. 1M, 10M, 100M)",
    )
    parser.add_argument(
        "--ef_search", type=int, required=True, help="HNSW efSearch parameter"
    )
    args = parser.parse_args()

    size = args.size
    ef_search = args.ef_search

    # Fixed parameters
    M = 32
    ef_const = 500
    K = 10

    base_path = os.path.join(BASE_DIR, f"glove/glove_{size}_300d_test.fvecs")
    query_path = os.path.join(BASE_DIR, "glove/query.fvecs")
    gt_path = os.path.join(BASE_DIR, f"glove/glove_gt_{size}_50K.bin")
    idx_dir = os.path.join(BASE_DIR, "glove")
    idx_name = f"hnsw_M{M}_ef{ef_const}_{size}.faiss"
    idx_path = os.path.join(idx_dir, idx_name)

    print(f"Parameters: size={size}, ef_search={ef_search}, M={M}, ef_const={ef_const}")

    print("Loading base vectors…")
    xb = read_fvecs(base_path)
    print("Loading query vectors…")
    xq = read_fvecs(query_path)

    # Build or load index using the helper class
    d = xb.shape[1]
    hnsw = HNSW_faiss(d, ef_const, M, idx_path)
    hnsw.build(xb)
    if not os.path.exists(idx_path):
        hnsw.save(idx_path)
    index = hnsw.index

    # Search and time it
    index.hnsw.efSearch = ef_search
    print(f"Searching top-{K} for {xq.shape[0]} queries…")
    start_time = time.time()
    D, I = hnsw.search(xq, K)
    query_time = time.time() - start_time
    print(f"Query time: {query_time:.4f} seconds")

    # Evaluate recall
    print("Loading ground truth…")
    gt = read_ground_truth(gt_path)
    recall = recall_at_k(gt, I, K)
    print(f"Recall@{K}: {recall:.5f}")
