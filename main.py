#!/usr/bin/env python3
import os
import faiss
import argparse
import time

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

    base_path = f"/nvmepool/smaghrebi/glove/glove_{size}_300d_test.fvecs"
    query_path = "/nvmepool/smaghrebi/glove/query.fvecs"
    gt_path = f"/nvmepool/smaghrebi/glove/glove_gt_{size}_50K.bin"
    idx_dir = "/nvmepool/smaghrebi/glove"
    idx_name = f"hnsw_M{M}_ef{ef_const}_{size}.faiss"
    idx_path = os.path.join(idx_dir, idx_name)

    print(f"Parameters: size={size}, ef_search={ef_search}, M={M}, ef_const={ef_const}")

    print("Loading base vectors…")
    xb = read_fvecs(base_path)
    print("Loading query vectors…")
    xq = read_fvecs(query_path)

    # Build or load index
    if os.path.exists(idx_path):
        print(f"Loading existing index from {idx_path}")
        index = faiss.read_index(idx_path)
    else:
        print("Building new HNSW index…")
        d = xb.shape[1]
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = ef_const
        index.add(xb)
        print(f"Saving index to {idx_path}")
        faiss.write_index(index, idx_path)

    # Search and time it
    index.hnsw.efSearch = ef_search
    print(f"Searching top-{K} for {xq.shape[0]} queries…")
    start_time = time.time()
    D, I = index.search(xq, K)
    query_time = time.time() - start_time
    print(f"Query time: {query_time:.4f} seconds")

    # Evaluate recall
    print("Loading ground truth…")
    gt = read_ground_truth(gt_path)
    recall = recall_at_k(gt, I, K)
    print(f"Recall@{K}: {recall:.5f}")
