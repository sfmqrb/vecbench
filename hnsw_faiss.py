#!/usr/bin/env python3
import faiss
import os


class HNSW_faiss:
    def __init__(self, dim, ef_construction, m, index_path):
        self.dim = dim
        self.metric = faiss.METRIC_L2
        self.ef_construction = ef_construction
        self.m = m
        self.name = index_path
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efConstruction = ef_construction

    def build(self, vectors, force=False):
        if not force and os.path.exists(self.name):
            print(f"Index already exists at {self.name}. Loading...")
            self.index = self.load(self.name)
            return
        self.index.add(vectors)

    def search(self, query_vectors, k):
        return self.index.search(query_vectors, k)

    def save(self, path):
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
        return self.index
