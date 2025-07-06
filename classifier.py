# classifier.py
import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

class Classifier:
    def __init__(self, json_path="hierarchy.json"):
        self.json_tree = json.load(open(json_path))
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self._build_index()

    def _build_index(self):
        leaf_records = []
        def collect(node, path):
            if "phrase" in node:
                leaf_records.append({"leaf_id": node["id"],
                                     "path": " ▸ ".join(path),
                                     "examples": [node["phrase"]]})
            else:
                for ch in node["children"]:
                    collect(ch, path+[node["name"]])
        collect(self.json_tree, [self.json_tree["name"]])
        self.id2rec = {rec["leaf_id"]: rec for rec in leaf_records}
        self.leaf_ids = [rec["leaf_id"] for rec in leaf_records]
        self.embeds = np.stack([
            self.model.encode(rec["examples"], normalize_embeddings=True).mean(axis=0)
            for rec in leaf_records
        ]).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeds.shape[1])
        self.index.add(self.embeds)

    def classify(self, text: str, top_k: int = 3):
        q = self.model.encode(text, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            rec = self.id2rec[self.leaf_ids[idx]]
            results.append({
                "score": float(score),
                "leaf_id": rec["leaf_id"],
                "name": rec["examples"][0],
                "path": rec["path"]
            })
        return results