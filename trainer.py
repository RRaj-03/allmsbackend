# trainer.py
import importlib, subprocess, sys, re, math, json, os
from collections import Counter, defaultdict
import base64
import io
# ── install lightweight deps silently ────────────────────────────
for pkg in ("numpy", "scipy", "matplotlib", "scikit-learn",
            "transformers", "sentencepiece", "accelerate"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── std + scientific imports ─────────────────────────────────────
import numpy as np
from scipy.sparse             import csr_matrix
from scipy.cluster.hierarchy  import linkage, cophenet, fcluster,dendrogram
from scipy.spatial.distance   import squareform
from sklearn.metrics          import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt   
import matplotlib
matplotlib.use('Agg')                              # comment if headless
from transformers import pipeline                               # HF zero-shot naming


def train_and_save(phrases: list[str], output_path="hierarchy.json"):
    # Insert your full pipeline here replacing:
    # phrases = [...]
    # return root dictionary at the end
    # ================================================================
# 1-CELL PIPELINE  —  clustering • metrics • HF zero-shot naming
# ================================================================

# ================================================================
# DATA  (swap in your 750 phrases)
# ================================================================
    # phrases = ['apple','mango','banana','lichi','guava','flower']
#     phrases = [
#     "order physical card", "lost or stolen card", "cancel transfer",
#     "balance not updated after bank transfer", "card not working",
#     "cash withdrawal charge", "edit personal details", "top up by card charge",
#     "verify my identity", "request refund"
# ]
# ================================================================
# TOKENISE → sparse probability matrix
# ================================================================
    tokeniser = re.compile(r"\b\w+\b").findall
    tokenised  = [tokeniser(t.lower()) for t in phrases]
    vocab      = sorted({tok for sent in tokenised for tok in sent})
    v2i        = {w: i for i, w in enumerate(vocab)}

    rows, cols, data = [], [], []
    for r, sent in enumerate(tokenised):
        cnt = Counter(sent); tot = sum(cnt.values())
        for w, c in cnt.items():
            rows.append(r); cols.append(v2i[w]); data.append(c / tot)
    X = csr_matrix((data, (rows, cols)), shape=(len(phrases), len(vocab)))

    # ================================================================
    # global stats for divergences
    # ================================================================
    N = len(phrases)
    df = Counter(tok for sent in tokenised for tok in set(sent))
    idf_vec = np.array([math.log(N / df[w]) for w in vocab])
    idf_map = {i: idf_vec[i] for i in range(len(vocab))}

    bg_counts = Counter(tok for sent in tokenised for tok in sent)
    bg_total  = sum(bg_counts.values())
    bg        = np.array([bg_counts.get(w, 0)/bg_total for w in vocab])
    beta_vec  = 0.2 * idf_vec / idf_vec.max()
    eps = 1e-9

    # ================================================================
    # divergence definitions
    # ================================================================
    def wkls(u, v):
        out = 0.0
        for idx, pu in zip(u.indices, u.data):
            pv = v[0, idx]; w = idf_map[idx]
            out += w * pu * math.log((pu+eps)/(pv+eps))
        for idx, pv in zip(v.indices, v.data):
            pu = u[0, idx]; w = idf_map[idx]
            out += w * pv * math.log((pv+eps)/(pu+eps))
        return out

    def ap_jsd(u, v):
        p = u.toarray().ravel(); q = v.toarray().ravel()
        p_ = (1-beta_vec)*p + beta_vec*bg
        q_ = (1-beta_vec)*q + beta_vec*bg
        m  = 0.5*(p_+q_)
        return 0.5*np.sum(p_*np.log((p_+eps)/(m+eps))) + \
            0.5*np.sum(q_*np.log((q_+eps)/(m+eps)))

    def renyi_jsd(u, v, a=0.8):
        p = u.toarray().ravel()+eps; q = v.toarray().ravel()+eps
        m = 0.5*(p+q)
        num   = (m**a).sum()
        denom = 0.5*(p**a).sum() + 0.5*(q**a).sum()
        return max((math.log(num)-math.log(denom))/(a-1), 0.0)

    divergences = {"WKLS": wkls, "AP-JSD": ap_jsd, #"Rényi-JSD": renyi_jsd
                }

    def condensed(div):
        n = X.shape[0]; out = np.empty(n*(n-1)//2)
        k = 0
        for i in range(n-1):
            ui = X.getrow(i)
            for j in range(i+1, n):
                out[k] = div(ui, X.getrow(j)); k += 1
        return out

    # ================================================================
    # choose best divergence by silhouette (k=5)
    # ================================================================
    k_eval = 5
    best_name, best_Z, best_D = None, None, None
    best_sil = -1.0
    dendrogram_images = {}
    for name, f in divergences.items():
        D = condensed(f);  Z = linkage(D, method="complete")
        sil = silhouette_score(squareform(D),
                            fcluster(Z, k_eval, criterion="maxclust"),
                            metric="precomputed")
        print(f"{name:8s} silhouette@{k_eval}: {sil:5.3f}")
        if sil > best_sil:
            best_name, best_Z, best_D, best_sil = name, Z, D, sil
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, labels=phrases, leaf_rotation=90, leaf_font_size=8, ax=ax)
        plt.title(f"{name} dendrogram")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        dendrogram_images[name] = img_base64
    print(f"\n>> chosen divergence: {best_name}\n")

    # ================================================================
    # build parent/child dictionaries from linkage
    # ================================================================
    n = len(phrases)
    parents  = {}
    children = defaultdict(list)
    for idx, (a, b, _, _) in enumerate(best_Z, start=n):
        parents[int(a)] = parents[int(b)] = idx
        children[idx].extend([int(a), int(b)])

    # ================================================================
    # Hugging-Face zero-shot naming model
    # ================================================================
    hf_gen = pipeline("text2text-generation",
                    model="google/flan-t5-small",
                    max_length=8,
                    do_sample=False)

    def zero_shot_name(cands):
        prompt = ("Give a concise 3-word category title for: " +
                "; ".join(cands))
        text = hf_gen(prompt, num_return_sequences=1)[0]["generated_text"]
        # take first ≤4 words
        return " ".join(text.split()[:4]).title()

    def tfidf_terms(member_ids, k=6):
        texts = [phrases[i] for i in member_ids]
        vec   = TfidfVectorizer(ngram_range=(1,2), stop_words="english").fit(texts)
        scores = np.asarray(vec.transform(texts).sum(axis=0)).ravel()
        idxs   = scores.argsort()[::-1][:k]
        return [vec.get_feature_names_out()[i] for i in idxs]

    # ================================================================
    # build node representations bottom-up
    # ================================================================
    node = {i: {"id": i, "phrase": phrases[i]} for i in range(n)}

    # internal nodes
    for idx in range(n, n+len(best_Z)):
        # gather all descendant leaves
        stack = [idx]; leaves = []
        while stack:
            v = stack.pop()
            if v < n: leaves.append(v)
            else:     stack.extend(children[v])
        cands = tfidf_terms(leaves)
        node[idx] = {
            "id": idx,
            "name": zero_shot_name(cands),
            "children": [node[c] for c in children[idx]]
        }

    root = node[n+len(best_Z)-1]

    # ================================================================
    # save + preview JSON
    # ================================================================
    with open("hierarchy.json", "w") as f:
        json.dump(root, f, indent=2)
    print(json.dumps(root, indent=2)[:1200], "\n...\n(JSON truncated)")
    print("\nHierarchy saved to hierarchy.json")
    return dendrogram_images
