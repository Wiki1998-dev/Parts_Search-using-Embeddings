
import pandas as pd
import argparse
from pathlib import Path
from typing import  List
import numpy as np

from sentence_transformers import SentenceTransformer

# ---------------------------
# Data loading & preparation
# ---------------------------
def load_parts_csv(path: Path) -> pd.DataFrame:
    """
    Load the semicolon-delimited Parts.csv and do light normalization.

    """
    df = pd.read_csv(path, delimiter=";", engine="python")
    if "DESCRIPTION" not in df.columns:
        raise ValueError("Input must contain a 'DESCRIPTION' column.")
    # Normalize DESCRIPTION & ID
    df["DESCRIPTION"] = df["DESCRIPTION"].fillna("").astype(str)
    if "ID" not in df.columns:
        # Create a synthetic ID if none is provided
        df["ID"] = [f"PART_{i:05d}" for i in range(len(df))]

    # Build a compact "context string" from a few useful columns as optional augmentation
    likely_text_cols = [
        "Material", "Size", "Rating", "Characteristic", "Application", "Code",
        "Rated Current (A)", "Rated Voltage (V)", "Temp", "Height", "Length in mm",
        "Rated Voltage(AC) (V)", "Rated Voltage(DC) (V)"
    ]
    chosen = [c for c in likely_text_cols if c in df.columns]
    def join_context(row) -> str:
        bits = []
        for c in chosen:
            v = row.get(c)
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower() != "nan":
                    bits.append(f"{c}={s}")
        return " ; ".join(bits)

    df["_CONTEXT_"] = df.apply(join_context, axis=1)
    return df

def filter_rows_with_signal(df: pd.DataFrame):
    """Keep rows that have either DESCRIPTION or _CONTEXT_ non-empty."""
    has_desc = df["DESCRIPTION"].fillna("").str.strip().astype(bool)
    has_ctx  = df["_CONTEXT_"].fillna("").str.strip().astype(bool)
    has_signal = has_desc | has_ctx
    dropped = df.loc[~has_signal, ["ID", "DESCRIPTION", "_CONTEXT_"]].copy()
    kept = df.loc[has_signal].reset_index(drop=True)
    return kept, dropped
# ---------------------------
# Descriptive analysis
# ---------------------------
def descriptive_analysis(df: pd.DataFrame) -> None:

    n_rows, n_cols = df.shape


    missing_desc = (df["DESCRIPTION"].str.len() == 0).sum()
    print(f"Empty DESCRIPTION rows: {missing_desc:,} ({missing_desc/max(1,n_rows):.1%})")

    dup_desc = df["DESCRIPTION"].duplicated(keep=False).sum()
    print(f"Rows with non-unique DESCRIPTION: {dup_desc:,} ({dup_desc/max(1,n_rows):.1%})")

# ---------------------------
# Text building
# ---------------------------
def build_texts(df: pd.DataFrame) -> List[str]:
    """
    Construct the  synthetic description to missing ones from the parts
    """
    texts: List[str] = []
    for desc, ctx in zip(df["DESCRIPTION"].tolist(), df["_CONTEXT_"].tolist()):
        desc = desc.strip()
        # fallback to context if desc is empty
        full = desc if desc else ctx
        texts.append(full.strip())
    return texts


# ---------------------------
# Embedding backends
# ---------------------------
def embed_tfidf(texts: List[str]):
    """TF-IDF baseline: returns (embeddings as sparse matrix, 'tfidf')."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vec.fit_transform(texts)
    # We’ll handle similarity with sklearn cosine for sparse.
    return X, "tfidf"


def embed_bge(texts: List[str], model_name: str = "BAAI/bge-large-en-v1.5"):
    """
    BGE embeddings via SentenceTransformers.
    L2-normalize so cosine == dot product.
    """
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True).astype("float32")
    # emb :dense (n, d) npy array
    return emb.astype("float32"), "bge"


# ---------------------------
# Similarity search
# ---------------------------
def topk_from_similarity_matrix(sim: np.ndarray, k: int) -> np.ndarray:
    """
    Given a dense similarity matrix, return top-k neighbor indices for each row,
    excluding self. Works for cosine similarities in [-1, 1].
    """
    n = sim.shape[0]
    # Use argpartition for speed, then refine by sorting a small slice.
    idx = np.argpartition(-sim, kth=min(k, n - 1), axis=1)[:, : (k + 1)]
    small_sorted = np.take_along_axis(
        idx,
        np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1),
        axis=1
    )
    out = []
    for i in range(n):
        row = [j for j in small_sorted[i] if j != i][:k]
        if len(row) < k:
            # Extremely rare: if self wasn’t in the slice, do a full sort fallback
            order = np.argsort(-sim[i])
            row = [j for j in order if j != i][:k]
        out.append(row)
    return np.array(out, dtype=int)


def similarity_and_topk(emb, backend: str, k: int) -> np.ndarray:
    """
    Compute similarities and get top-k indices.
    - For TF-IDF (sparse), we’ll use sklearn cosine_similarity (dense result).
    - For BGE (dense, normalized), dot product == cosine, so we can do a fast gemm.
    """
    if backend == "tfidf":
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(emb, emb)  # returns dense (n, n)
        return topk_from_similarity_matrix(sim, k=k)

    # Dense backend (BGE)
    # as embedding is normalised cosine == dot(emb, emb.T)
    sim = emb @ emb.T
    return topk_from_similarity_matrix(sim, k=k)


# ---------------------------
# Results assembly
# ---------------------------
def make_results(df: pd.DataFrame, topk_idx: np.ndarray, texts: List[str], k: int):
    """
    Produce both LONG and WIDE outputs.
    LONG: one row per (original, alternative)
    WIDE: one row per original, with Alt1..AltK columns
    """
    rows = []
    n = len(df)
    for i in range(n):
        nbrs = topk_idx[i][:k]
        for rank, j in enumerate(nbrs, start=1):
            rows.append({
                "Original_Row": i,
                "Original_ID": df.iloc[i]["ID"],
                "Original_DESCRIPTION": df.iloc[i]["DESCRIPTION"],
                "Original_Text_Embedded": texts[i],
                "Alt_Rank": rank,
                "Alt_Row": int(j),
                "Alt_ID": df.iloc[j]["ID"],
                "Alt_DESCRIPTION": df.iloc[j]["DESCRIPTION"],
                "Alt_Text_Embedded": texts[j],
            })
    long_df = pd.DataFrame(rows)

    # Wide version (more spreadsheet-friendly)
    wide_rows = []
    for i in range(n):
        recs = [r for r in rows if r["Original_Row"] == i]
        recs.sort(key=lambda r: r["Alt_Rank"])
        row = {
            "Original_Row": i,
            "Original_ID": df.iloc[i]["ID"],
            "Original_DESCRIPTION": df.iloc[i]["DESCRIPTION"],
            "Original_Text_Embedded": texts[i],
        }
        for r in recs[:k]:
            rk = r["Alt_Rank"]
            row[f"Alt{rk}_Row"] = r["Alt_Row"]
            row[f"Alt{rk}_ID"] = r["Alt_ID"]
            row[f"Alt{rk}_DESCRIPTION"] = r["Alt_DESCRIPTION"]
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)

    return long_df, wide_df


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Find similar parts by DESCRIPTION.")
    parser.add_argument("--input", type=str, default="../data/Parts.csv", help="Path to Parts.csv (semicolon-delimited).")
    parser.add_argument("--outdir", type=str, default="./out", help="Output directory.")
    parser.add_argument("--k", type=int, default=5, help="Number of alternatives per part.")
    parser.add_argument(
        "--embed",
        choices=["bge", "tfidf"],
        default="tfidf",
        help="Embedding backend: 'bge' (BAAI/bge-large-en-v1.5) or 'tfidf' (baseline)."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & analyze
    df = load_parts_csv(input_path)
    descriptive_analysis(df)
    df, dropped = filter_rows_with_signal(df)
    if len(dropped):
        print(f"Skipped {len(dropped)} row(s) with no DESCRIPTION and no _CONTEXT_.")
        dropped_path = out_dir / "skipped_no_text.csv"
        dropped.to_csv(dropped_path, index=False)
    # 2) Build texts to embed
    texts = build_texts(df)

    # 3) Embeddings
    print(f"Embedding backend: {'BGE (BAAI/bge-large-en-v1.5)' if args.embed=='bge' else 'TF-IDF'}")
    if args.embed == "bge":
        emb, backend = embed_bge(texts, model_name="BAAI/bge-large-en-v1.5")
    else:
        emb, backend = embed_tfidf(texts)

    # 4) Similarity search (Top-K per part)
    print("Computing similarities & retrieving top-K alternatives")
    topk_idx = similarity_and_topk(emb, backend=backend, k=args.k)

    # 5) Build outputs
    long_df, wide_df = make_results(df, topk_idx, texts, k=args.k)

    # 6) Save
    #long_path = out_dir / f"parts_top{args.k}_similarity_{args.embed}_LONG.csv"
    wide_path = out_dir / f"parts_top{args.k}_similarity_{args.embed}_WIDE.csv"
    #long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("Done")
    #print(f"Saved LONG results → {long_path}")
    print(f"Saved WIDE results → {wide_path}")
    print("\nPreview (first 5 WIDE rows):")
    with pd.option_context("display.max_colwidth", 120, "display.width", 150):
        print(wide_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
