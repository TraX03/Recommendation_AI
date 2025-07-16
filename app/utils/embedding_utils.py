import os
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

CACHE_DIR = "app/cache"
MAX_AGE_DAYS = 7

os.makedirs(CACHE_DIR, exist_ok=True)

try:
    # NOTE: This block will be skipped on Render free tier due to memory/storage limits.
    # If sentence-transformers is unavailable (not in requirements), embedding will be disabled.
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    embedding_model = None
    print("[NOTE] sentence-transformers not found. Embedding is disabled.")


def load_or_embed(
    name: str,
    fetch_func: Callable[[], pd.DataFrame],
    text_column: str = "combined_text",
) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{name}_embed.parquet")
    now = datetime.now()

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = fetch_func().copy()
        df["embedding"] = None
        df["embedding_ts"] = None

    if text_column not in df.columns:
        raise ValueError(f"DataFrame must include '{text_column}' column")

    df["embedding_ts"] = pd.to_datetime(df["embedding_ts"], errors="coerce")
    age_cutoff = now - timedelta(days=MAX_AGE_DAYS)

    needs_embedding = df["embedding"].isna() | df["embedding_ts"].lt(age_cutoff)

    if embedding_model is None:
        print("[Embedding] Skipping embedding step because model is not available.")
        return df

    if needs_embedding.any():
        print(
            f"[Embedding] Re-embedding {needs_embedding.sum()} stale/missing rows for {name}..."
        )

        to_embed_mask = needs_embedding & df[text_column].notna()
        to_embed = df.loc[to_embed_mask, text_column].fillna("").tolist()
        embeddings = _embed_batches(to_embed)

        df.loc[to_embed_mask, "embedding"] = pd.Series(
            embeddings, index=df[to_embed_mask].index, dtype="object"
        )
        df.loc[to_embed_mask, "embedding_ts"] = now

        df.to_parquet(path, index=False)

    return df


def _get_local_embeddings(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    if embedding_model is None:
        raise RuntimeError("Embedding model is not available.")
    return list(
        embedding_model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    )


def _embed_batches(
    texts: List[str], batch_size: int = 100
) -> List[Optional[np.ndarray]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_embeddings = _get_local_embeddings(batch)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"[Embedding] Failed on batch {i // batch_size + 1}: {e}")
            embeddings.extend([None] * len(batch))
    return embeddings
