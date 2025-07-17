import os
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app import dependencies

CACHE_DIR = "app/cache"
MAX_AGE_DAYS = 7

os.makedirs(CACHE_DIR, exist_ok=True)


def get_embedding_model():
    if dependencies.embedding_model is None:
        dependencies.embedding_model = SentenceTransformer(
            "paraphrase-MiniLM-L3-v2", device="cpu"
        )
    return dependencies.embedding_model


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

    model = get_embedding_model()
    if model is None:
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


def _get_local_embeddings(texts: List[str], batch_size: int = 8) -> List[np.ndarray]:
    model = get_embedding_model()
    if model is None:
        raise RuntimeError("Embedding model is not available.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float16)

    return list(embeddings)


def _embed_batches(texts: List[str], batch_size: int = 8) -> List[Optional[np.ndarray]]:
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
