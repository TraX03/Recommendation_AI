import os
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from app.services.recommendation_service import get_local_embeddings


def load_or_embed(
    name: str,
    fetch_func: Callable[[], pd.DataFrame],
    cache_dir: str = "app/cache",
    max_age_days: int = 7,
    batch_size: int = 100,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_embed.parquet")

    def embed_batches(texts: List[str]) -> List[Optional[np.ndarray]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                embeddings.extend(get_local_embeddings(batch))
            except Exception as e:
                print(f"[Embedding] Failed on batch {i // batch_size + 1}: {e}")
                embeddings.extend([None] * len(batch))
        return embeddings

    if os.path.exists(path):
        df = pd.read_parquet(path)
        if datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)) < timedelta(
            days=max_age_days
        ):
            if df["embedding"].notna().all():
                return df
            print(f"[Embedding] Retrying missing embeddings for {name}...")
            mask = df["embedding"].isna()
            df.loc[mask, "embedding"] = embed_batches(
                df.loc[mask, "combined_text"].fillna("").tolist()
            )
            df.to_parquet(path)
            return df

    df = fetch_func().copy()
    print(f"[Embedding] Generating embeddings for {name} in batches...")
    df["embedding"] = embed_batches(df["combined_text"].fillna("").tolist())
    df.to_parquet(path)
    return df
