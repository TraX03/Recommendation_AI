from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

recipes_df: pd.DataFrame = None
tfidf_matrix: Any = None
tfidf_vectorizer: TfidfVectorizer = None
