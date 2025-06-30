from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.recommendation_service import RecommendationEngine

engine: RecommendationEngine = None
recipes_df: pd.DataFrame = None
tfidf_matrix: Any = None
tfidf_vectorizer: TfidfVectorizer = None
