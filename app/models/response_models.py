from typing import List, Optional

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    recipe_ids: List[str]
    titles: List[str]
    images: List[str]
    author_ids: Optional[List[str]] = None
