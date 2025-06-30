from typing import List, Optional

from pydantic import BaseModel


class PostList(BaseModel):
    post_ids: List[str]
    titles: List[str]
    images: List[str]
    author_ids: Optional[List[str]] = None


class RecommendationResponse(BaseModel):
    recipe: PostList
    tip: PostList
    discussion: PostList
    community: PostList
