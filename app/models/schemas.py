from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class PostList(BaseModel):
    post_ids: List[str]
    titles: List[str]
    images: List[str]
    author_ids: Optional[List[str]] = None


class RecipeItem(BaseModel):
    id: str
    title: str
    image: str


class MealItem(BaseModel):
    mealtime: str
    recipes: List[RecipeItem]
    session: str


class MealPlanRequest(BaseModel):
    mealtime: List[str]
    date: date
    target_recipe_id: Optional[str] = None


class MealPlanResponse(BaseModel):
    user_id: str
    date: date
    meals: List[MealItem]


class FeedbackRequest(BaseModel):
    date: str
    mealtime: List[str]
    target_recipe_id: Optional[str] = None
    is_regenerate: Optional[bool] = False


class RecommendationResponse(BaseModel):
    recipe: PostList
    tip: PostList
    discussion: PostList
    community: PostList
