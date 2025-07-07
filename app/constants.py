import os

from dotenv import load_dotenv

load_dotenv()

APPWRITE_PROJECT_ID = os.environ["APPWRITE_PROJECT_ID"]
APPWRITE_API_KEY = os.environ["APPWRITE_API_KEY"]
APPWRITE_DATABASE_ID = os.environ["APPWRITE_DATABASE_ID"]
USERS_COLLECTION_ID = os.environ.get("APPWRITE_USERS_COLLECTION_ID")
RECIPES_COLLECTION_ID = os.environ.get("APPWRITE_RECIPES_COLLECTION_ID")
INTERACTIONS_COLLECTION_ID = os.environ.get("APPWRITE_INTERACTIONS_COLLECTION_ID")
POSTS_COLLECTION_ID = os.environ.get("APPWRITE_POSTS_COLLECTION_ID")
COMMUNITIES_COLLECTION_ID = os.environ.get("APPWRITE_COMMUNITIES_COLLECTION_ID")
RECOMMENDATION_DATA_COLLECTION_ID = os.environ.get(
    "APPWRITE_RECOMMENDATION_DATA_COLLECTION_ID"
)

INTERACTION_WEIGHTS = {
    "like": 1.0,
    "bookmark": 0.5,
    "view": 0.2,
}

BLOCKED_KEYWORDS_BY_DIET = {
    "vegan": [
        "meat",
        "beef",
        "chicken",
        "pork",
        "lamb",
        "goat",
        "seafood",
        "fish",
        "egg",
        "cheese",
        "milk",
    ],
    "vegetarian": [
        "meat",
        "beef",
        "chicken",
        "pork",
        "lamb",
        "goat",
        "seafood",
        "fish",
    ],
    "pescatarian": ["meat", "beef", "chicken", "pork", "lamb", "goat"],
}

CONTENT_TYPE_MAP = {
    "recipe": {
        "id_col": "recipe_id",
        "attr": "recipes_df",
    },
    "tip": {
        "id_col": "post_id",
        "attr": "tips_df",
    },
    "discussion": {
        "id_col": "post_id",
        "attr": "discussions_df",
    },
    "community": {
        "id_col": "community_id",
        "attr": "communities_df",
    },
}
