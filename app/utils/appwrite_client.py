import os

from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.databases import Databases
from dotenv import load_dotenv

load_dotenv()

# Initialize Appwrite client
client = Client()
client.set_endpoint("https://cloud.appwrite.io/v1").set_project(
    os.environ.get("APPWRITE_PROJECT_ID")
).set_key(os.environ.get("APPWRITE_API_KEY"))

database = Databases(client)


def fetch_documents(
    collection_id: str, database_id: str = None, limit: int = 100
) -> list[dict]:
    """Fetch all documents from a given Appwrite collection, paginated."""
    if database_id is None:
        database_id = os.environ["APPWRITE_DATABASE_ID"]

    documents, offset = [], 0
    while True:
        result = database.list_documents(
            database_id=database_id,
            collection_id=collection_id,
            queries=[Query.limit(limit), Query.offset(offset)],
        )
        docs = result["documents"]
        if not docs:
            break
        documents.extend(docs)
        offset += limit
    return documents


def get_user_document(user_id: str) -> dict:
    """Fetch a single user document by ID."""
    try:
        return database.get_document(
            database_id=os.environ["APPWRITE_DATABASE_ID"],
            collection_id=os.environ["APPWRITE_USERS_COLLECTION_ID"],
            document_id=user_id,
        )
    except Exception as e:
        raise e
