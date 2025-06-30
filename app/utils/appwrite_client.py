from appwrite.client import Client
from appwrite.id import ID
from appwrite.query import Query
from appwrite.services.databases import Databases
from dotenv import load_dotenv

from app.constants import APPWRITE_API_KEY, APPWRITE_DATABASE_ID, APPWRITE_PROJECT_ID

load_dotenv()

client = Client()
client.set_endpoint("https://cloud.appwrite.io/v1").set_project(
    APPWRITE_PROJECT_ID
).set_key(APPWRITE_API_KEY)

database = Databases(client)


def list_documents(collection_id: str, queries: list = None) -> list[dict]:
    if queries is None:
        queries = []

    try:
        result = database.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=collection_id,
            queries=queries,
        )
        return result
    except Exception as err:
        print(f"list_documents failed: {err}")
        return []


def fetch_documents(
    collection_id: str,
    limit: int = 300,
    custom_queries: list = None,
) -> list[dict]:
    if custom_queries is None:
        custom_queries = []

    documents, offset = [], 0
    while True:
        queries = [*custom_queries, Query.limit(limit), Query.offset(offset)]

        result = list_documents(
            collection_id=collection_id,
            queries=queries,
        )
        docs = result["documents"]
        if not docs:
            break
        documents.extend(docs)
        offset += limit
    return documents


def get_document_by_id(
    collection_id: str,
    document_id: str,
) -> dict:
    try:
        return database.get_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=collection_id,
            document_id=document_id,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get document {document_id}: {e}")


def create_document(
    collection_id: str,
    data: dict,
    document_id: str = None,
) -> dict:
    if document_id is None:
        document_id = ID.unique()

    try:
        return database.create_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=collection_id,
            document_id=document_id,
            data=data,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create document in {collection_id}: {e}")


def update_document(
    collection_id: str,
    document_id: str,
    data: dict,
) -> dict:
    try:
        return database.update_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=collection_id,
            document_id=document_id,
            data=data,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to update document {document_id}: {e}")


def create_or_update_document(
    collection_id: str,
    document_id: str,
    data: dict,
) -> dict:
    try:
        return update_document(
            collection_id=collection_id,
            document_id=document_id,
            data=data,
        )
    except Exception:
        return create_document(
            collection_id=collection_id,
            document_id=document_id,
            data=data,
        )
