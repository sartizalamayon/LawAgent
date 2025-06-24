import os
import sys
from dotenv import load_dotenv
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

def test_qdrant_connection():
    """
    Tests the connection to the Qdrant cloud instance using credentials from .env
    """
    load_dotenv()
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    assert qdrant_url, "QDRANT_URL not found in .env file"
    assert qdrant_api_key, "QDRANT_API_KEY not found in .env file"

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        # The simplest way to check the connection and key is to list collections
        collections = client.get_collections()
        print("Successfully connected to Qdrant and retrieved collections.")
        print(f"Collections found: {[c.name for c in collections.collections]}")
        assert True
    except UnexpectedResponse as e:
        if "invalid api key" in str(e).lower():
            pytest.fail("Qdrant API key is invalid. Please generate a new one and update your .env file.")
        else:
            pytest.fail(f"Failed to connect to Qdrant due to an unexpected response: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred while connecting to Qdrant: {e}") 