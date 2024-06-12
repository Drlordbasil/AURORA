# utilities.py

import logging
import chromadb

def setup_logging():
    """
    Set up logging configuration for the application.
    """
    try:
        logging.basicConfig(
            filename='lobes_log.txt',
            level=logging.INFO,
            format='%(asctime)s %(message)s'
        )
        print("Logging setup completed.")
    except Exception as e:
        print(f"Error setting up logging: {e}")

def setup_embedding_collection():
    """
    Set up the embedding collection using ChromaDB.

    Returns:
        tuple: A tuple containing the created collection and its initial size (0).
    """
    print("Setting up embedding collection.")
    try:
        client = chromadb.Client()
        collection = client.create_collection(name="convo_memory")
        print("Embedding collection setup completed.")
        return collection, 0
    except Exception as e:
        print(f"Error setting up embedding collection: {e}")
        return None, 0
