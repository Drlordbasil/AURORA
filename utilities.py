import logging
import chromadb

def setup_logging():
    logging.basicConfig(filename='lobes_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

def setup_embedding_collection():
    print("Setting up embedding collection.")
    client = chromadb.Client()
    collection = client.create_collection(name="convo_memory")
    print("Embedding collection setup completed.")
    return collection, 0
