import ollama
from kivy.clock import Clock

def generate_embedding(text, embeddings_model, collection, collection_size):
    """
    Generate an embedding for the given text and add it to the memory collection.
    """
    try:
        response = ollama.embeddings(model=embeddings_model, prompt=text)
        embedding = response["embedding"]
        if not embedding:
            raise ValueError("Generated embedding is empty.")
        collection.add(
            ids=[str(collection_size)],
            embeddings=[embedding],
            documents=[text]
        )
        return embedding
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")

def add_to_memory(text, embeddings_model, collection, collection_size):
    """
    Add the given text to the memory by generating its embedding and storing it in the collection.
    """
    try:
        embedding = generate_embedding(text, embeddings_model, collection, collection_size)
        # Increment the collection size after adding a new memory
        collection_size += 1
        return embedding
    except Exception as e:
        raise Exception(f"Error adding to memory: {e}")

def retrieve_relevant_memory(prompt_embedding, collection):
    """
    Retrieve relevant memories based on the provided prompt embedding.
    """
    try:
        results = collection.query(query_embeddings=[prompt_embedding])
        return [doc for doc in results['documents'][0]]
    except Exception as e:
        raise Exception(f"Error retrieving relevant memory: {e}")

