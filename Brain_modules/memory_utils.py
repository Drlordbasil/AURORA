import ollama
from kivy.clock import Clock

def generate_embedding(text, embeddings_model, collection, collection_size):
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
    try:
        embedding = generate_embedding(text, embeddings_model, collection, collection_size)
        collection_size += 1
        return embedding
    except Exception as e:
        raise Exception(f"Error adding to memory: {e}")

def retrieve_relevant_memory(prompt_embedding, collection):
    try:
        results = collection.query(query_embeddings=[prompt_embedding])
        return [doc for doc in results['documents'][0]]
    except Exception as e:
        raise Exception(f"Error retrieving relevant memory: {e}")
