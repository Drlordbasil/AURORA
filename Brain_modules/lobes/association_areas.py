# association_areas.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from collections import defaultdict
from scipy.spatial.distance import cosine


from typing import List, Dict

class ToolAssociation:
    def __init__(self, name: str, keywords: List[str], embedding: np.ndarray):
        self.name = name
        self.keywords = set(keywords)
        self.embedding = embedding
        self.usage_count = 0
        self.success_rate = 0.5

class AssociationAreas:
    def __init__(self, input_dim: int = 5000, hidden_layers: List[int] = [256, 128], learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_layers = [input_dim] + hidden_layers + [1]
        self.learning_rate = learning_rate
        self.model_filename = "association_areas_model.pkl"
        self.weights = []
        self.biases = []
        self.tools = self._initialize_tools()
        self.vocabulary = set()
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_embeddings = {}
        self.tfidf_vectorizer = TfidfVectorizer()
        self.suggestion_threshold = 0.2
        self._load_or_initialize_model()

    def _initialize_tools(self) -> Dict[str, ToolAssociation]:
        tools = {
            'web_search': ToolAssociation('web_search', ['search', 'find', 'lookup', 'research', 'google', 'internet', 'web', 'online', 'information', 'update', 'news'], np.random.randn(100)),
            'image_analysis': ToolAssociation('image_analysis', ['image', 'picture', 'photo', 'analyze', 'visual', 'see', 'look'], np.random.randn(100)),
            'pdf_extraction': ToolAssociation('pdf_extraction', ['pdf', 'document', 'extract', 'read', 'text', 'file'], np.random.randn(100)),
            'sentiment_analysis': ToolAssociation('sentiment_analysis', ['sentiment', 'feeling', 'emotion', 'opinion', 'mood', 'attitude', 'happy', 'sad', 'angry', 'excited', 'disappointed', 'satisfied', 'frustrated', 'emotional', 'feel'], np.random.randn(100)),
            'local_command': ToolAssociation('local_command', ['run', 'execute', 'command', 'system', 'local', 'computer'], np.random.randn(100)),
            'math_calculation': ToolAssociation('math_calculation', ['calculate', 'compute', 'math', 'arithmetic', 'number', 'equation', 'sum', 'difference', 'multiply', 'divide'], np.random.randn(100)),
            'time_date': ToolAssociation('time_date', ['time', 'date', 'schedule', 'calendar', 'when', 'now'], np.random.randn(100)),
            'weather': ToolAssociation('weather', ['weather', 'forecast', 'temperature', 'climate', 'rain', 'sun'], np.random.randn(100)),
            'translation': ToolAssociation('translation', ['translate', 'language', 'foreign', 'meaning', 'interpret'], np.random.randn(100)),
            'summarization': ToolAssociation('summarization', ['summarize', 'brief', 'short', 'concise', 'overview', 'gist'], np.random.randn(100)),
            'recipe_search': ToolAssociation('recipe_search', ['recipe', 'cook', 'bake', 'ingredients', 'dish', 'meal'], np.random.randn(100)),
            'code_explanation': ToolAssociation('code_explanation', ['code', 'programming', 'function', 'algorithm', 'debug', 'syntax'], np.random.randn(100)),
            'general_knowledge': ToolAssociation('general_knowledge', ['what', 'why', 'how', 'explain', 'define', 'meaning', 'philosophy'], np.random.randn(100))
        }
        return tools

    def _load_or_initialize_model(self):
        try:
            self._load_model()
        except (FileNotFoundError, KeyError, pickle.UnpicklingError):
            print("Error loading model. Initializing a new model.")
            self._initialize_new_model()

    def _initialize_new_model(self):
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
        self._update_vocabulary(set(word for tool in self.tools.values() for word in tool.keywords))
        self._initialize_word_embeddings()

    def _update_vocabulary(self, new_words):
        for word in new_words:
            if word not in self.vocabulary:
                index = len(self.vocabulary)
                if index < self.input_dim:
                    self.vocabulary.add(word)
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word

    def _initialize_word_embeddings(self):
        for word in self.vocabulary:
            self.word_embeddings[word] = np.random.randn(100)  # 100-dimensional embeddings

    def _initialize_weights(self) -> List[np.ndarray]:
        return [np.random.randn(i, j) * np.sqrt(2. / (i + j)) for i, j in zip(self.hidden_layers[:-1], self.hidden_layers[1:])]

    def _initialize_biases(self) -> List[np.ndarray]:
        return [np.zeros((1, nodes)) for nodes in self.hidden_layers[1:]]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def _forward_propagation(self, X: np.ndarray) -> List[np.ndarray]:
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = np.maximum(0.01 * z, z)  # Leaky ReLU
            activations.append(a)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self._sigmoid(z)
        activations.append(a)
        return activations

    def process(self, prompt: str) -> str:
        try:
            words = self._tokenize(prompt)
            self._update_vocabulary(words)
            X_input = self._preprocess_input(words)
            prediction = self._forward_propagation(X_input)[-1][0][0]
            tool_suggestions = self._suggest_tools(prompt)
            
            if not tool_suggestions:
                return f"Association Areas Response: Prediction: {prediction:.4f}, Action: Just reply to the user properly without tools"
            else:
                return f"Association Areas Response: Prediction: {prediction:.4f}, Suggested Tools: {tool_suggestions}"
        except Exception as e:
            return f"Error in Association Areas processing: {str(e)}"

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _preprocess_input(self, words: List[str]) -> np.ndarray:
        input_vector = np.zeros((1, self.input_dim))
        for word in words:
            if word in self.word_to_index:
                input_vector[0, self.word_to_index[word]] = 1
        return input_vector

    def _suggest_tools(self, prompt: str) -> List[str]:
        words = self._tokenize(prompt)
        prompt_embedding = self._get_text_embedding(prompt)
        
        tool_scores = []
        for tool in self.tools.values():
            keyword_match = len(set(words) & tool.keywords) / len(tool.keywords)
            embedding_similarity = 1 - cosine(prompt_embedding, tool.embedding)
            combined_score = 0.7 * keyword_match + 0.3 * embedding_similarity
            tool_scores.append((tool.name, combined_score))
        
        sorted_tools = sorted(tool_scores, key=lambda x: x[1], reverse=True)
        suggested_tools = [tool for tool, score in sorted_tools if score > self.suggestion_threshold][:3]
        
        return suggested_tools

    def _get_text_embedding(self, text: str) -> np.ndarray:
        words = self._tokenize(text)
        word_vectors = [self.word_embeddings.get(word, np.zeros(100)) for word in words]
        return np.mean(word_vectors, axis=0)

    def update_from_interaction(self, prompt: str, used_tool: str, success_rating: float):
        words = self._tokenize(prompt)
        self._update_vocabulary(words)
        X_input = self._preprocess_input(words)
        
        target = success_rating
        activations = self._forward_propagation(X_input)
        
        # Backpropagation
        delta = activations[-1] - target
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[i].T, delta)
            dB = np.sum(delta, axis=0, keepdims=True)
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * (activations[i] > 0)  # Leaky ReLU derivative
        
        # Update tool statistics
        if used_tool in self.tools:
            tool = self.tools[used_tool]
            tool.usage_count += 1
            tool.success_rate = (tool.success_rate * (tool.usage_count - 1) + success_rating) / tool.usage_count
            
            # Update tool embedding
            prompt_embedding = self._get_text_embedding(prompt)
            tool.embedding = 0.9 * tool.embedding + 0.1 * prompt_embedding
        
        # Dynamic threshold adjustment
        self.suggestion_threshold = max(0.1, min(0.3, np.mean([tool.success_rate for tool in self.tools.values()])))
        
        self._save_model()

    def _save_model(self):
        model_data = {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'weights': self.weights,
            'biases': self.biases,
            'tools': self.tools,
            'vocabulary': self.vocabulary,
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'word_embeddings': self.word_embeddings,
            'learning_rate': self.learning_rate,
            'suggestion_threshold': self.suggestion_threshold
        }
        with open(self.model_filename, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self):
        with open(self.model_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.input_dim = model_data['input_dim']
        self.hidden_layers = model_data['hidden_layers']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.tools = model_data['tools']
        self.vocabulary = model_data['vocabulary']
        self.word_to_index = model_data['word_to_index']
        self.index_to_word = model_data['index_to_word']
        self.word_embeddings = model_data['word_embeddings']
        self.learning_rate = model_data['learning_rate']
        self.suggestion_threshold = model_data['suggestion_threshold']

if __name__ == "__main__":
    association_areas = AssociationAreas()

    test_cases = [
        "I want to search for information online about climate change",
        "Can you analyze this image of a sunset and describe what you see?",
        "I need to extract text from a PDF document about renewable energy",
        "I'm feeling really happy today!",
        "Run this Python script on my local system",
        "Calculate the sum of 5, 10, and 15",
        "What's the weather forecast for tomorrow in New York?",
        "Translate 'Hello, how are you?' to French",
        "Summarize the main points of this long article about artificial intelligence",
        "What time is it in Tokyo right now?",
        "I'm not sure what I need, just chat with me",
        "Can you help me with my homework?",
        "Tell me a joke",
        "What's the meaning of life?",
        "How do I bake a chocolate cake?",
        "Explain this Python function to me",
        "What's the capital of France?",
        "Find a recipe for vegetarian lasagna"
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {case}")
        print(association_areas.process(case))

    print("\nTesting update_from_interaction:")
    association_areas.update_from_interaction("Search for recent news about AI", "web_search", 0.9)
    print("Model updated. Processing the same input again:")
    print(association_areas.process("Search for recent news about AI"))

    print("\nTesting with a new input after update:")
    print(association_areas.process("Find the latest research papers on machine learning"))

    print("\nAll tests completed.")