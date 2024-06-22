# wernickes_area.py

import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class WernickesArea:
    def __init__(self):
        self.language_keywords = ['understand', 'comprehend', 'meaning', 'language']
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)

        initial_data = ["I understand the meaning", "Comprehend the language", "The meaning is clear"]
        initial_labels = [0, 1, 0]
        self.pipeline.fit(initial_data, initial_labels)

        self._load_model()

    def _load_model(self):
        try:
            with open('wernickes_area_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
        except FileNotFoundError:
            pass

    def _save_model(self):
        with open('wernickes_area_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

    def process(self, prompt):
        try:
            features = self.pipeline.named_steps['countvectorizer'].transform([prompt])
            prediction = self.pipeline.named_steps['multinomialnb'].predict(features)
            language_analysis = self._analyze_language_content(prompt)
            self._train_model(prompt, language_analysis)
            return language_analysis
        except Exception as e:
            return f"Error processing wernickes area: {e}"

    def _analyze_language_content(self, prompt):
        words = prompt.lower().split()
        language_words = [word for word in words if word in self.language_keywords]
        if language_words:
            return f"Language elements detected: {', '.join(language_words)}"
        return "No explicit language elements detected"

    def _train_model(self, prompt, analysis):
        labels = [1 if "detected" in analysis else 0]
        feature_vector = self.pipeline.named_steps['countvectorizer'].transform([prompt])
        self.pipeline.named_steps['multinomialnb'].partial_fit(feature_vector, labels, classes=np.array([0, 1]))
        self._save_model()
