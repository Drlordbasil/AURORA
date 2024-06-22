# temporal_lobe.py

import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class TemporalLobe:
    def __init__(self):
        self.auditory_keywords = ['hear', 'listen', 'sound', 'music']
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)

        initial_data = ["I hear music", "Listen to the sound", "The sound is loud"]
        initial_labels = [0, 1, 0]
        self.pipeline.fit(initial_data, initial_labels)

        self._load_model()

    def _load_model(self):
        try:
            with open('temporal_lobe_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
        except FileNotFoundError:
            pass

    def _save_model(self):
        with open('temporal_lobe_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

    def process(self, prompt):
        try:
            features = self.pipeline.named_steps['countvectorizer'].transform([prompt])
            prediction = self.pipeline.named_steps['multinomialnb'].predict(features)
            auditory_analysis = self._analyze_auditory_content(prompt)
            self._train_model(prompt, auditory_analysis)
            return f"Auditory analysis complete. {auditory_analysis} Full analysis: {auditory_analysis}"
        except Exception as e:
            return f"Error processing temporal lobe: {e}"

    def _analyze_auditory_content(self, prompt):
        words = prompt.lower().split()
        auditory_words = [word for word in words if word in self.auditory_keywords]
        if auditory_words:
            return f"Auditory elements detected: {', '.join(auditory_words)}"
        return "No explicit auditory elements detected"

    def _train_model(self, prompt, analysis):
        labels = [1 if "detected" in analysis else 0]
        feature_vector = self.pipeline.named_steps['countvectorizer'].transform([prompt])
        self.pipeline.named_steps['multinomialnb'].partial_fit(feature_vector, labels, classes=np.array([0, 1]))
        self._save_model()
