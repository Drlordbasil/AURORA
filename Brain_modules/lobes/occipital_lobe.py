# occipital_lobe.py

import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class OccipitalLobe:
    def __init__(self):
        self.visual_keywords = ['see', 'saw', 'look', 'view', 'observe']
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)

        initial_data = ["I see a bird", "Look at the sky", "Observe the stars"]
        initial_labels = [0, 1, 0]
        self.pipeline.fit(initial_data, initial_labels)

        self._load_model()

    def _load_model(self):
        try:
            with open('occipital_lobe_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
        except FileNotFoundError:
            pass

    def _save_model(self):
        with open('occipital_lobe_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

    def process(self, prompt):
        try:
            features = self.pipeline.named_steps['countvectorizer'].transform([prompt])
            prediction = self.pipeline.named_steps['multinomialnb'].predict(features)
            visual_analysis = self._analyze_visual_content(prompt)
            self._train_model(prompt, visual_analysis)
            return visual_analysis
        except Exception as e:
            return f"Error processing occipital lobe: {e}"

    def _analyze_visual_content(self, prompt):
        words = prompt.lower().split()
        visual_words = [word for word in words if word in self.visual_keywords]
        if visual_words:
            return f"Visual elements detected: {', '.join(visual_words)}"
        return "No explicit visual elements detected"

    def _train_model(self, prompt, analysis):
        labels = [1 if "detected" in analysis else 0]
        feature_vector = self.pipeline.named_steps['countvectorizer'].transform([prompt])
        self.pipeline.named_steps['multinomialnb'].partial_fit(feature_vector, labels, classes=np.array([0, 1]))
        self._save_model()
