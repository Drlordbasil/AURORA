import numpy as np
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

class FrontalLobe:
    def __init__(self):
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential([
            Input(shape=(1,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def process(self, prompt):
        print(f"Frontal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform([prompt])
            
            # Fix for PCA issue
            n_components = min(X.shape[1], 2)  # Ensure n_components doesn't exceed the number of features
            pca = PCA(n_components=n_components)
            
            if X.shape[1] > 1:
                reduced_data = pca.fit_transform(X.toarray())
                normalized_data = normalize(reduced_data)
                mean_distance = np.mean(pairwise_distances(normalized_data, [[0] * n_components]))
            else:
                reduced_data = X.toarray()
                normalized_data = normalize(reduced_data)
                mean_distance = np.mean(pairwise_distances(normalized_data, [[0]]))

            decision = "Based on the analysis, the decision is to proceed with caution." if mean_distance < 0.5 else "Proceeding with confidence."
            X_input = np.array([len(prompt.split())])
            prediction = self.model.predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Frontal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            analysis = {
                "PCA Reduced Data": reduced_data.tolist(),
                "Normalized Data": normalized_data.tolist(),
                "Mean Distance from Origin": float(mean_distance),
                "Decision": decision,
                "Prediction": prediction.tolist()
            }
            return f"Frontal Lobe Analysis: {json.dumps(analysis, indent=2)}"
        except Exception as e:
            return f"Error in frontal lobe processing: {str(e)}"