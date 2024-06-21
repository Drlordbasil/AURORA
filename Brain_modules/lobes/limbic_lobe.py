import numpy as np
import time
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

class LimbicLobe:
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
        print(f"Limbic lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            sentiment = TextBlob(prompt).sentiment
            emotional_response = f"The emotional tone detected is {'positive' if sentiment.polarity > 0 else 'negative' if sentiment.polarity < 0 else 'neutral'}."
            X_input = np.array([sentiment.polarity])
            prediction = self.model.predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Limbic lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Limbic Lobe Analysis: {emotional_response}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in limbic lobe processing: {str(e)}"