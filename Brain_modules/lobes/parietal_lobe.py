import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

class ParietalLobe:
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
        print(f"Parietal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            numbers = [int(s) for s in prompt.split() if s.isdigit()]
            if not numbers:
                return "No numerical data found in the input."
            mean_value = np.mean(numbers)
            median_value = np.median(numbers)
            std_dev = np.std(numbers)
            spatial_analysis = f"The average is {mean_value}, median is {median_value}, and standard deviation is {std_dev}."
            X_input = np.array([len(numbers)])
            prediction = self.model.predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Parietal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Parietal Lobe Analysis: {spatial_analysis}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in parietal lobe processing: {str(e)}"