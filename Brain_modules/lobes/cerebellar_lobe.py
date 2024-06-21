import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class CerebellarLobe:
    def __init__(self):
        self.model = self._create_model()
        self.threshold = 0.5  # Threshold for binary classification

    def _create_model(self):
        model = Sequential([
            Input(shape=(1,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def process(self, prompt):
        print(f"Cerebellar lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            sequence_steps = prompt.split(',')
            if not sequence_steps:
                return "No sequence steps found."
            
            sequence_analysis = f"Steps to be followed: {', '.join(sequence_steps)}"
            X_input = np.array([len(sequence_steps)])
            prediction = self.model.predict(X_input.reshape(1, -1))
            binary_prediction = (prediction > self.threshold).astype(int)

            for _ in range(5):
                time.sleep(1)
                print(f"Cerebellar lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            return f"Cerebellar Lobe Analysis: {sequence_analysis}, Prediction: {binary_prediction}"
        except Exception as e:
            return f"Error in cerebellar lobe processing: {str(e)}"

    def train(self, X_train, y_train, epochs=50, batch_size=8):
        early_stopping = EarlyStopping(monitor='loss', patience=5)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        print(f"Model trained at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Model evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}: Loss = {loss}, Accuracy = {accuracy}")
        return loss, accuracy
