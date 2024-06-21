import numpy as np
import time
import os
import pyautogui
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from Brain_modules.image_vision import ImageVision

class OccipitalLobe:
    def __init__(self):
        self.model = self._create_model()
        self.image_vision = ImageVision()

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
        print(f"Occipital lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        date_str = time.strftime("%Y-%m-%d")
        if not os.path.exists(date_str):
            os.makedirs(date_str)
        
        timestamp_str = time.strftime("%H-%M-%S")
        screenshot_path = os.path.join(date_str, f"screenshot_{timestamp_str}.jpg")
        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)

        try:
            images = [os.path.join(date_str, img) for img in os.listdir(date_str) if img.endswith('.jpg')]
            descriptions = []
            for img_path in images:
                try:
                    description = self.image_vision.analyze_image(img_path)
                    descriptions.append(description)
                except Exception as img_error:
                    print(f"Error analyzing image {img_path}: {str(img_error)}")
            combined_description = " ".join(descriptions)
            X_input = np.array([len(images)])
            prediction = self.model.predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Occipital lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Occipital Lobe Analysis: {combined_description}, Prediction: {prediction}"
        except Exception as e:
            return f"Error analyzing screenshot: {str(e)}"