# lobes_processing.py

import json
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from textblob import TextBlob
from nltk import word_tokenize, pos_tag
import pyautogui
import os
from Brain_modules.image_vision import ImageVision
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

def create_lobe_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

class LobesProcessing:
    def __init__(self, image_vision):
        self.image_vision = image_vision
        self.lobes = self._initialize_lobes()

    def _initialize_lobes(self):
        lobe_names = [
            "frontal", "parietal", "temporal", "occipital", "limbic",
            "cerebellar", "brocas_area", "wernickes_area", "insular", "association_areas"
        ]
        return {name: create_lobe_model(input_shape=(1,)) for name in lobe_names}

    def process_lobe(self, lobe_name, prompt):
        lobe_method = getattr(self, f"_{lobe_name}", None)
        if lobe_method:
            return lobe_method(prompt)
        else:
            return f"Error: {lobe_name} processing method not found."

    def _frontal(self, prompt):
        print(f"Frontal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform([prompt])
            pca = PCA(n_components=min(X.shape[1], 2))
            reduced_data = pca.fit_transform(X.toarray())
            normalized_data = normalize(reduced_data)
            mean_distance = np.mean(pairwise_distances(normalized_data, [[0] * reduced_data.shape[1]]))
            decision = "Based on the analysis, the decision is to proceed with caution." if mean_distance < 0.5 else "Proceeding with confidence."
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["frontal"].predict(X_input.reshape(1, -1))
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

    def _parietal(self, prompt):
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
            prediction = self.lobes["parietal"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Parietal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Parietal Lobe Analysis: {spatial_analysis}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in parietal lobe processing: {str(e)}"

    def _temporal(self, prompt):
        print(f"Temporal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            blob = TextBlob(prompt)
            sentiment = blob.sentiment
            pos_tags = pos_tag(word_tokenize(prompt))
            keywords = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]
            X_input = np.array([len(keywords)])
            prediction = self.lobes["temporal"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Temporal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Temporal Lobe Analysis: Sentiment - {sentiment}, Keywords - {keywords}, POS Tags - {pos_tags}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in temporal lobe processing: {str(e)}"

    def _occipital(self, prompt):
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
            prediction = self.lobes["occipital"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Occipital lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Occipital Lobe Analysis: {combined_description}, Prediction: {prediction}"
        except Exception as e:
            return f"Error analyzing screenshot: {str(e)}"

    def _limbic(self, prompt):
        print(f"Limbic lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            sentiment = TextBlob(prompt).sentiment
            emotional_response = f"The emotional tone detected is {'positive' if sentiment.polarity > 0 else 'negative' if sentiment.polarity < 0 else 'neutral'}."
            X_input = np.array([sentiment.polarity])
            prediction = self.lobes["limbic"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Limbic lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Limbic Lobe Analysis: {emotional_response}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in limbic lobe processing: {str(e)}"

    def _cerebellar(self, prompt):
        print(f"Cerebellar lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            sequence_steps = prompt.split(',')
            if not sequence_steps:
                return "No sequence steps found."
            sequence_analysis = f"Steps to be followed: {', '.join(sequence_steps)}"
            X_input = np.array([len(sequence_steps)])
            prediction = self.lobes["cerebellar"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Cerebellar lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Cerebellar Lobe Analysis: {sequence_analysis}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in cerebellar lobe processing: {str(e)}"

    def _brocas_area(self, prompt):
        print(f"Broca's Area processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            response = f"Broca's Area Response: {prompt}"
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["brocas_area"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Broca's Area thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Broca's Area Response: {response}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in Broca's Area processing: {str(e)}"

    def _wernickes_area(self, prompt):
        print(f"Wernicke's Area processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            comprehension = f"Wernicke's Area comprehends the following: {prompt}"
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["wernickes_area"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Wernicke's Area thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Wernicke's Area Response: {comprehension}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in Wernicke's Area processing: {str(e)}"

    def _insular(self, prompt):
        print(f"Insular Cortex processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            performance_check = "System performance is optimal."
            emotions = ["neutral", "curious", "thoughtful"]
            selected_emotion = np.random.choice(emotions)
            X_input = np.array([1])
            prediction = self.lobes["insular"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Insular Cortex ({selected_emotion}): thinking at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            internal_state = "calm"
            sensory_feedback = "All sensory systems are functioning within normal parameters."
            decision = "Proceed with current operational parameters."
            result = (f"Insular Cortex Analysis: {performance_check}, Emotion: {selected_emotion}, "
                      f"Prediction: {prediction}, Internal State: {internal_state}, "
                      f"Sensory Feedback: {sensory_feedback}, Decision: {decision}")
            return result
        except Exception as e:
            return f"Error in Insular Cortex processing: {str(e)}"

    def _association_areas(self, prompt):
        print(f"Association Areas processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            integrated_response = f"Association Areas integrated the information: {prompt}"
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["association_areas"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"Association Areas thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Association Areas Response: {integrated_response}, Prediction: {prediction}"
        except Exception as e:
            return f"Error in Association Areas processing: {str(e)}"