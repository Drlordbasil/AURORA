# parietal_lobe.py

import numpy as np
import time
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class ParietalLobe:
    def __init__(self):
        """
        Initializes the ParietalLobe class with predefined sets of keywords for spatial, sensory, and navigation processing.
        Also initializes the machine learning model for learning and adaptation.
        """
        print("Initializing ParietalLobe...")
        self.spatial_keywords = ['up', 'down', 'left', 'right', 'above', 'below', 'near', 'far']
        self.sensory_keywords = ['touch', 'feel', 'texture', 'temperature', 'pressure', 'pain']
        self.navigation_keywords = ['map', 'route', 'direction', 'location', 'distance', 'navigate']

        # Initialize the CountVectorizer and Naive Bayes model
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)

        # Initial training data to avoid the "not fitted" error
        initial_data = [
            "The box is above the table, near the window",
            "I feel a rough texture and cold temperature",
            "Navigate to the nearest exit using the map",
            "Calculate the distance between points A (2,3) and B (5,7)",
            "The room temperature is 72 degrees",
            "Process this sentence without any spatial or numerical content"
        ]
        initial_labels = [0, 1, 0, 1, 0, 0]  # Arbitrary labels
        print("Fitting initial data to pipeline...")
        self.pipeline.fit(initial_data, initial_labels)

        # Load existing model and vectorizer state if available
        print("Loading model...")
        self._load_model()

        # Error collection structure
        self.error_log = []

    def _load_model(self):
        """
        Loads the model and vectorizer state from a file, if available.
        """
        try:
            with open('parietal_lobe_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Proceeding with the initial model.")

    def _save_model(self):
        """
        Saves the model and vectorizer state to a file.
        """
        with open('parietal_lobe_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)
        print("Model saved successfully.")

    def _preprocess_prompt(self, prompt):
        """
        Preprocesses the input prompt to ensure it is clean and consistent.
        """
        # Implement any additional preprocessing steps here
        return prompt

    def _extract_features(self, prompt):
        """
        Extracts feature vectors from the given prompt using CountVectorizer.
        Ensures consistent feature extraction for both training and prediction.
        """
        return self.pipeline.named_steps['countvectorizer'].transform([prompt])

    def process(self, prompt):
        """
        Processes the given prompt to analyze spatial, sensory, navigation, and numerical content.
        Also trains the model incrementally with the new prompt.

        Args:
            prompt (str): The input sentence to be processed.

        Returns:
            str: A detailed response summarizing the analysis of the prompt.
        """
        print(f"Processing prompt: '{prompt}'")
        prompt = self._preprocess_prompt(prompt)
        print(f"Parietal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            features = self._extract_features(prompt)
            print(f"Features: {features.toarray()}")
            prediction = self.pipeline.named_steps['multinomialnb'].predict(features)
            print(f"Prediction: {prediction}")

            spatial_analysis = self._analyze_spatial_content(prompt)
            print(f"Spatial Analysis: {spatial_analysis}")
            sensory_integration = self._integrate_sensory_information(prompt)
            print(f"Sensory Integration: {sensory_integration}")
            navigation_assessment = self._assess_navigation(prompt)
            print(f"Navigation Assessment: {navigation_assessment}")
            numerical_analysis = self._analyze_numerical_data(prompt)
            print(f"Numerical Analysis: {numerical_analysis}")

            for _ in range(3):
                time.sleep(0.5)
                print(f"Parietal lobe processing: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            analysis = {
                "Spatial Analysis": spatial_analysis,
                "Sensory Integration": sensory_integration,
                "Navigation Assessment": navigation_assessment,
                "Numerical Analysis": numerical_analysis
            }
            print(f"Full Analysis: {analysis}")

            # Train the model incrementally with the new data
            self._train_model(prompt, analysis)

            return f"Parietal Lobe Response: Spatial-sensory integration complete. {self._summarize_analysis(analysis)}"
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            self._handle_error(prompt, e)
            return f"Parietal Lobe Response: Error in processing: {str(e)}. Spatial-sensory systems recalibrating."

    def _train_model(self, prompt, analysis):
        """
        Trains the model incrementally with the new prompt and its analysis.

        Args:
            prompt (str): The input sentence.
            analysis (dict): The analysis result of the prompt.
        """
        print(f"Training model with prompt: '{prompt}'")
        labels = [1 if "detected" in label or "processing" in label or "identified" in label or "found" in label else 0 for label in [
            analysis["Spatial Analysis"], 
            analysis["Sensory Integration"], 
            analysis["Navigation Assessment"], 
            analysis["Numerical Analysis"]
        ]]
        print(f"Labels: {labels}")

        # Update the vectorizer and model with the new data
        feature_vector = self._extract_features(prompt)
        feature_vector = np.vstack([feature_vector.toarray()] * len(labels))
        self.pipeline.named_steps['multinomialnb'].partial_fit(feature_vector, labels, classes=np.array([0, 1]))
        print("Model trained successfully.")

        # Save the updated model
        self._save_model()

    def _analyze_spatial_content(self, prompt):
        """
        Analyzes the prompt for spatial content based on predefined spatial keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of spatial elements detected in the prompt.
        """
        words = prompt.lower().split()
        spatial_words = [word for word in words if word in self.spatial_keywords]
        if spatial_words:
            return f"Spatial elements detected: {', '.join(spatial_words)}"
        return "No explicit spatial elements detected"

    def _integrate_sensory_information(self, prompt):
        """
        Analyzes the prompt for sensory information based on predefined sensory keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of sensory information detected in the prompt.
        """
        sensory_words = [word for word in prompt.lower().split() if word in self.sensory_keywords]
        if sensory_words:
            return f"Sensory information processing: {', '.join(sensory_words)}"
        return "No specific sensory information to process"

    def _assess_navigation(self, prompt):
        """
        Analyzes the prompt for navigation-related content based on predefined navigation keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of navigation-related concepts detected in the prompt.
        """
        nav_words = [word for word in prompt.lower().split() if word in self.navigation_keywords]
        if nav_words:
            return f"Navigation-related concepts identified: {', '.join(nav_words)}"
        return "No navigation-specific elements found"

    def _analyze_numerical_data(self, prompt):
        """
        Analyzes the prompt for numerical data and performs basic statistical analysis if numerical data is found.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of numerical data detected in the prompt and basic statistical analysis.
        """
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            numbers = [int(num) for num in numbers]
            return f"Numerical data found: mean={np.mean(numbers):.2f}, median={np.median(numbers):.2f}, count={len(numbers)}"
        return "No numerical data found"

    def _summarize_analysis(self, analysis):
        """
        Summarizes the analysis results into a comprehensive response.

        Args:
            analysis (dict): The dictionary containing the analysis results.

        Returns:
            str: A summary of the analysis results.
        """
        summary = []
        if "elements detected" in analysis["Spatial Analysis"]:
            summary.append("Spatial processing activated")
        if "information processing" in analysis["Sensory Integration"]:
            summary.append("Sensory integration in progress")
        if "concepts identified" in analysis["Navigation Assessment"]:
            summary.append("Navigation systems engaged")
        if "Numerical data found" in analysis["Numerical Analysis"]:
            summary.append("Quantitative analysis performed")
        
        if not summary:
            return "No significant spatial-sensory patterns identified. Maintaining baseline awareness."
        
        return " ".join(summary) + f" Full analysis: {analysis}"

    def _handle_error(self, prompt, error):
        """
        Handles errors encountered during processing and adapts the system to prevent future errors.

        Args:
            prompt (str): The input sentence that caused the error.
            error (Exception): The error encountered during processing.
        """
        print(f"Handling error: {error}")
        self.error_log.append((prompt, str(error)))

        # Simple adaptive logic: log the error and retrain with a modified label
        if 'setting an array element with a sequence' in str(error):
            # Example logic: Add the prompt with a generic label and retrain
            self._retrain_with_error_prompt(prompt)

    def _retrain_with_error_prompt(self, prompt):
        """
        Retrains the model with the prompt that caused an error, using a generic label.

        Args:
            prompt (str): The input sentence that caused the error.
        """
        print(f"Retraining model with error prompt: '{prompt}'")
        error_data = [prompt]
        error_labels = [0]  # Generic label for error handling
        self.pipeline.fit(error_data, error_labels)
        print("Model retrained with error prompt.")

        # Save the updated model
        self._save_model()

if __name__ == "__main__":
    parietal_lobe = ParietalLobe()
    test_prompts = [
        "The box is above the table, near the window",
        "I feel a rough texture and cold temperature",
        "Navigate to the nearest exit using the map",
        "Calculate the distance between points A (2,3) and B (5,7)",
        "The room temperature is 72 degrees",
        "Process this sentence without any spatial or numerical content",
        "Error: setting an array element with a sequence",
        "sup man its me Anthony"
    ]
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        result = parietal_lobe.process(prompt)
        print(result)
