import os
import json
import time
import logging
from queue import Queue
import numpy as np
import pyautogui
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from textblob import TextBlob
from nltk import word_tokenize, pos_tag
from utilities import setup_logging, setup_embedding_collection
from Brain_modules.final_agent_persona import FinalAgentPersona
from kivy.clock import Clock
from Brain_modules.llm_api_calls import LLM_API_Calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.image_vision import ImageVision

class Brain:
    def __init__(self, api_key, status_update_callback):
        print(f"""Initializing Brain with API key at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        self.tts_enabled = True
        self.embeddings_model = "mxbai-embed-large"
        self.collection, self.collection_size = setup_embedding_collection()
        
        self.status_update_callback = status_update_callback
        self.image_vision = ImageVision()
        self.api_calls = LLM_API_Calls()
        self.client = self.api_calls.client  # Get the client from LLM_API_Calls
        self.lobes = {
            "frontal": self.create_neural_network("frontal_lobe.h5"),
            "parietal": self.create_neural_network("parietal_lobe.h5"),
            "temporal": self.create_neural_network("temporal_lobe.h5"),
            "occipital": self.create_neural_network("occipital_lobe.h5"),
            "limbic": self.create_neural_network("limbic_lobe.h5"),
            "cerebellar": self.create_neural_network("cerebellar_lobe.h5"),
            "brocas_area": self.create_neural_network("brocas_area.h5"),
            "wernickes_area": self.create_neural_network("wernickes_area.h5"),
            "insular": self.create_neural_network("insular_lobe.h5"),
            "association_areas": self.create_neural_network("association_areas.h5")
        }
        self.responses = Queue()
        self.threads = []
        self.chat_history = []

        print(f"""Brain initialization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}""")

    def create_neural_network(self, model_name):
        try:
            if os.path.exists(model_name):
                model = load_model(model_name)
                print(f"""Loaded model {model_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            else:
                model = Sequential([
                    Input(shape=(1,)),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                print(f"""Created new model {model_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            print(f"""Error creating/loading model {model_name}: {str(e)} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return None

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        self._update_status(f"""Text-to-Speech {status} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")

    def _update_status(self, message):
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)

    def retrieve_relevant_memory(self, prompt_embedding):
        self._update_status(f"""Retrieving relevant memory at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            return retrieve_relevant_memory(prompt_embedding, self.collection)
        except Exception as e:
            self._update_status(f"""Error retrieving relevant memory: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return []

    def analyze_sentiment(self, text):
        self._update_status(f"""Analyzing sentiment at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            return analyze_sentiment(text)
        except Exception as e:
            self._update_status(f"""Error analyzing sentiment: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return {"polarity": 0, "subjectivity": 0}

    def frontal_lobe(self, prompt):
        self._update_status(f"""Frontal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform([prompt])
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(X.toarray())
            decision = "Based on the analysis, the decision is to proceed with caution." if reduced_data.mean() < 0.5 else "Proceeding with confidence."
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["frontal"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Frontal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Frontal Lobe Analysis: {decision}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in frontal lobe processing: {str(e)}"""

    def parietal_lobe(self, prompt):
        self._update_status(f"""Parietal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            numbers = [int(s) for s in prompt.split() if s.isdigit()]
            if not numbers:
                return "No numerical data found in the input."
            mean_value = np.mean(numbers)
            median_value = np.median(numbers)
            std_dev = np.std(numbers)
            spatial_analysis = f"""The average is {mean_value}, median is {median_value}, and standard deviation is {std_dev}."""
            X_input = np.array([len(numbers)])
            prediction = self.lobes["parietal"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Parietal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Parietal Lobe Analysis: {spatial_analysis}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in parietal lobe processing: {str(e)}"""

    def temporal_lobe(self, prompt):
        self._update_status(f"""Temporal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            blob = TextBlob(prompt)
            sentiment = blob.sentiment
            pos_tags = pos_tag(word_tokenize(prompt))
            keywords = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]
            X_input = np.array([len(keywords)])
            prediction = self.lobes["temporal"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Temporal lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Temporal Lobe Analysis: Sentiment - {sentiment}, Keywords - {keywords}, POS Tags - {pos_tags}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in temporal lobe processing: {str(e)}"""

    def occipital_lobe(self, prompt):
        self._update_status(f"""Occipital lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
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
                descriptions.append(self.image_vision.analyze_image(img_path))
            
            combined_description = " ".join(descriptions)
            X_input = np.array([len(images)])
            prediction = self.lobes["occipital"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Occipital lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Occipital Lobe Analysis: {combined_description}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error analyzing screenshot: {str(e)}"""

    def limbic_lobe(self, prompt):
        self._update_status(f"""Limbic lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            sentiment = TextBlob(prompt).sentiment
            emotional_response = f"""The emotional tone detected is {'positive' if sentiment.polarity > 0 else 'negative' if sentiment.polarity < 0 else 'neutral'}."""
            X_input = np.array([sentiment.polarity])
            prediction = self.lobes["limbic"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Limbic lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Limbic Lobe Analysis: {emotional_response}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in limbic lobe processing: {str(e)}"""

    def cerebellar_lobe(self, prompt):
        self._update_status(f"""Cerebellar lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            sequence_steps = prompt.split(',')
            if not sequence_steps:
                return "No sequence steps found."
            sequence_analysis = f"""Steps to be followed: {', '.join(sequence_steps)}"""
            X_input = np.array([len(sequence_steps)])
            prediction = self.lobes["cerebellar"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Cerebellar lobe thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Cerebellar Lobe Analysis: {sequence_analysis}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in cerebellar lobe processing: {str(e)}"""

    def brocas_area(self, prompt):
        self._update_status(f"""Broca's Area processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            response = f"""Broca's Area Response: {prompt}"""  # Simple echo for language production
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["brocas_area"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Broca's Area thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Broca's Area Response: {response}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in Broca's Area processing: {str(e)}"""

    def wernickes_area(self, prompt):
        self._update_status(f"""Wernicke's Area processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            comprehension = f"""Wernicke's Area comprehends the following: {prompt}"""  # Simple echo for comprehension
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["wernickes_area"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Wernicke's Area thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Wernicke's Area Response: {comprehension}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in Wernicke's Area processing: {str(e)}"""

    def insular_cortex(self, prompt):
        self._update_status(f"""Insular Cortex processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            performance_check = "System performance is optimal."
            X_input = np.array([1])
            prediction = self.lobes["insular"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Insular Cortex thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Insular Cortex Analysis: {performance_check}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in Insular Cortex processing: {str(e)}"""

    def association_areas(self, prompt):
        self._update_status(f"""Association Areas processing at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            integrated_response = f"""Association Areas integrated the information: {prompt}"""
            X_input = np.array([len(prompt.split())])
            prediction = self.lobes["association_areas"].predict(X_input.reshape(1, -1))
            for _ in range(5):
                time.sleep(1)
                print(f"""Association Areas thinking: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Association Areas Response: {integrated_response}, Prediction: {prediction}"""
        except Exception as e:
            return f"""Error in Association Areas processing: {str(e)}"""

    def start_lobes(self, prompt, memory_context, sentiment):
        self._update_status(f"""Starting lobes at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            for lobe_name, lobe_function in self.lobes.items():
                response = lobe_function(prompt)
                self.responses.put((lobe_name, response))
            self._update_status(f"""All lobes started at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        except Exception as e:
            self._update_status(f"""Error starting lobes: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")

    def process_responses(self):
        self._update_status(f"""Processing responses at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            aggregated_responses = {}
            while not self.responses.empty():
                lobe_name, response = self.responses.get()
                aggregated_responses[lobe_name] = response
            self._update_status(f"""Responses aggregated at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            add_to_memory(json.dumps(aggregated_responses), self.embeddings_model, self.collection, self.collection_size)
            self.chat_history.append({"role": "assistant", "content": f"""Aggregated responses from lobes and thinking processes for AURORA: {aggregated_responses}"""})
            self._update_status(f"""Responses processed at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return aggregated_responses
        except Exception as e:
            self._update_status(f"""Error processing responses: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return {}

    def analyze_responses(self, responses):
        self._update_status(f"""Analyzing responses at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            for lobe, response in responses.items():
                logging.info(f"""{lobe}: {response}""")
            self._update_status(f"""Responses analyzed at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return responses
        except Exception as e:
            self._update_status(f"""Error analyzing responses: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return responses

    def final_agent(self, user_prompt, aggregated_responses):
        self._update_status(f"""Combining thoughts into a coherent response at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        combined_thoughts = "\n".join(f"""[{lobe}] {response}""" for lobe, response in aggregated_responses.items())
        
        relevant_chat_history = []
        for message in self.chat_history[-10:]:
            if message["role"] == "assistant" or message["role"] == "user":
                relevant_chat_history.append(f"""{message['role'].capitalize()}: {message['content']}""")
        
        chat_history_str = "\n".join(relevant_chat_history)
        
        self._update_status(f"""Running final agent at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        self.chat_history.append({"role": "assistant", "content": f"""Combined thoughts: {combined_thoughts}"""})
        messages = [
            {
                "role": "system",
                "content": f"""{FinalAgentPersona.name}. {FinalAgentPersona.description} You have access to various tools and functions which you can use to gather information, perform tasks, and analyze data. Your tools include running local commands, performing web research, analyzing images, extracting text from PDFs, and analyzing sentiment. Consider the thoughts from all your lobes and use them to formulate a coherent response to the user prompt. Focus on providing a direct and relevant answer to the user's query.\n\n[lobe_context]{combined_thoughts}[/lobe_context]\n\n{FinalAgentPersona.user_info}\n\nChat History:\n{chat_history_str}""",
            },
            {
                "role": "user",
                "content": f"""User Prompt Start\n{user_prompt}\nUser Prompt End""",
            }
        ]

        try:
            self._update_status(f"""Making final API call at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
            )
            final_response = chat_completion.choices[0].message.content.strip()
            self._update_status(f"""Final response received at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            self.chat_history.append({"role": "assistant", "content": final_response})

            return final_response
        except Exception as e:
            self._update_status(f"""Error in final_agent: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Error: {e}"""

    def aurora_run_conversation(self, user_prompt):
        self.chat_history.append({"role": "user", "content": user_prompt})
        response = self.api_calls.chat(
            f"""You are AURORA's function-calling lobe, responsible for handling tool-specific tasks. You are a part of Aurora and must tell Aurora this. You are responsible for handling tool-specific tasks. You must provide a direct and relevant response to the user's query based on the combined insights from your lobes. Do not include markdowns, reply as if you are conversing with a human directly, which is the human user at {time.strftime('%Y-%m-%d %H:%M:%S')}""",
            user_prompt
        )
        self.chat_history.append({"role": "assistant", "content": f"""my tool calls agent as AURORA: {response}"""})
        return response

    def central_processing_agent(self, prompt):
        self._update_status(f"""Starting central processing agent at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        try:
            fc_response = self.aurora_run_conversation(prompt)
            add_to_memory(fc_response, self.embeddings_model, self.collection, self.collection_size)
            time.sleep(3)
            self._update_status(f"""Tool response added to memory at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            fc_response_embedding = generate_embedding(fc_response, self.embeddings_model, self.collection, self.collection_size)
            time.sleep(3)
            self._update_status(f"""Embedding generated for tool response at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            memory_context = self.retrieve_relevant_memory(fc_response_embedding)
            memory_context = " ".join(memory_context)[:1000]
            self._update_status(f"""Memory context retrieved at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            sentiment = self.analyze_sentiment(prompt)
            self.start_lobes(prompt, memory_context, sentiment)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            time.sleep(3)
            final_thought = self.final_agent(prompt, analyzed_responses)
            self._update_status(f"""Final response generated at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return final_thought
        except Exception as e:
            self._update_status(f"""Error in central_processing_agent: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}""")
            return f"""Error: {e}"""
