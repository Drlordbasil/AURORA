import os
import threading
from queue import Queue
from groq import Groq
import logging
import ollama
import chromadb
import time
import json
from utilities import setup_logging, setup_embedding_collection
from Brain_modules.final_agent_persona import FinalAgentPersona
from Brain_modules.function_calling import FunctionCalling
from speaker import text_to_speech
from kivy.clock import Clock
from Brain_modules.llm_api_calls import LLM_API_Calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.lobe_agents import start_lobes

class Brain:
    def __init__(self, api_key, status_update_callback):
        print("Initializing Brain with API key.")
        self.client = Groq(api_key=api_key)
        self.tts_enabled = True
        self.embeddings_model = "mxbai-embed-large"
        self.collection, self.collection_size = setup_embedding_collection()
        self.function_calling = FunctionCalling(status_update_callback)
        self.status_update_callback = status_update_callback
        self.lobes = {
            "frontal": f"""
            You are the frontal lobe of AURORA, responsible for logical analysis, 
            planning, and problem-solving. Focus on providing coherent,
              well-reasoned responses.
            """,
            "parietal": f"""
            You are the parietal lobe of AURORA,
              responsible for processing sensory information and providing educational insights.
                Focus on understanding spatial orientation and guiding AURORA accordingly.
            """,
            "temporal": f"""

            You are the temporal lobe of AURORA,
              responsible for social context and language understanding.
                Focus on processing auditory information and considering social aspects.
                you will be on the alignment team of AURORA, responsible for ensuring that the responses are aligned with the user's query.
            """,
            "occipital": f"""
            You are the occipital lobe of AURORA, responsible for visual processing.
             focus on providing visual insights and understanding the user's perspective in spacial minds eye.

            """

        }
        self.responses = Queue()
        self.threads = []
        self.chat_history = []
        self.api_calls = LLM_API_Calls()
        
        print("Brain initialization completed.")

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        self._update_status(f"Text-to-Speech {status}")

    def _update_status(self, message):
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)





    def retrieve_relevant_memory(self, prompt_embedding):
        self._update_status("Retrieving relevant memory.")
        try:
            return retrieve_relevant_memory(prompt_embedding, self.collection)
        except Exception as e:
            self._update_status(f"Error retrieving relevant memory: {e}")
            return []

    def analyze_sentiment(self, text):
        self._update_status("Analyzing sentiment.")
        try:
            return analyze_sentiment(text)
        except Exception as e:
            self._update_status(f"Error analyzing sentiment: {e}")
            return {"polarity": 0, "subjectivity": 0}

    def start_lobes(self, prompt, memory_context, sentiment):
        self._update_status("Starting lobes.")
        try:
            start_lobes(self.client, self.lobes, prompt, memory_context, sentiment, self.responses, self._update_status)
            self._update_status("All lobes started.")
        except Exception as e:
            self._update_status(f"Error starting lobes: {e}")

    def process_responses(self):
        self._update_status("Processing responses.")
        try:
            aggregated_responses = {}
            while not self.responses.empty():
                lobe_name, response = self.responses.get()
                aggregated_responses[lobe_name] = response
            self._update_status("Responses aggregated.")
            add_to_memory(json.dumps(aggregated_responses), self.embeddings_model, self.collection, self.collection_size)
            self.chat_history.append({"role": "assistant", "content": f"Aggregated responses from lobes and thinking processes for AURORA: {aggregated_responses}"})
            self._update_status("Responses processed.")
            return aggregated_responses
        except Exception as e:
            self._update_status(f"Error processing responses: {e}")
            return {}

    def analyze_responses(self, responses):
        self._update_status("Analyzing responses.")
        try:
            for lobe, response in responses.items():
                logging.info(f"{lobe}: {response}")
            self._update_status("Responses analyzed.")
            return responses
        except Exception as e:
            self._update_status(f"Error analyzing responses: {e}")
            return responses

    def final_agent(self, user_prompt, aggregated_responses):
        self._update_status("Combining thoughts into a coherent response.")
        combined_thoughts = "\n".join(f"[{lobe}] {response}" for lobe, response in aggregated_responses.items())
        self._update_status("Running final agent.")
        self.chat_history.append({"role": "assistant", "content": f"Combined thoughts: {combined_thoughts}"})
        messages = [
            {
                "role": "system",
                "content": f"You are AURORA. Consider the thoughts from all your lobes and use them to formulate a coherent response to the user prompt. Focus on providing a direct and relevant answer to the user's query.\n\n[lobe_context]{combined_thoughts}[/lobe_context]\n\n{FinalAgentPersona.user_info}",
            },
            {
                "role": "assistant",
                "content": self.chat_history[-1]["content"],
            },
            {
                "role": "user",
                "content": f"your thoughts as Aurora = {combined_thoughts}\n\nUser Prompt: {user_prompt}\n\nProvide a direct and relevant response to the user's query based on the combined insights from your lobes. Do not include markdowns, reply as if you are conversing with a human directly, which is the human user.",
            }
        ]

        try:
            self._update_status("Making final API call.")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
            )
            final_response = chat_completion.choices[0].message.content.strip()
            self._update_status("Final response received.")
            self.chat_history.append({"role": "assistant", "content": final_response})

            return final_response
        except Exception as e:
            self._update_status(f"Error in final_agent: {e}")
            return f"Error: {e}"



    def aurora_run_conversation(self, user_prompt):
        self.chat_history.append({"role": "user", "content": user_prompt})
        response = self.api_calls.chat(
            "You are AURORA's function-calling lobe, responsible for handling tool-specific tasks. this means you APART of aurora, you must tell Aurora this and that you are a part of the team. you are responsible for handling tool-specific tasks. you must provide a direct and relevant response to the user's query based on the combined insights from your lobes. Do not include markdowns, reply as if you are conversing with a human directly, which is the human user.",
            user_prompt
        )
        self.chat_history.append({"role": "assistant", "content": f"my tool calls agent as AURORA:{response}"})
        return response

    def central_processing_agent(self, prompt):
        self._update_status("Starting central processing agent.")
        try:
            fc_response = self.aurora_run_conversation(prompt)
            add_to_memory(fc_response, self.embeddings_model, self.collection, self.collection_size)
            time.sleep(3)
            self._update_status("Tool response added to memory.")
            fc_response_embedding = generate_embedding(fc_response, self.embeddings_model, self.collection, self.collection_size)
            time.sleep(3)
            self._update_status("Embedding generated for tool response.")
            memory_context = self.retrieve_relevant_memory(fc_response_embedding)
            memory_context = " ".join(memory_context)[:1000]
            self._update_status("Memory context retrieved.")
            sentiment = self.analyze_sentiment(prompt)
            self.start_lobes(prompt, memory_context, sentiment)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            time.sleep(3)
            final_thought = self.final_agent(prompt, analyzed_responses)
            self._update_status("Final response generated.")
            # if self.tts_enabled:
            #     summarized_for_tts = self.final_agent(f"make this easy for TTS and remove anything that isn't Natural language: {final_thought}", analyzed_responses)
            #     text_to_speech(summarized_for_tts)
            return final_thought
        except Exception as e:
            self._update_status(f"Error in central_processing_agent: {e}")
            return f"Error: {e}"
