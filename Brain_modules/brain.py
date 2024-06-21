# brain.py

import os
import json
import time
import logging
from queue import Queue
import numpy as np
from utilities import setup_embedding_collection
from Brain_modules.final_agent_persona import FinalAgentPersona
from kivy.clock import Clock
from Brain_modules.llm_api_calls import LLM_API_Calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.image_vision import ImageVision
from Brain_modules.lobes_processing import LobesProcessing

class Brain:
    def __init__(self, api_key, status_update_callback):
        self._initialize(api_key, status_update_callback)

    def _initialize(self, api_key, status_update_callback):
        print(f"Initializing Brain with API key at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.tts_enabled = True
        self.embeddings_model = "mxbai-embed-large"
        self.collection, self.collection_size = setup_embedding_collection()
        self.status_update_callback = status_update_callback
        self.image_vision = ImageVision()
        self.api_calls = LLM_API_Calls(self.status_update_callback)
        self.client = self.api_calls.client
        self.lobes_processing = LobesProcessing(self.image_vision)
        self.responses = Queue()
        self.threads = []
        self.chat_history = []
        print(f"Brain initialization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _update_status(self, message):
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        self._update_status(f"Text-to-Speech {status} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def retrieve_relevant_memory(self, prompt_embedding):
        self._update_status(f"Retrieving relevant memory at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            return retrieve_relevant_memory(prompt_embedding, self.collection)
        except Exception as e:
            self._update_status(f"Error retrieving relevant memory: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return []

    def analyze_sentiment(self, text):
        self._update_status(f"Analyzing sentiment at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            return analyze_sentiment(text)
        except Exception as e:
            self._update_status(f"Error analyzing sentiment: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return {"polarity": 0, "subjectivity": 0}

    def start_lobes(self, prompt, memory_context, sentiment):
        self._update_status(f"Starting lobes at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            for lobe_name in self.lobes_processing.lobes.keys():
                response = self.lobes_processing.process_lobe(lobe_name, prompt)
                self.responses.put((lobe_name, response))
            self._update_status(f"All lobes started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            self._update_status(f"Error starting lobes: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def process_responses(self):
        self._update_status(f"Processing responses at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            aggregated_responses = self._aggregate_responses()
            self._add_responses_to_memory(aggregated_responses)
            return aggregated_responses
        except Exception as e:
            self._update_status(f"Error processing responses: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return {}

    def _aggregate_responses(self):
        aggregated_responses = {}
        while not self.responses.empty():
            lobe_name, response = self.responses.get()
            aggregated_responses[lobe_name] = response
        self._update_status(f"Responses aggregated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return aggregated_responses

    def _add_responses_to_memory(self, aggregated_responses):
        add_to_memory(json.dumps(aggregated_responses), self.embeddings_model, self.collection, self.collection_size)
        self.chat_history.append({"role": "assistant", "content": f"Aggregated responses from lobes and thinking processes for AURORA: {aggregated_responses}"})
        self._update_status(f"Responses processed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def analyze_responses(self, responses):
        self._update_status(f"Analyzing responses at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            for lobe, response in responses.items():
                logging.info(f"{lobe}: {response}")
            self._update_status(f"Responses analyzed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return responses
        except Exception as e:
            self._update_status(f"Error analyzing responses: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return responses

    def final_agent(self, user_prompt, aggregated_responses):
        self._update_status(f"Combining thoughts into a coherent response at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        combined_thoughts = self._combine_thoughts(aggregated_responses)
        chat_history_str = self._get_relevant_chat_history()
        
        self._update_status(f"Running final agent at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.chat_history.append({"role": "assistant", "content": f"Combined thoughts: {combined_thoughts}"})
        
        messages = self._prepare_messages(user_prompt, combined_thoughts, chat_history_str)

        try:
            return self._make_final_api_call(messages)
        except Exception as e:
            self._update_status(f"Error in final_agent: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Error: {e}"

    def _combine_thoughts(self, aggregated_responses):
        return "\n".join(f"[{lobe}] {response}" for lobe, response in aggregated_responses.items())

    def _get_relevant_chat_history(self):
        relevant_chat_history = []
        for message in self.chat_history[-10:]:
            if message["role"] in ["assistant", "user"]:
                relevant_chat_history.append(f"{message['role'].capitalize()}: {message['content']}")
        return "\n".join(relevant_chat_history)

    def _prepare_messages(self, user_prompt, combined_thoughts, chat_history_str):
        return [
            {
                "role": "system",
                "content": f"{FinalAgentPersona.name}. {FinalAgentPersona.description} You have access to various tools and functions which you can use to gather information, perform tasks, and analyze data. Your tools include running local commands, performing web research, analyzing images, extracting text from PDFs, and analyzing sentiment. Consider the thoughts from all your lobes and use them to formulate a coherent response to the user prompt. Focus on providing a direct and relevant answer to the user's query.\n\n[lobe_context]{combined_thoughts}[/lobe_context]\n\n{FinalAgentPersona.user_info}\n\nChat History:\n{chat_history_str}",
            },
            {
                "role": "user",
                "content": f"User Prompt Start\n{user_prompt}\nUser Prompt End",
            }
        ]

    def _make_final_api_call(self, messages):
        self._update_status(f"Making final API call at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
        )
        final_response = chat_completion.choices[0].message.content.strip()
        self._update_status(f"Final response received at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.chat_history.append({"role": "assistant", "content": final_response})
        return final_response

    def aurora_run_conversation(self, user_prompt):
        self._update_status(f"AURORA conversation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.chat_history.append({"role": "user", "content": user_prompt})

        system_message = self._get_aurora_system_message()

        try:
            self._update_status(f"Making API call at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            response = self.api_calls.chat(system_message, user_prompt)
            
            tool_response = response.strip()
            self._update_status(f"Tool response received at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            self.chat_history.append({"role": "assistant", "content": f"my tool calls agent as AURORA: {tool_response}"})
            
            return tool_response
        except Exception as e:
            self._update_status(f"Error in aurora_run_conversation: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Error: {e}"

    def _get_aurora_system_message(self):
        return (
            f"You are AURORA's function-calling lobe, responsible for handling tool-specific tasks. "
            "You are a part of Aurora and must tell Aurora this. You are responsible for handling tool-specific tasks. "
            "You must provide a direct and relevant response to the user's query based on the combined insights from your lobes. "
            "Do not include markdowns, reply as if you are conversing with a human directly, which is the human user at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def central_processing_agent(self, prompt):
        self._update_status(f"Starting central processing agent at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            fc_response = self.aurora_run_conversation(prompt)
            self._process_fc_response(fc_response)
            memory_context = self._get_memory_context(fc_response)
            sentiment = self.analyze_sentiment(prompt)
            self.start_lobes(prompt, memory_context, sentiment)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            time.sleep(3)
            final_thought = self.final_agent(prompt, analyzed_responses)
            self._update_status(f"Final response generated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return final_thought
        except Exception as e:
            self._update_status(f"Error in central_processing_agent: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Error: {e}"

    def _process_fc_response(self, fc_response):
        add_to_memory(fc_response, self.embeddings_model, self.collection, self.collection_size)
        time.sleep(3)
        self._update_status(f"Tool response added to memory at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _get_memory_context(self, fc_response):
        fc_response_embedding = generate_embedding(fc_response, self.embeddings_model, self.collection, self.collection_size)
        time.sleep(3)
        self._update_status(f"Embedding generated for tool response at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        memory_context = self.retrieve_relevant_memory(fc_response_embedding)
        memory_context = " ".join(memory_context)[:1000]
        self._update_status(f"Memory context retrieved at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return memory_context