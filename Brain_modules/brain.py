import json
import time
from queue import Queue
from Brain_modules.final_agent_persona import FinalAgentPersona
from Brain_modules.llm_api_calls import LLM_API_Calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.image_vision import ImageVision
from Brain_modules.lobes.lobes_processing import LobesProcessing
from utilities import setup_embedding_collection

class Brain:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        print(f"Initializing Brain at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.tts_enabled = True
        self.collection, self.collection_size = setup_embedding_collection()
        self.image_vision = ImageVision()
        self.api_calls = LLM_API_Calls()
        self.client = self.api_calls.client
        self.lobes_processing = LobesProcessing(self.image_vision)
        self.embeddings_model = "mxbai-embed-large"
        self.responses = Queue()
        self.chat_history = []
        self.last_response = ""
        print(f"Brain initialization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        return "enabled" if self.tts_enabled else "disabled"

    def start_lobes(self, combined_input, memory_context, sentiment):
        for lobe_name, lobe in self.lobes_processing.lobes.items():
            response = lobe.process(combined_input)
            self.responses.put((lobe_name, response))

    def process_responses(self):
        aggregated_responses = self._aggregate_responses()
        self._add_responses_to_memory(aggregated_responses)
        return aggregated_responses

    def _aggregate_responses(self):
        return {lobe_name: response for lobe_name, response in self.responses.queue}

    def _add_responses_to_memory(self, aggregated_responses):
        add_to_memory(json.dumps(aggregated_responses), self.embeddings_model, self.collection, self.collection_size)
        self.chat_history.append({"role": "assistant", "content": f"AURORA responses: {aggregated_responses}"})

    def _combine_thoughts(self, aggregated_responses):
        return "\n".join(f"[{lobe}] {str(response)[:100]}" for lobe, response in aggregated_responses.items())

    def _get_relevant_chat_history(self):
        return "\n".join(f"{msg['role'].capitalize()}: {msg['content'][:50]}" 
                         for msg in self.chat_history[-3:] 
                         if msg['role'] in ["assistant", "user"])

    def _prepare_messages(self, user_prompt, combined_thoughts, chat_history_str):
        return [
            {"role": "system", "content": f"NEVER EXPLAIN YOUR THOUGHTS UNLESS BREIFLY UPON REQUEST ONLY. You are {FinalAgentPersona.name}. {FinalAgentPersona.description[:100]}"},
            {"role": "user", "content": f"Prompt: {user_prompt}\nContext: {combined_thoughts}NEVER EXPLAIN YOUR THOUGHTS UNLESS BREIFLY UPON REQUEST ONLY\nHistory: {chat_history_str}"}
        ]

    def aurora_run_conversation(self, user_prompt):
        self.chat_history.append({"role": "user", "content": user_prompt})
        system_message = f"you are {FinalAgentPersona.name} {FinalAgentPersona.description}"

        try:
            response = self.api_calls.chat(system_message, user_prompt)
            if isinstance(response, dict) and 'error' in response:
                return f"Error: {response['error']} at {response['datetime']}"
            
            tool_response = response.strip() if isinstance(response, str) else json.dumps(response)
            self.chat_history.append({"role": "assistant", "content": f"AURORA: {tool_response[:100]}"})
            return tool_response
        except Exception as e:
            return f"Error in aurora_run_conversation: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"

    def central_processing_agent(self, prompt):
        try:
            combined_input = f"{self.last_response[:50]} {prompt}"
            print(f"Processing: {combined_input[:100]}")
            
            fc_response = self.aurora_run_conversation(combined_input)
            if fc_response.startswith("Error"):
                return fc_response

            self._process_fc_response(fc_response)
            memory_context = self._get_memory_context(fc_response)
            sentiment = analyze_sentiment(combined_input)
            self.start_lobes(combined_input, memory_context, sentiment)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            final_thought = self.final_agent(combined_input, analyzed_responses)
            print(f"Final response generated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return final_thought
        except Exception as e:
            return f"Error in central_processing_agent: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"

    def _process_fc_response(self, fc_response):
        add_to_memory(fc_response[:1500], self.embeddings_model, self.collection, self.collection_size)
        print(f"Processed fc_response at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _get_memory_context(self, fc_response):
        fc_response_embedding = add_to_memory(fc_response[:500], self.embeddings_model, self.collection, self.collection_size)
        memory_context = retrieve_relevant_memory(fc_response_embedding, self.collection)
        return " ".join(memory_context)[:1500]

    def analyze_responses(self, responses):
        return {lobe: {"response": str(response)[:100], "sentiment": analyze_sentiment(str(response))} 
                for lobe, response in responses.items()}

    def final_agent(self, user_prompt, aggregated_responses):
        combined_thoughts = self._combine_thoughts(aggregated_responses)
        chat_history_str = self._get_relevant_chat_history()
        messages = self._prepare_messages(user_prompt, combined_thoughts, chat_history_str)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
                
            )
            final_response = chat_completion.choices[0].message.content.strip()
            self.chat_history.append({"role": "assistant", "content": final_response[:100]})
            self.last_response = final_response
            return final_response
        except Exception as e:
            return f"Error in final_agent: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"