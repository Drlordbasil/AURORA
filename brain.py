import threading
from queue import Queue
from groq import Groq
import logging
import ollama
import chromadb
import time
import json
from utilities import setup_logging, setup_embedding_collection
from final_agent_persona import FinalAgentPersona
from function_calling import FunctionCalling
from speaker import text_to_speech
from kivy.clock import Clock

class Brain:
    """
    The Brain class simulates a human-like brain structure with different lobes responsible for
    distinct cognitive functions. It integrates various AI models and APIs to process user prompts,
    generate embeddings, retrieve relevant memory, and produce coherent responses.
    """
    
    def __init__(self, api_key, status_update_callback):
        """
        Initialize the Brain class with the provided API key.

        Args:
            api_key (str): The API key for accessing the Groq service.
            status_update_callback (callable): Callback function to update status in the GUI.
        """
        print("Initializing Brain with API key.")
        self.client = Groq(api_key=api_key)
        self.tts_enabled = True  # Text-to-Speech flag
        self.embeddings_model = "mxbai-embed-large"
        self.collection, self.collection_size = setup_embedding_collection()
        self.function_calling = FunctionCalling(status_update_callback)
        self.status_update_callback = status_update_callback
        self.lobes = {
            "frontal": "You are the frontal lobe of AURORA, responsible for analyzing user prompts logically and providing coherent, well-reasoned responses. Focus on reasoning, planning, and problem-solving. Guide AURORA based on the user input without directly responding to the user.",
            "parietal": "You are the parietal lobe of AURORA, responsible for providing educational insights based on user prompts. Focus on processing sensory information and understanding spatial orientation to guide AURORA based on the user input.",
            "temporal": "You are the temporal lobe of AURORA, responsible for contextualizing user prompts socially and providing responses considering social aspects. Focus on processing auditory information and understanding language to guide AURORA based on the user input.",
            "occipital": "You are the occipital lobe of AURORA, responsible for describing things visually based on user prompts, providing vivid and clear descriptions. Focus on processing visual information to guide AURORA based on the user input.",
        }
        self.responses = Queue()
        self.threads = []
        self.chat_history = []  # To keep track of the chat history
        print("Brain initialization completed.")

    def toggle_tts(self):
        """
        Toggle the Text-to-Speech functionality on or off.
        """
        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        print(f"Text-to-Speech {status}")
        self._update_status(f"Text-to-Speech {status}")

    def _update_status(self, message):
        """
        Update the GUI with the current status message.

        Args:
            message (str): The status message to be displayed on the GUI.
        """
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)

    def add_to_memory(self, text):
        """
        Add the given text to the memory by generating its embedding and storing it in the collection.

        Args:
            text (str): The text to be added to memory.
        """
        print("Adding to memory.")
        self._update_status("Adding to memory.")
        try:
            response = ollama.embeddings(model=self.embeddings_model, prompt=text)
            embedding = response["embedding"]
            self.collection.add(
                ids=[str(self.collection_size)],
                embeddings=[embedding],
                documents=[text]
            )
            self.collection_size += 1
            print("Memory added.")
            self._update_status("Memory added.")
        except Exception as e:
            print(f"Error adding to memory: {e}")
            self._update_status(f"Error adding to memory: {e}")

    def generate_embedding(self, text):
        """
        Generate an embedding for the given text.

        Args:
            text (str): The text for which the embedding is to be generated.

        Returns:
            list: The generated embedding or None if there was an error.
        """
        print("Generating embedding.")
        self._update_status("Generating embedding.")
        try:
            response = ollama.embeddings(model=self.embeddings_model, prompt=text)
            print("Embedding generated.")
            self._update_status("Embedding generated.")
            self.collection.add(
                ids=[str(self.collection_size)],
                embeddings=[response["embedding"]],
                documents=[text]
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            self._update_status(f"Error generating embedding: {e}")
            return None

    def retrieve_relevant_memory(self, prompt_embedding):
        """
        Retrieve relevant memories based on the provided prompt embedding.

        Args:
            prompt_embedding (list): The embedding of the prompt for querying relevant memories.

        Returns:
            list: A list of relevant memory documents.
        """
        print("Retrieving relevant memory.")
        self._update_status("Retrieving relevant memory.")
        try:
            results = self.collection.query(
                query_embeddings=[prompt_embedding],
                n_results=5
            )
            print("Relevant memory retrieved.")
            self._update_status("Relevant memory retrieved.")
            return [doc for doc in results['documents'][0]]
        except Exception as e:
            print(f"Error retrieving relevant memory: {e}")
            self._update_status(f"Error retrieving relevant memory: {e}")
            return []

    def lobe_agent(self, lobe_name, user_prompt, memory_context):
        """
        Execute a lobe agent to process the user prompt within the context of the specified lobe.

        Args:
            lobe_name (str): The name of the lobe.
            user_prompt (str): The user prompt.
            memory_context (str): The memory context for the lobe to use.
        """
        print(f"Starting lobe agent for {lobe_name}.")
        self._update_status(f"Starting lobe agent for {lobe_name}.")
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are the {lobe_name} lobe of AURORA (Artificial Unified Responsive Optimized Reasoning Agent). Your role is to provide verbose guidance to AURORA based on the user prompt and memory context. Analyze the user prompt from the perspective of the {lobe_name} lobe and offer detailed insights, suggestions, and reasoning to assist AURORA in formulating a coherent response. Remember, you are an integral part of AURORA's thought process, and your input will shape AURORA's final response.",
                },
                {
                    "role": "user",
                    "content": f"[user_prompt]Message from the user: {user_prompt}[/user_prompt]\n\nMemory Context: {memory_context}\n\n### Provide verbose guidance to AURORA as its inner thoughts. Analyze the user prompt from the perspective of the {lobe_name} lobe and offer detailed insights, suggestions, and reasoning. ###",
                },
            ]
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
                temperature=1,
            )
            response = chat_completion.choices[0].message.content
            self.responses.put((lobe_name, response))
            print(f"Lobe agent for {lobe_name} completed.")
            self._update_status(f"Lobe agent for {lobe_name} completed.")
            time.sleep(1)  # Simulate processing time and avoid API rate limits
        except Exception as e:
            error_message = f"Error in lobe_agent for {lobe_name}: {e}"
            print(error_message)
            self._update_status(error_message)
            self.responses.put((lobe_name, f"Error: {e}"))

    def retrieve_context(self, user_prompt, memory_context):
        """
        Retrieve the context for the user prompt based on the memory context provided.

        Args:
            user_prompt (str): The user prompt.
            memory_context (str): The memory context for the user prompt.

        Returns:
            str: The context for the user prompt.
        """
        print("Retrieving context.")
        self._update_status("Retrieving context.")
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are the context retrieval agent for AURORA (Artificial Unified Responsive Optimized Reasoning Agent). Your role is to analyze the user prompt and memory context to provide a coherent context that combines both elements. Consider the user prompt and memory context to create a meaningful context that helps AURORA generate a well-informed response.",
                },
                {
                    "role": "user",
                    "content": f"[user_prompt]Message from the user: {user_prompt}[/user_prompt]\n\nMemory Context: {memory_context}\n\n### Provide a coherent context that combines the user prompt and memory context to guide AURORA in generating a well-informed response. ###",
                },
            ]
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
                temperature=1,
            )
            response = chat_completion.choices[0].message.content
            print("Context retrieved.")
            self._update_status("Context retrieved.")
            return response
        except Exception as e:
            error_message = f"Error in retrieve_context: {e}"
            print(error_message)
            self._update_status(error_message)
            return f"Error: {e}"

    def start_lobes(self, prompt, memory_context):
        """
        Start the processing of user prompt by each lobe.

        Args:
            prompt (str): The user prompt to be processed.
            memory_context (str): The memory context for the lobe to use.
        """
        print("Starting lobes.")
        self._update_status("Starting lobes.")
        try:
            for lobe_name in self.lobes.keys():
                thread = threading.Thread(target=self.lobe_agent, args=(lobe_name, prompt, memory_context))
                thread.start()
                self.threads.append(thread)
                print(f"Lobe {lobe_name} started.")
                self._update_status(f"Lobe {lobe_name} started.")
                time.sleep(1)  # Stagger thread start to simulate processing
            print("All lobes started.")
            self._update_status("All lobes started.")
        except Exception as e:
            print(f"Error starting lobes: {e}")
            self._update_status(f"Error starting lobes: {e}")

    def process_responses(self):
        """
        Process the responses from all lobes after they have finished execution.

        Returns:
            dict: Aggregated responses from all lobes.
        """
        print("Processing responses.")
        self._update_status("Processing responses.")
        try:
            for thread in self.threads:
                thread.join()

            aggregated_responses = {}
            self._update_status("Aggregating responses.")
            while not self.responses.empty():
                lobe_name, response = self.responses.get()
                aggregated_responses[lobe_name] = response
            self._update_status("Responses aggregated.")
            self.add_to_memory("\n".join(aggregated_responses.values()))
            self._update_status("Responses processed.")
            return aggregated_responses
        except Exception as e:
            print(f"Error processing responses: {e}")
            self._update_status(f"Error processing responses: {e}")
            return {}

    def analyze_responses(self, responses):
        """
        Analyze the responses received from all lobes.

        Args:
            responses (dict): The responses from each lobe.

        Returns:
            dict: The analyzed responses.
        """
        print("Analyzing responses.")
        self._update_status("Analyzing responses.")
        try:
            # Log the responses for later analysis
            for lobe, response in responses.items():
                logging.info(f"{lobe}: {response}")
            print("Responses analyzed.")
            self._update_status("Responses analyzed.")
            return responses
        except Exception as e:
            print(f"Error analyzing responses: {e}")
            self._update_status(f"Error analyzing responses: {e}")
            return responses
    def final_agent(self, user_prompt, aggregated_responses):
        """
        Combine the thoughts from all lobes into a coherent final response.

        Args:
            user_prompt (str): The original user prompt.
            aggregated_responses (dict): The aggregated responses from all lobes.

        Returns:
            str: The final coherent response.
        """
        print("Combining thoughts into a coherent response.")
        self._update_status("Combining thoughts into a coherent response.")
        combined_thoughts = "\n".join(f"[{lobe}] {response}" for lobe, response in aggregated_responses.items())
        self._update_status("Running final agent.")
        
        messages = [
            {
                "role": "system",
                "content": f"You are AURORA, an entity that uses its lobes like a human does subconsciously. Consider the thoughts from all your lobes and use them to formulate a coherent response to the user prompt.\n\n[lobe_context]{combined_thoughts}[/lobe_context]\n\n{FinalAgentPersona.user_info} You have access to amazing tools in your chat chains. You can use the tool-use agent to handle tool-specific tasks and return the result. This agent can browse the web for live knowledge, use local commands on a Windows 11 PC, has vision capabilities, and can help you do a variety of tasks. You can also use the tool-use agent to interact with other AI models and APIs to enhance your responses.",
            },
            {
                "role": "user",
                "content": f"[context]{self.retrieve_context}[/context][user_prompt]{user_prompt}[/user_prompt]\n\nBased on the thoughts from your lobes, provide a coherent response to the user prompt. Incorporate the insights and suggestions provided by your lobes to address the user's query effectively. Send only your response to the user and don't send anything else besides YOUR response as AURORA to the user. DO NOT include your lobes, tool calls, and/or other thought processes unless asked directly, otherwise only respond to the user prompt [user_prompt]{user_prompt}[/user_prompt].",
            }
        ]

        try:
            print("Making final API call.")
            self._update_status("Making final API call.")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
            )
            final_response = chat_completion.choices[0].message.content.strip()
            print("Final response received.")
            self._update_status("Final response received.")
            
            # Simplify the final response
            simplified_response = self.simplify_response(final_response)
            return simplified_response
        except Exception as e:
            error_message = f"Error in final_agent: {e}"
            print(error_message)
            self._update_status(error_message)
            return f"Error: {e}"
    def simplify_response(self, final_response):
        """
        Simplify the final response to make it more concise.

        Args:
            final_response (str): The original final response.

        Returns:
            str: The simplified final response.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a concise assistant. Your goal is to simplify responses without losing the main message.",
            },
            {
                "role": "user",
                "content": f"Please simplify the following response: {final_response} and only return the response, no context, just the response as if you are sending it to the intended user.",
            }
        ]
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
            )
            simplified_response = chat_completion.choices[0].message.content.strip()
            return simplified_response
        except Exception as e:
            error_message = f"Error in simplify_response: {e}"
            print(error_message)
            return final_response  # Fallback to original response if there's an error



    def aurora_run_conversation(self, user_prompt):
        """
        Run a conversation with the assistant to handle tool calls.

        Args:
            user_prompt (str): The user prompt to be processed.

        Returns:
            str: The final response generated after processing the prompt.
        """
        self.chat_history.append({"role": "user", "content": user_prompt})
        messages = [
            {
                "role": "system",
                "content": "You are a function calling LLM that uses the data extracted from various functions to provide detailed responses to the user. You are Aurora's function calling lobe, responsible for handling tool-specific tasks and returning the result. You can interact with the tool-use agent to perform a variety of tasks, including browsing the web for live knowledge, using local commands on a Windows 11 PC, and leveraging vision capabilities to assist Aurora in responding to the user prompt. You will guide Aurora in utilizing the tool-use agent effectively to enhance its responses."
            },
        ] + self.chat_history
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "chat_with_tool_use_agent",
                    "description": "Send a prompt to the tool-use agent to handle tool-specific tasks and return the result. This agent can browse the web for live knowledge, use local commands on a windows 11 pc, has vision capabilities, and can help you do a variety of tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to send to the tool-use agent. IE: research the best way to cook a steak., or find the weather in New York City., or find the best way to learn python., or create a new document in the documents folder.",
                            }
                        },
                        "required": ["prompt"],
                    },
                },
            }
        ]
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )
        self._update_status("Running conversation with tool-use agent in brain for Aurora..")
        response_message = response.choices[0].message
        self.chat_history.append(response_message)
        self._update_status("First response received from Aurora.")
        time.sleep(10)
        tool_calls = response_message.tool_calls
        time.sleep(10)
        self._update_status("Checking for tool calls.")
        if tool_calls:
            available_functions = {
                "chat_with_tool_use_agent": self.function_calling.run_conversation,
            }
            self._update_status("Running tool calls.")
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                self._update_status(f"Running tool call: {function_name}")
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(function_args['prompt'])
                self._update_status("Tool call completed.")
                self.chat_history.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                time.sleep(10)  # Sleep for 10 seconds between function calls
            second_response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=self.chat_history
            )
            self._update_status("Second response received from Aurora.")
            time.sleep(10)  # Sleep for 10 seconds between responses
            self.chat_history.append(second_response.choices[0].message)
            return second_response.choices[0].message.content
        self.chat_history.append(response_message)
        return response_message.content

    def central_processing_agent(self, prompt):
        """
        The central processing function that coordinates the entire process of handling the user prompt.

        Args:
            prompt (str): The user prompt to be processed.

        Returns:
            str: The final response generated after processing the prompt.
        """
        print("Starting central processing agent.")
        self._update_status("Starting central processing agent.")
        try:
            # Run the function calling conversation first
            fc_response = self.aurora_run_conversation(prompt)
            self.add_to_memory(fc_response)  # Save the tool response to memory
            time.sleep(3)  # Additional sleep to avoid rate limits
            self._update_status("Tool response added to memory.")
            # Generate embedding for the tool response
            fc_response_embedding = self.generate_embedding(fc_response)
            time.sleep(3)  # Additional sleep to avoid rate limits
            self._update_status("Embedding generated for tool response.")
            # Retrieve memory using the tool response embedding
            memory_context = self.retrieve_relevant_memory(fc_response_embedding)
            memory_context = " ".join(memory_context)[:1000]  # Limit context to 1,000 tokens
            self._update_status("Memory context retrieved.")
            # Process the normal flow after the function call
            self.start_lobes(prompt, memory_context)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            time.sleep(3)  # Additional sleep to ensure rate limits are respected
            final_thought = self.final_agent(prompt, analyzed_responses)
            self._update_status("Final response generated.")
            print("Central processing agent completed.")
            self._update_status("Central processing agent completed.")
            if self.tts_enabled:
                summarized_for_tts = self.final_agent(f"make this easy for TTS and remove anything that isn't Natural language: {final_thought}", analyzed_responses)
                text_to_speech(summarized_for_tts)
            return final_thought
        except Exception as e:
            print(f"Error in central_processing_agent: {e}")
            self._update_status(f"Error in central_processing_agent: {e}")
            return f"Error: {e}"

