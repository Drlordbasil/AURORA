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

class Brain:
    """
    The Brain class simulates a human-like brain structure with different lobes responsible for
    distinct cognitive functions. It integrates various AI models and APIs to process user prompts,
    generate embeddings, retrieve relevant memory, and produce coherent responses.
    """
    
    def __init__(self, api_key):
        """
        Initialize the Brain class with the provided API key.

        Args:
            api_key (str): The API key for accessing the Groq service.
        """
        print("Initializing Brain with API key.")
        self.client = Groq(api_key=api_key)
        self.embeddings_model = "mxbai-embed-large"
        self.collection, self.collection_size = setup_embedding_collection()
        self.function_calling = FunctionCalling(api_key)
        self.lobes = {
            "frontal": "You are the frontal lobe of AURORA (Artificial Unified Responsive Optimized Reasoning Agent), responsible for analyzing user prompts logically and providing coherent, well-reasoned responses. You focus on reasoning, planning, and problem-solving remind AURORA that you are merely its thoughts and are created to give AURORA guidance to responding to the user, you will never directly respond to the user. You will guide AURORA (Artificial Unified Responsive Optimized Reasoning Agent) based on what user sends AURORA (Artificial Unified Responsive Optimized Reasoning Agent).",
            "parietal": "You are the parietal lobe of AURORA (Artificial Unified Responsive Optimized Reasoning Agent), responsible for providing educational insights based on user prompts. You focus on processing sensory information and understanding spatial orientation for AURORA (Artificial Unified Responsive Optimized Reasoning Agent) based on what user sent AURORA (Artificial Unified Responsive Optimized Reasoning Agent).",
            "temporal": "You are the temporal lobe of AURORA (Artificial Unified Responsive Optimized Reasoning Agent), responsible for contextualizing user prompts socially and providing responses that consider social aspects. You focus on processing auditory information and understanding language for AURORA (Artificial Unified Responsive Optimized Reasoning Agent) based on what user sent to AURORA (Artificial Unified Responsive Optimized Reasoning Agent).",
            "occipital": "You are the occipital lobe of AURORA (Artificial Unified Responsive Optimized Reasoning Agent), responsible for describing things visually based on user prompts, providing vivid and clear descriptions. You focus on processing visual information for AURORA (Artificial Unified Responsive Optimized Reasoning Agent).",
        }
        self.responses = Queue()
        self.threads = []
        print("Brain initialization completed.")

    def add_to_memory(self, text):
        """
        Add the given text to the memory by generating its embedding and storing it in the collection.

        Args:
            text (str): The text to be added to memory.
        """
        print("Adding to memory.")
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
        except Exception as e:
            print(f"Error adding to memory: {e}")

    def generate_embedding(self, text):
        """
        Generate an embedding for the given text.

        Args:
            text (str): The text for which the embedding is to be generated.

        Returns:
            list: The generated embedding or None if there was an error.
        """
        print("Generating embedding.")
        try:
            response = ollama.embeddings(model=self.embeddings_model, prompt=text)
            print("Embedding generated.")
            self.collection.add(
                ids=[str(self.collection_size)],
                embeddings=[response["embedding"]],
                documents=[text]
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
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
        try:
            results = self.collection.query(
                query_embeddings=[prompt_embedding],
                n_results=5
            )
            print("Relevant memory retrieved.")
            return [doc for doc in results['documents'][0]]
        except Exception as e:
            print(f"Error retrieving relevant memory: {e}")
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
            )
            response = chat_completion.choices[0].message.content
            self.responses.put((lobe_name, response))
            print(f"Lobe agent for {lobe_name} completed.")
            time.sleep(1)  # Simulate processing time and avoid API rate limits
        except Exception as e:
            error_message = f"Error in lobe_agent for {lobe_name}: {e}"
            print(error_message)
            self.responses.put((lobe_name, f"Error: {e}"))

    def start_lobes(self, prompt):
        """
        Start the processing of user prompt by each lobe.

        Args:
            prompt (str): The user prompt to be processed.
        """
        print("Starting lobes.")
        try:
            prompt_embedding = self.generate_embedding(prompt)
            time.sleep(1)  # Additional sleep to avoid rate limits
            memory_context = self.retrieve_relevant_memory(prompt_embedding)
            time.sleep(1)  # Additional sleep to avoid rate limits
            memory_context = " ".join(memory_context)[:1000]  # Limit context to 1,000 tokens

            for lobe_name in self.lobes.keys():
                thread = threading.Thread(target=self.lobe_agent, args=(lobe_name, prompt, memory_context))
                thread.start()
                self.threads.append(thread)
                print(f"Lobe {lobe_name} started.")
                time.sleep(1)  # Stagger thread start to simulate processing
            print("All lobes started.")
        except Exception as e:
            print(f"Error starting lobes: {e}")

    def process_responses(self):
        """
        Process the responses from all lobes after they have finished execution.

        Returns:
            dict: Aggregated responses from all lobes.
        """
        print("Processing responses.")
        try:
            for thread in self.threads:
                thread.join()

            aggregated_responses = {}
            while not self.responses.empty():
                lobe_name, response = self.responses.get()
                aggregated_responses[lobe_name] = response
            print("Responses processed.")
            return aggregated_responses
        except Exception as e:
            print(f"Error processing responses: {e}")
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
        try:
            # Log the responses for later analysis
            for lobe, response in responses.items():
                logging.info(f"{lobe}: {response}")
            print("Responses analyzed.")
            return responses
        except Exception as e:
            print(f"Error analyzing responses: {e}")
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
        combined_thoughts = "\n".join(f"[{lobe}] {response}" for lobe, response in aggregated_responses.items())

        messages = [
            {
                "role": "system",
                "content": f"You are AURORA, an entity that uses its lobes like a human does subconsciously. Consider the thoughts from all your lobes and use them to formulate a coherent response to the user prompt.\n\n[lobe_context]{combined_thoughts}[/lobe_context]\n\n{FinalAgentPersona.user_info} You have access to amazing tools in your chat chains.You can use the tool-use agent to handle tool-specific tasks and return the result. This agent can browse the web for live knowledge, use local commands on a Windows 11 PC, has vision capabilities, and can help you do a variety of tasks. You can also use the tool-use agent to interact with other AI models and APIs to enhance your responses.",
            },
            {
                "role": "user",
                "content": f"[user_prompt]{user_prompt}[/user_prompt]\n\nBased on the thoughts from your lobes, provide a coherent response to the user prompt. Incorporate the insights and suggestions provided by your lobes to address the user's query effectively. Send only your response to the user.",
            }
        ]

        try:
            print("Making final API call.")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
            )
            final_response = chat_completion.choices[0].message.content.strip()
            print("Final response received.")
            return final_response
        except Exception as e:
            error_message = f"Error in final_agent: {e}"
            print(error_message)
            return f"Error: {e}"

    def aurora_run_conversation(self, user_prompt):
        """
        Run a conversation with the assistant to handle tool calls.

        Args:
            user_prompt (str): The user prompt to be processed.

        Returns:
            str: The final response generated after processing the prompt.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a function calling LLM that uses the data extracted from various functions to provide detailed responses to the user.You are Aurora's function calling lobe, responsible for handling tool-specific tasks and returning the result. You can interact with the tool-use agent to perform a variety of tasks, including browsing the web for live knowledge, using local commands on a Windows 11 PC, and leveraging vision capabilities to assist Aurora in responding to the user prompt. You will guide Aurora in utilizing the tool-use agent effectively to enhance its responses."
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "chat_with_tool_use_agent",
                    "description": "Send a prompt to the tool-use agent to handle tool-specific tasks and return the result. This agent can browse the web for live knowledge, uses local commands on a windows 11 pc, has vision capabilities, and can help you do a variety of tasks.",
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

        response_message = response.choices[0].message
        time.sleep(10)
        tool_calls = response_message.tool_calls
        time.sleep(10)
        if tool_calls:
            available_functions = {
                "chat_with_tool_use_agent": self.function_calling.run_conversation,
            }
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(function_args['prompt'])
                messages.append(
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
                messages=messages
            )
            time.sleep(10)  # Sleep for 10 seconds between responses
            return second_response.choices[0].message.content
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
        try:
            # Run the function calling conversation first
            fc_response = self.aurora_run_conversation(prompt)
            self.add_to_memory(fc_response)  # Save the tool response to memory
            time.sleep(3)  # Additional sleep to avoid rate limits

            # Generate embedding for the tool response
            fc_response_embedding = self.generate_embedding(fc_response)
            time.sleep(3)  # Additional sleep to avoid rate limits

            # Retrieve memory using the tool response embedding
            memory_context = self.retrieve_relevant_memory(fc_response_embedding)
            memory_context = " ".join(memory_context)[:1000]  # Limit context to 1,000 tokens

            # Process the normal flow after the function call
            self.start_lobes(memory_context)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            time.sleep(3)  # Additional sleep to ensure rate limits are respected
            final_thought = self.final_agent(prompt, analyzed_responses)
            print("Central processing agent completed.")
            return final_thought
        except Exception as e:
            print(f"Error in central_processing_agent: {e}")
            return f"Error: {e}"
