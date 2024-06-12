# brain.py

import os
import threading
from queue import Queue
from groq import Groq
import logging
import ollama
import chromadb
import time
from utilities import setup_logging, setup_embedding_collection
from final_agent_persona import FinalAgentPersona
from function_calling import FunctionCalling  # Import FunctionCalling

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
        self.function_caller = FunctionCalling(api_key, self.add_to_memory)  # Initialize FunctionCalling with add_to_memory
        self.embeddings_model = "mxbai-embed-large"
        self.collection, self.collection_size = setup_embedding_collection()
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
                    "content": f"You are only a {self.lobes[lobe_name]}, it doesnt mean you are not important though, you guide aurora as the human brain guides the body.",
                },
                {"role": "user", "content": f"[user_prompt]Message from the user:{user_prompt}[/user_prompt] {memory_context} ### only provide thoughts to give to Aurora as Auroras inner thoughts ONLY ###"},
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
        combined_thoughts = " ".join(f"[{lobe}] {response}" for lobe, response in aggregated_responses.items())

        messages = [
            {
                "role": "system",
                "content": f"You are AURORA, an entity that uses its lobes like a human does subconsciously. [lobe_context]##These are your thoughts, don't reply to them## {combined_thoughts} {FinalAgentPersona.user_info}[/lobe_context] Remember to keep your thoughts to yourself.",
            },
            {
                "role": "user",
                "content": f"[user_prompt]{user_prompt}[/user_prompt] Only respond to what this user prompt is asking for. Dont include thoughts or past questions unless relevant.",
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
            self.add_to_memory(prompt)
            time.sleep(1)  # Additional sleep to avoid rate limits
            self.start_lobes(prompt)
            responses = self.process_responses()
            analyzed_responses = self.analyze_responses(responses)
            time.sleep(1)  # Additional sleep to ensure rate limits are respected
            thoughts = self.final_agent(prompt, analyzed_responses)

            # Integrate FunctionCalling to enhance final response
            final_response = self.function_caller.run_conversation(thoughts)

            print("Central processing agent completed.")
            return final_response
        except Exception as e:
            print(f"Error in central_processing_agent: {e}")
            return f"Error: {e}"

if __name__ == "__main__":
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
    else:
        brain = Brain(api_key)
        user_prompt = "Tell me about the latest advancements in AI."
        print(brain.central_processing_agent(user_prompt))
