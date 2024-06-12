import threading
from queue import Queue
from groq import Groq
import logging
import ollama
import chromadb
import time
from utilities import setup_logging, setup_embedding_collection
from final_agent_persona import FinalAgentPersona

class Brain:
    def __init__(self, api_key):
        print("Initializing Brain with API key.")
        self.client = Groq(api_key=api_key)
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
        print("Adding to memory.")
        response = ollama.embeddings(model=self.embeddings_model, prompt=text)
        embedding = response["embedding"]
        self.collection.add(
            ids=[str(self.collection_size)],
            embeddings=[embedding],
            documents=[text]
        )
        self.collection_size += 1
        print("Memory added.")

    def generate_embedding(self, text):
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
        print("Retrieving relevant memory.")
        results = self.collection.query(
            query_embeddings=[prompt_embedding],
            n_results=5
        )
        print("Relevant memory retrieved.")
        return [doc for doc in results['documents'][0]]

    def lobe_agent(self, lobe_name, user_prompt, memory_context):
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
        print("Starting lobes.")
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

    def process_responses(self):
        print("Processing responses.")
        for thread in self.threads:
            thread.join()

        aggregated_responses = {}
        while not self.responses.empty():
            lobe_name, response = self.responses.get()
            aggregated_responses[lobe_name] = response
        print("Responses processed.")
        return aggregated_responses

    def analyze_responses(self, responses):
        print("Analyzing responses.")
        # Log the responses for later analysis
        for lobe, response in responses.items():
            logging.info(f"{lobe}: {response}")
        print("Responses analyzed.")
        return responses

    def final_agent(self, user_prompt, aggregated_responses):
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
        print("Starting central processing agent.")
        self.add_to_memory(prompt)
        time.sleep(1)  # Additional sleep to avoid rate limits
        self.start_lobes(prompt)
        responses = self.process_responses()
        analyzed_responses = self.analyze_responses(responses)
        time.sleep(1)  # Additional sleep to ensure rate limits are respected
        final_thought = self.final_agent(prompt, analyzed_responses)
        print("Central processing agent completed.")
        return final_thought
