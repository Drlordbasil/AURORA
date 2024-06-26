import json
import time
from typing import Dict, Any

from Brain_modules.llm_api_calls import LLM_API_Calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.image_vision import ImageVision
from Brain_modules.lobes_processing import LobesProcessing
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
        self.chat_history = []
        self.last_response = ""
        print(f"Brain initialization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        return "enabled" if self.tts_enabled else "disabled"

    def process_input(self, user_input: str, progress_callback: callable) -> str:
        try:
            progress_callback("Initiating cognitive processes...")
            
            initial_response = self._get_initial_response(user_input, progress_callback)
            lobe_responses = self._process_lobes(user_input, initial_response, progress_callback)
            memory_context = self._integrate_memory(user_input, initial_response, lobe_responses)
            sentiment = analyze_sentiment(user_input)
            
            final_response = self._generate_final_response(
                user_input, initial_response, lobe_responses, memory_context, sentiment
            )
            
            progress_callback("Cognitive processing complete. Formulating response...")
            return final_response
        except Exception as e:
            error_message = f"Cognitive error encountered: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            progress_callback(error_message)
            return error_message

    def _get_initial_response(self, user_input: str, progress_callback: callable) -> str:
        progress_callback("Engaging primary language model...")
        initial_prompt = self._construct_initial_prompt(user_input)
        return self.api_calls.chat(initial_prompt)

    def _process_lobes(self, user_input: str, initial_response: str, progress_callback: callable) -> Dict[str, Any]:
        lobe_responses = {}
        for lobe_name, lobe in self.lobes_processing.lobes.items():
            progress_callback(f"Activating {lobe_name} neural pathway...")
            response = lobe.process(user_input)
            lobe_responses[lobe_name] = response
        return lobe_responses

    def _integrate_memory(self, user_input: str, initial_response: str, lobe_responses: Dict[str, Any]) -> str:
        combined_input = f"{user_input}\n{initial_response}\n{json.dumps(lobe_responses)}"
        embedding = generate_embedding(combined_input, self.embeddings_model, self.collection, self.collection_size)
        add_to_memory(combined_input, self.embeddings_model, self.collection, self.collection_size)
        return " ".join(retrieve_relevant_memory(embedding, self.collection))

    def _generate_final_response(self, user_input: str, initial_response: str, 
                                 lobe_responses: Dict[str, Any], memory_context: str, sentiment: float) -> str:
        context = self._construct_final_prompt(user_input, initial_response, lobe_responses, memory_context, sentiment)
        final_response = self.api_calls.chat(context)
        self.last_response = final_response
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": final_response})
        return final_response

    def _construct_initial_prompt(self, user_input: str) -> str:
        return f"""As AURORA, an advanced AI with multi-faceted cognitive capabilities, 
        analyze the following context and user input to generate an initial response to the user_input section with a tool use
        as a function calling LLM the tool use function with the user_input as the argument:
        ###TOOLS###
        tools: {tools}
        ###END TOOLS###

        #######user_input#######
        User Input: "{user_input}"
        ######## end_user_input#######



        your tool call:
        """

    def _construct_final_prompt(self, user_input: str, initial_response: str, 
                                lobe_responses: Dict[str, Any], memory_context: str, sentiment: float) -> str:
        return f"""As AURORA, an advanced AI with multi-faceted cognitive capabilities, synthesize the following information to formulate a comprehensive response:

User Input: "{user_input}"

tool use results: {initial_response}

Lobe Processing Results:
{json.dumps(lobe_responses, indent=2)}

Relevant Memory Context: {memory_context}

Detected Sentiment: {sentiment}

Based on this information, generate a response that addresses the user's input comprehensively. Your response should:

1. Directly address the user's main point or question
2. Incorporate relevant insights from the lobe processing results
3. Utilize any pertinent information from the memory context
4. Adjust your tone based on the detected sentiment
5. If necessary, suggest or initiate the use of additional tools or processes to better assist the user

Remember to maintain a coherent narrative throughout your response, ensuring that all parts contribute to a unified and helpful answer. If you need to use any tools or perform additional actions, incorporate them naturally into your response.

Example Structure:
1. Acknowledgment of the user's input
2. Main response addressing the core issue
3. Integration of relevant lobe insights and memory context
4. Suggestions for further actions or tool usage if applicable
5. Conclusion or follow-up question to ensure user satisfaction
Be as friendly and conversationally concise and to the point but also informative as possible in your response.
use emojis to make the response more engaging and human-like

#####BEGIN_USER_input_TO_YOU#####
User Input: "{user_input}"
#####END_USER_input_TO_YOU#####

#####BEGIN_YOU_TO_USER#####
Begin your response now:
"""

    def get_detailed_info(self):
        try:
            detailed_info = {
                "chat_history": self.chat_history,
                "tts_enabled": self.tts_enabled
            }
            return json.dumps(detailed_info, indent=2)
        except Exception as e:
            return f"Error retrieving detailed info: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"