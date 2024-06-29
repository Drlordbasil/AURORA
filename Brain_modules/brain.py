import json
import time
from typing import Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from Brain_modules.llm_api_calls import llm_api_calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.image_vision import ImageVision
from Brain_modules.lobes_processing import LobesProcessing
from utilities import setup_embedding_collection
from Brain_modules.final_agent_persona import FinalAgentPersona

class Brain:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        self._initialize()

    def _initialize(self):
        self.progress_callback(f"Initializing Brain at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.tts_enabled = True
        self.collection, self.collection_size = setup_embedding_collection()
        self.image_vision = ImageVision()
        self.lobes_processing = LobesProcessing(self.image_vision)
        self.embeddings_model = "mxbai-embed-large"
        self.chat_history = []
        self.last_response = ""
        self.progress_callback(f"Brain initialization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def toggle_tts(self):
        try:
            self.tts_enabled = not self.tts_enabled
            status = "enabled" if self.tts_enabled else "disabled"
            self.progress_callback(f"TTS toggled to {status}")
            return status
        except Exception as e:
            error_message = f"Error toggling TTS: {str(e)}"
            self.progress_callback(error_message)
            raise

    def process_input(self, user_input: str) -> str:
        try:
            self.progress_callback("Initiating cognitive processes...")
            
            initial_response, tool_calls = self._get_initial_response(user_input)
            self.progress_callback(f"Primary language model response received. Processing lobes... {initial_response}")
            
            if tool_calls:
                tool_responses = self._process_tool_calls(tool_calls)
                self.progress_callback(f"Tool calls processed: {tool_responses}")
                initial_response += f"\nTool responses: {json.dumps(tool_responses)}"

            lobe_responses = self._process_lobes(user_input, initial_response)
            self.progress_callback(f"Lobe processing complete. {lobe_responses}")

            memory_context = self._integrate_memory(user_input, initial_response, lobe_responses)
            self.progress_callback("Memory integration complete.")
            sentiment = analyze_sentiment(user_input)
            
            final_response = self._generate_final_response(
                user_input, initial_response, lobe_responses, memory_context, sentiment
            )
            
            self.progress_callback("Cognitive processing complete. Formulating response...")
            return final_response
        except Exception as e:
            error_message = f"Cognitive error encountered: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self.progress_callback(error_message)
            return f"I apologize, but I encountered an unexpected error while processing your request. Here's what happened: {error_message}. How else can I assist you?"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _attempt_llm_call(self, initial_prompt):
        return llm_api_calls.chat(initial_prompt, "", tools, progress_callback=self.progress_callback)

    def _get_initial_response(self, user_input: str) -> Tuple[str, List[Any]]:
        self.progress_callback("Initiating primary language model response...")
        initial_prompt = self._construct_initial_prompt(user_input)
        
        try:
            initial_response, tool_calls = self._attempt_llm_call(initial_prompt)
        except Exception as e:
            self.progress_callback(f"Error in LLM API call after retries: {str(e)}")
            initial_response = self._generate_fallback_response(user_input)
            tool_calls = []

        if not initial_response or initial_response == "I apologize, but I couldn't generate a response. How else can I assist you?":
            initial_response = self._generate_fallback_response(user_input)

        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": initial_response})
        
        return initial_response, tool_calls

    def _generate_fallback_response(self, user_input: str) -> str:
        words = user_input.lower().split()
        
        if any(word in words for word in ['hi', 'hello', 'hey']):
            return "Hello! I'm here to assist you. What would you like to know or discuss today?"
        
        if any(word in words for word in ['help', 'assist', 'support']):
            return "I'm here to help! Could you please provide more details about what you need assistance with?"
        
        if '?' in user_input:
            return "I understand you've asked a question, but I'm having trouble generating a specific answer. Could you rephrase your question or provide more context?"
        
        if len(words) < 5:
            return f"I see that you've provided a brief input: '{user_input}'. Could you please elaborate on what you'd like to know or discuss?"
        
        return f"I apologize, but I'm having difficulty generating a specific response to your input: '{user_input}'. Could you please provide more context or rephrase your request? I'm here to assist you with a wide range of topics and tasks."

    def _process_tool_calls(self, tool_calls):
        tool_responses = {}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = llm_api_calls.available_functions.get(function_name)
            if function_to_call:
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args, progress_callback=self.progress_callback)
                tool_responses[function_name] = function_response
        return tool_responses

    def _process_lobes(self, user_input: str, initial_response: str) -> Dict[str, Any]:
        lobe_responses = {}
        for lobe_name, lobe in self.lobes_processing.lobes.items():
            self.progress_callback(f"Activating {lobe_name} neural pathway...")
            combined_input = f"{user_input}\n{initial_response if initial_response else ''}"
            response = lobe.process(combined_input)
            lobe_responses[lobe_name] = response
        return lobe_responses

    def _integrate_memory(self, user_input: str, initial_response: str, lobe_responses: Dict[str, Any]) -> str:
        self.progress_callback("Integrating memory and context...")
        combined_input = f"{user_input}\n{initial_response}\n{json.dumps(lobe_responses)}"
        embedding = generate_embedding(combined_input, self.embeddings_model, self.collection, self.collection_size)
        add_to_memory(combined_input, self.embeddings_model, self.collection, self.collection_size)
        relevant_memory = retrieve_relevant_memory(embedding, self.collection)
        self.progress_callback("Memory integration complete.")
        return " ".join(relevant_memory)

    def _generate_final_response(self, user_input: str, initial_response: str, 
                                 lobe_responses: Dict[str, Any], memory_context: str, sentiment: float) -> str:
        self.progress_callback("Generating final response...")
        context = self._construct_final_prompt(user_input, initial_response, lobe_responses, memory_context, sentiment)
        system_message = self._construct_system_message()
        final_response, _ = llm_api_calls.chat(context, system_message, tools, progress_callback=self.progress_callback)
        if not final_response:
            final_response = self._construct_final_prompt(user_input, initial_response, lobe_responses, memory_context, sentiment)
        self.last_response = final_response
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": final_response})
        self.progress_callback("Final response generated.")
        return final_response

    def _construct_system_message(self) -> str:
        return f"""You are {FinalAgentPersona.name}. {FinalAgentPersona.description}
        who has access to function calling and here is which ones you can use and how to get creative"""

    def _construct_initial_prompt(self, user_input: str) -> str:
        return f"""
        As AURORA, an advanced AI with multi-faceted cognitive capabilities, 
        analyze the following context and user input to generate an initial response to the user_input section with a tool use
        as a function calling LLM the tool use function with the user_input as the argument. You directly send your tool calls to yourself in the next response from yourself.

        User Input: "{user_input}"

        Your task is to analyze this input and determine if any tool use is necessary. If needed, use the appropriate tool. If no tool is needed or if it's just a normal conversation, use the 'do_nothing' tool.

        Respond with your analysis and any tool calls you deem necessary.
        tool list: {json.dumps(tools, indent=2)}


your tool calls:
        """

    def _construct_final_prompt(self, user_input: str, initial_response: str, 
                                lobe_responses: Dict[str, Any], memory_context: str, sentiment: float) -> str:
        return f"""As AURORA, an advanced AI with multi-faceted cognitive capabilities, synthesize the following information to formulate a comprehensive response:

                User Input: "{user_input}"

                Initial Response and Tool Use Results: {initial_response}

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
                2. Main response addressing the core issue or question
                3. Conclusion or follow-up question to ensure user satisfaction
                Be as friendly and conversationally concise and to the point but also informative as possible in your response.
                Use emojis to make the response more engaging and human-like.
                If it's a simple greeting or normal conversation, be as conversationally pleasing as possible as the user would expect.
                You have many thoughts, but you can respond as an adult named Aurora.

                Your response:
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