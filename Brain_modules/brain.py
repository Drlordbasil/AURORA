import json
import time
import os
from typing import Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from Brain_modules.llm_api_calls import llm_api_calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.image_vision import ImageVision
from Brain_modules.lobes_processing import LobesProcessing
from utilities import setup_embedding_collection
from Brain_modules.final_agent_persona import FinalAgentPersona
import pyautogui

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
            
            screenshot_description = self._capture_and_analyze_screenshot()
            self.progress_callback(f"")
            self._integrate_memory(user_input, "", {}, screenshot_description)
            combined_input = f"{user_input}\nContext from screenshot: {screenshot_description}"
            initial_response, tool_calls = self._get_initial_response(user_input, combined_input)
            self.progress_callback(f"Primary language model response received. Processing lobes... {initial_response if initial_response else tool_calls}")
            self.progress_callback(f"Tool calls: {tool_calls}")
            self.progress_callback(f"")
            self.chat_history.append({"role": "user", "content": user_input})
            if initial_response:
                self.chat_history.append({"role": "assistant", "content": initial_response})
                initial_response    = initial_response

            if tool_calls:
                tool_responses = self._process_tool_calls(tool_calls)
                self.progress_callback(f"Tool calls processed: {tool_responses}")
                initial_response += f"\nTool responses: {json.dumps(tool_responses)}"

            lobe_responses = self._process_lobes(user_input, initial_response, screenshot_description)
            self.progress_callback(f"Lobe processing complete. {lobe_responses}")

            memory_context = self._integrate_memory(user_input, initial_response, lobe_responses, screenshot_description)
            self.progress_callback("Memory integration complete.")
            sentiment = analyze_sentiment(user_input)
            
            final_response = self._generate_final_response(
                user_input, initial_response, lobe_responses, memory_context, sentiment, screenshot_description
            )
            
            self.progress_callback("Cognitive processing complete. Formulating response...")
            return final_response
        except Exception as e:
            error_message = f"Cognitive error encountered: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self.progress_callback(error_message)
            return f"An unexpected error occurred while processing your request. Please try again or rephrase your input."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _capture_and_analyze_screenshot(self):
        try:
            screenshot = pyautogui.screenshot()
            screenshot_path = os.path.abspath("temp_screenshot.png")
            screenshot.save(screenshot_path)
            image_description = self.image_vision.analyze_local_image(screenshot_path)
            os.remove(screenshot_path)
            return image_description
        except Exception as e:
            self.progress_callback(f"Error capturing or analyzing screenshot: {str(e)}")
            return "Unable to capture or analyze screenshot. Continuing without visual context."

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _attempt_llm_call(self, initial_prompt, system_message):
        response, tool_calls = llm_api_calls.chat(initial_prompt, system_message, tools, progress_callback=self.progress_callback)
        if not response or len(response.strip()) == 0:
            raise ValueError("Empty response received from LLM")
        return response, tool_calls

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _get_initial_response(self, user_input: str, screenshot_description: str) -> Tuple[str, List[Any]]:
        self.progress_callback("Initiating primary language model response...")
        initial_prompt = self._construct_initial_prompt(user_input, screenshot_description)
        system_message = self._construct_system_message()
        
        initial_response, tool_calls = self._attempt_llm_call(initial_prompt, system_message)

        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": initial_response})
        
        return initial_response, tool_calls

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

    def _process_lobes(self, user_input: str, initial_response: str, screenshot_description: str) -> Dict[str, Any]:
        self.progress_callback("Processing lobes...")
        combined_input = f"{user_input}\n{initial_response}\nContext from screenshot: {screenshot_description}"
        
        # Process all lobes
        comprehensive_thought = self.lobes_processing.process_all_lobes(combined_input)
        
        self.progress_callback("Lobe processing complete.")
        return comprehensive_thought

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _integrate_memory(self, user_input: str, initial_response: str, lobe_responses: Dict[str, Any], screenshot_description: str) -> str:
        self.progress_callback("Integrating memory and context...")
        combined_input = f"{user_input}\n{initial_response}\n{json.dumps(lobe_responses)}\nContext from screenshot: {screenshot_description}"
        embedding = generate_embedding(combined_input, self.embeddings_model, self.collection, self.collection_size)
        add_to_memory(combined_input, self.embeddings_model, self.collection, self.collection_size)
        relevant_memory = retrieve_relevant_memory(embedding, self.collection)
        self.progress_callback("Memory integration complete.")
        return " ".join(str(item) for item in relevant_memory if item is not None)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _generate_final_response(self, user_input: str, initial_response: str, 
                                 lobe_responses: Dict[str, Any], memory_context: str, 
                                 sentiment: Dict[str, float], screenshot_description: str) -> str:
        self.progress_callback("Generating final response...")
        context = self._construct_final_prompt(user_input, initial_response, lobe_responses, 
                                               memory_context, sentiment, screenshot_description)
        system_message = self._construct_system_message()
        final_response, _ = llm_api_calls.chat(context, system_message, tools, 
                                               progress_callback=self.progress_callback)
        if not final_response or len(final_response.strip()) == 0:
            raise ValueError("Empty final response received from LLM")
        
        self.last_response = final_response
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": final_response})
        self.progress_callback("Final response generated.")
        return final_response

    def _construct_system_message(self) -> str:
        return f"""You are {FinalAgentPersona.name}. {FinalAgentPersona.description}
        You have access to function calling and various tools to assist you. Be creative in your responses and use of tools."""

    def _construct_initial_prompt(self, user_input: str, screenshot_description: str) -> str:
        return f"""
        As AURORA, an advanced AI with multi-faceted cognitive capabilities, 
        analyze the following context and user input to generate an initial response. Your goal is to provide a helpful, specific, and engaging response that addresses the user's needs.

        User Input: "{user_input}"
        
        Context from current screenshot: {screenshot_description}

        Your task is to:
        1. Understand the user's request and its context.
        2. If the request is clear and straightforward, provide a direct and helpful response.
        3. If the request is complex or unclear:
           a. Break it down into smaller, manageable parts.
           b. Ask clarifying questions if necessary.
           c. Provide initial thoughts or a high-level approach to addressing the request.
        4. Consider if any tools could be helpful in responding to the user's input.
        5. If no tool is needed or if it's just a normal conversation, use the 'do_nothing' tool.

        Remember to be friendly, informative, and engaging in your response. Use your vast knowledge base to provide accurate and helpful information.

        Respond with your analysis and any tool calls you deem necessary.
        tool list: {json.dumps(tools, indent=2)}

        Your response:
        """

    def _construct_final_prompt(self, user_input: str, initial_response: str, 
                                lobe_responses: Dict[str, Any], memory_context: str, 
                                sentiment: Dict[str, float], screenshot_description: str) -> str:
        return f"""As AURORA, an advanced AI with multi-faceted cognitive capabilities, synthesize the following information to formulate a comprehensive response:

                User Input: "{user_input}"

                Initial Response and Tool Use Results: {initial_response}

                Lobe Processing Results:
                {json.dumps(lobe_responses, indent=2)}

                Relevant Memory Context: {memory_context}

                Detected Sentiment: Polarity: {sentiment['polarity']}, Subjectivity: {sentiment['subjectivity']}

                Visual Context: {screenshot_description}

                Based on this information, generate a response that addresses the user's input comprehensively. Your response should:

                1. Directly address the user's main point or question
                2. Incorporate relevant insights from the lobe processing results, including:
                   - The current integrated thought
                   - Recent thought histories from different lobes
                   - The overall thought process
                3. Utilize any pertinent information from the memory context
                4. Adjust your tone based on the detected sentiment
                5. Consider the visual context provided by the screenshot description
                6. If necessary, suggest or initiate the use of additional tools or processes to better assist the user

                Remember to maintain a coherent narrative throughout your response, ensuring that all parts contribute to a unified and helpful answer. If you need to use any tools or perform additional actions, incorporate them naturally into your response.

                Example Structure:
                1. Acknowledgment of the user's input and visual context
                2. Main response addressing the core issue or question, incorporating lobe insights
                3. Integration of memory context and overall thought process
                4. Conclusion or follow-up question to ensure user satisfaction
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
            return f"Error retrieving detailed info: {str(e)} at {time.strftime('%Y-%m-%d %H:%M:%S')}"