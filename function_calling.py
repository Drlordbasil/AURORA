
from kivy.clock import Clock
from llm_api_calls import LLM_API_Calls
class FunctionCalling:


    def __init__(self, status_update_callback):

         self.status_update_callback = status_update_callback

    def _update_status(self, message):
        """
        Update the GUI with the current status message.

        Args:
            message (str): The status message to be displayed on the GUI.
        """
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)


    def run_conversation(self, user_prompt):
        self._update_status("Running conversation")
        system_prompt = "you are a function calling LLM"
        prompt = user_prompt
        llm_api = LLM_API_Calls()
        response = llm_api.chat(system_prompt=system_prompt, prompt=prompt)
        return response

