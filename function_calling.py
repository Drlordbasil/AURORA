# import json
# import subprocess
# import os
# import time
# import requests
# from groq import Groq
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# from nltk.tokenize import sent_tokenize
# from textblob import TextBlob
# from image_vision import ImageVision
# from pdfminer.high_level import extract_text
from kivy.clock import Clock
from llm_api_calls import LLM_API_Calls
class FunctionCalling:
    # MODEL = 'llama3-70b-8192'

    def __init__(self, status_update_callback):
        #  self.client = ""#Groq(api_key=api_key)
        #  self.image_vision = ""#ImageVision() 
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

