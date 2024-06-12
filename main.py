# main.py

import os
from brain import Brain

def chatbot_loop(api_key):
    """
    The main loop for the chatbot. It continuously prompts the user for input
    and processes the input through the Brain class until the user decides to exit.

    Args:
        api_key (str): The API key for initializing the Brain class.
    """
    print("Starting chatbot loop.")
    brain = Brain(api_key)
    while True:
        prompt = input("Send message: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        try:
            response = brain.central_processing_agent(prompt)
            print("Response:", response)
        except Exception as e:
            print(f"Error processing prompt: {e}")

if __name__ == "__main__":
    """
    Main entry point for the script. It checks for the GROQ_API_KEY environment variable
    and starts the chatbot loop if the API key is found.
    """
    print("Checking for API key.")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
    else:
        chatbot_loop(api_key)
