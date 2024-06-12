# main.py file for brain.py to properly be used in a chatloop.
import os
from brain import Brain

def chatbot_loop(api_key):
    print("Starting chatbot loop.")
    brain = Brain(api_key)
    while True:
        prompt = input("Send message: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        response = brain.central_processing_agent(prompt)
        print("Response:", response)

if __name__ == "__main__":
    print("Checking for API key.")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
    else:
        chatbot_loop(api_key)
