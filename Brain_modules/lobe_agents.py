import threading
import time

def lobe_agent(client, lobe_name, user_prompt, memory_context, sentiment, responses, update_status):
    """
    Execute a lobe agent to process the user prompt within the context of the specified lobe.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": f"You are the {lobe_name} lobe of AURORA. Your role is to provide clear and concise guidance based on the user prompt, memory context, and sentiment analysis. Focus on the specific function of the {lobe_name} lobe to offer relevant insights and suggestions to assist AURORA in formulating a direct and cohesive response.",
            },
            {
                "role": "user",
                "content": f"Message from the user: {user_prompt}\n\nMemory Context: {memory_context}\n\nSentiment: {sentiment}\n\n### Provide clear and concise guidance from the perspective of the {lobe_name} lobe. Focus on your specific function and offer relevant insights and suggestions. ###",
            },
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=1,
        )
        response = chat_completion.choices[0].message.content
        responses.put((lobe_name, response))
        time.sleep(1)
    except Exception as e:
        error_message = f"Error in lobe_agent for {lobe_name}: {e}"
        responses.put((lobe_name, f"Error: {e}"))

def start_lobes(client, lobes, prompt, memory_context, sentiment, responses, update_status):
    """
    Start the processing of user prompt by each lobe.
    """
    try:
        threads = []
        for lobe_name in lobes.keys():
            thread = threading.Thread(target=lobe_agent, args=(client, lobe_name, prompt, memory_context, sentiment, responses, update_status))
            thread.start()
            threads.append(thread)
            time.sleep(1)
        for thread in threads:
            thread.join()
    except Exception as e:
        raise Exception(f"Error starting lobes: {e}")
