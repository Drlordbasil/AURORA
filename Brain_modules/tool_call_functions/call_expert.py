from openai import OpenAI


def call_expert(expertise, question, progress_callback=None):
    progress_callback(f"Calling {expertise} expert to answer the question: {question}")
    messages = [
        {"role": "system", "content": f"You are a {expertise} expert, you will answer these questions only focusing on {expertise}."},
        {"role": "user", "content": question},
    ]
    openai = OpenAI(base_url="http://localhost:11434/v1")
    response = openai.chat.completions.create(
        model="llama3:instruct",
        messages=messages,
        max_tokens=1000,
    )
    response_text = response.choices[0].message.content
    print(response_text)
    return response_text

# call_expert("math", "What is the integral of x^2?")
