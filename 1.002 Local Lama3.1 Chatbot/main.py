from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = '''
Answer the question below:
Here is conversation history  {context}.
Question: {input}
Answer:
'''

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to AI chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        # Prepare the input for the chain
        input_data = {"context": context, "input": user_input}
        # Stream the response
        print("AI: ", end="")
        for chunk in model.stream(prompt.format(**input_data)):
            print(chunk, end="", flush=True)
        print()  # New line after streaming the full response
        # Update context
        context += f"\nHuman: {user_input}\nAI: {chunk}"

if __name__ == "__main__":
    handle_conversation()


# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate

# template =  '''
# Answer the question below:
# Here is conversation history  {context}.
# Question: {input}
# Answer:
# '''

# model = OllamaLLM(model="llama3.1")
# prompt=ChatPromptTemplate.from_template(template)
# chain = prompt | model
# def handle_conversation():
#     context = ""
#     print("Welcome to AI chatbot! Type 'quit' to exit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "quit":
#             break
#         for chunk in model.stream("Write me a song about sparkling water."):
#             print(chunk, end="", flush=True)
#         result = chain.invoke({"context": context, "input": user_input})
#         print(f"AI: {result}")
#         context += f"\nHuman: {user_input}\nAI: {result}"

# if __name__ == "__main__":
#     handle_conversation()


# result = model.invoke(input="What is the meaning of life?")
# result = chain.invoke({"context": "we previously discussed about spiritual way is good", "input": "What is the meaning of life?"})
# print(result)