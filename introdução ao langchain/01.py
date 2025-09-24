import os
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

# llm = GoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

# llm_response = llm.invoke(
#     'Quem foi Alan Turing?',
#     config={'max_output_tokens': 500}
# )

# print(llm_response)

chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

messages = [
    SystemMessage(content='Você é um assistente que fornece informações sobre figuras históricas.'),
    HumanMessage(content='Quem foi Alan Turing?'),
]

# 'invoke' também funciona da mesma forma aqui
response_chat = chat_model.invoke(messages)

print(response_chat)
print(response_chat.content)
