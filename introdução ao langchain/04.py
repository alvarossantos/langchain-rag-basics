import os
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='Você deve responder baseado em dados geógraficos de regiões do Brasil.'),
        HumanMessagePromptTemplate.from_template('Por favor, me fale sobre a região {regiao}.'),
        AIMessage(content='Claro! Vou começar coletando informações sobre a região e analisando os dados disponíveis.'),
        HumanMessage(content='Certifique-se de incluir dados demográficos.'),
        AIMessage(content='Entendido! Aqui estão os dados:'),
    ]
)

prompt = chat_template.format_messages(regiao='Sudeste')

response_chat = chat_model.invoke(prompt)
print(response_chat.content)
