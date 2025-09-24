import os
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

template = '''
Traduza o texto {idioma1} para o {idioma2}:
{texto}
'''

prompt_template = PromptTemplate.from_template(
    template=template,
)

prompt = prompt_template.format(
    idioma1='português',
    idioma2='inglês',
    texto='Boa tarde, como você está?'
)

response_chat = chat_model.invoke(prompt)
print(response_chat.content)
